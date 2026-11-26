#include "basin/basin_snapshot.hpp"
#include "basin/forward/basin_forward_solver.hpp"
#include "basin/geometry/basin_geometry.hpp"
#include "basin/io/basin_binary.hpp"
#include "basin/io/basin_forward_options.hpp"
#include "basin/io/basin_statistics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

void write_layer_fields(std::ostream& output, const BasinConfig& cfg,
                        const BasinSnapshot& snapshot) {
    const auto& grid = snapshot.grid;
    const auto geometry = make_basin_geometry(cfg);
    std::vector<double> centers;
    std::vector<ScalarProfile> temperatures;
    for (const auto& column : grid.columns) {
        const auto& first = grid.cells[column.front()];
        centers.push_back(first.center().x);
        auto& profile = temperatures.emplace_back();
        profile.x.push_back(first.z_top);
        profile.values.push_back(cfg.T_surf.at(centers.back()));
        for (const auto i : column) {
            profile.x.push_back(grid.cells[i].center().z);
            profile.values.push_back(snapshot.T[i]);
        }
    }
    for (const auto& layer : cfg.layers) {
        std::vector<double> x;
        std::vector<std::array<double, 2>> ro;
        for (const auto& column : grid.columns) {
            const auto first = std::find_if(column.begin(), column.end(), [&](std::size_t i) {
                return grid.cells[i].layer_id == layer.id;
            });
            if (first == column.end())
                continue;
            const auto last = std::find_if_not(first, column.end(), [&](std::size_t i) {
                return grid.cells[i].layer_id == layer.id;
            });
            const auto& cell = grid.cells[*first];
            const auto top = cell.maturity_top;
            const auto bottom = grid.cells[*(last - 1)].maturity_top + 1;
            x.push_back(cell.center().x);
            ro.push_back({snapshot.Ro[top], snapshot.Ro[bottom]});
        }
        std::size_t upper_column = 0;
        for (std::size_t i = 0; i < layer.x.size(); ++i) {
            const double coordinate = layer.x[i];
            while (upper_column < centers.size() && centers[upper_column] < coordinate)
                ++upper_column;
            const auto left_column = upper_column == 0 ? 0 : upper_column - 1;
            const auto right_column = std::min(upper_column, centers.size() - 1);
            const double wx = left_column == right_column
                                  ? 0.0
                                  : (coordinate - centers[left_column]) /
                                        (centers[right_column] - centers[left_column]);
            const double surface = geometry.boundary_at(0, coordinate);
            const auto temperature = [&](double z) {
                if (z <= surface)
                    return cfg.T_surf.at(coordinate);
                return std::lerp(temperatures[left_column].at(z), temperatures[right_column].at(z),
                                 wx);
            };
            const double top = temperature(layer.top[i]) - 273.15;
            const double bottom = temperature(layer.bottom[i]) - 273.15;
            std::size_t left = 0, right = 0;
            double weight = 0.0;
            if (x.size() > 1) {
                const auto upper = std::lower_bound(x.begin(), x.end(), coordinate);
                right = std::clamp(static_cast<std::size_t>(upper - x.begin()), std::size_t{1},
                                   x.size() - 1);
                left = right - 1;
                weight = (coordinate - x[left]) / (x[right] - x[left]);
            }
            const double ro_top = std::lerp(ro[left][0], ro[right][0], weight);
            const double ro_bottom = std::lerp(ro[left][1], ro[right][1], weight);
            output << cfg.age_ma << ',' << layer.id << ',' << coordinate << ',' << top << ','
                   << bottom << ',' << 0.5 * (top + bottom) << ',' << ro_top << ',' << ro_bottom
                   << ',' << 0.5 * (ro_top + ro_bottom) << '\n';
        }
    }
}

} // namespace

BasinForwardResult BasinForwardSolver::solve(const std::filesystem::path& output,
                                             const BasinObserver& observe) const {
    const auto solve_start = Clock::now();
    auto output_start = solve_start;
    const auto snapshots = output / "snapshots";
    std::filesystem::create_directories(snapshots);
    std::ofstream fields(output / "fields.csv");
    fields << "age_ma,layer_id,x,temperature_top,temperature_bottom,temperature_average,"
              "reflectance_top,reflectance_bottom,reflectance_average\n"
           << std::setprecision(17);
    const double prepare_output_s = elapsed_seconds(output_start);
    BasinForwardResult result = solve(
        [&](const BasinSnapshot& snapshot) {
            const bool final = snapshot.kind == BasinSnapshot::Kind::after_event &&
                               snapshot.epoch + 1 == configs().size();
            if (snapshot.kind == BasinSnapshot::Kind::initial || final ||
                (snapshot.kind == BasinSnapshot::Kind::before_event &&
                 snapshot.epoch + 2 < configs().size())) {
                const auto index = snapshot.kind == BasinSnapshot::Kind::before_event
                                       ? snapshot.epoch + 1
                                       : snapshot.epoch;
                write_layer_fields(fields, configs()[index], snapshot);
            }
            if (settings_.intermediate_snapshots || final) {
                std::ostringstream name;
                name << "snapshot_e" << std::setw(2) << std::setfill('0') << snapshot.epoch + 1
                     << "_s" << std::setw(4) << snapshot.step << ".cgrd";
                write_basin_binary(snapshots / name.str(), snapshot);
            }
            if (observe)
                observe(snapshot);
        },
        true);
    output_start = Clock::now();
    fields.close();
    if (!fields)
        throw std::runtime_error("Failed writing original-grid fields");
    std::ofstream summary(output / "summary.txt");
    write_basin_forward_settings(summary, settings_);
    summary << "timesteps=" << result.timesteps << '\n'
            << "nonlinear_iterations=" << result.nonlinear_iterations << '\n'
            << "linear_iterations=" << result.linear_iterations << '\n';
    if (!summary)
        throw std::runtime_error("Failed writing basin summary");
    summary.close();
    if (!summary)
        throw std::runtime_error("Failed closing basin summary");
    write_basin_intervals(output, result);
    result.timings.output_s += prepare_output_s + elapsed_seconds(output_start);
    result.timings.total_s = elapsed_seconds(solve_start);
    return result;
}
