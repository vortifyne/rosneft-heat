#include "basin/basin_snapshot.hpp"
#include "basin/forward/basin_forward_solver.hpp"
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

void write_layer_fields(std::ostream& output, const BasinConfig& cfg, const BasinSnapshot& snapshot,
                        std::span<const double> node_T) {
    const auto& grid = snapshot.grid;
    for (const auto& layer : cfg.layers) {
        std::vector<double> x;
        std::vector<std::array<double, 4>> values;
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
            const auto top = *first + cell.column;
            const auto bottom = *(last - 1) + cell.column + 1;
            x.push_back(cell.center().x);
            values.push_back({node_T[top] - 273.15, node_T[bottom] - 273.15, snapshot.Ro[top],
                              snapshot.Ro[bottom]});
        }
        for (const double coordinate : layer.x) {
            std::size_t left = 0, right = 0;
            double weight = 0.0;
            if (x.size() > 1) {
                const auto upper = std::lower_bound(x.begin(), x.end(), coordinate);
                right = std::clamp(static_cast<std::size_t>(upper - x.begin()), std::size_t{1},
                                   x.size() - 1);
                left = right - 1;
                weight = (coordinate - x[left]) / (x[right] - x[left]);
            }
            std::array<double, 4> v;
            for (std::size_t i = 0; i < v.size(); ++i)
                v[i] = std::lerp(values[left][i], values[right][i], weight);
            output << cfg.age_ma << ',' << layer.id << ',' << coordinate << ',' << v[0] << ','
                   << v[1] << ',' << 0.5 * (v[0] + v[1]) << ',' << v[2] << ',' << v[3] << ','
                   << 0.5 * (v[2] + v[3]) << '\n';
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
                std::vector<double> node_T(snapshot.Ro.size());
                temperature_nodes(snapshot.grid, index, snapshot.T, node_T);
                write_layer_fields(fields, configs()[index], snapshot, node_T);
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
