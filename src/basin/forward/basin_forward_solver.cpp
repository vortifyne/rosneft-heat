#include "basin/forward/basin_forward_solver.hpp"

#include "basin/basin_snapshot.hpp"
#include "basin/discretization/col_heat_system.hpp"
#include "basin/geometry/basin_geometry_rdp.hpp"
#include "basin/mesh/col_epoch_grid.hpp"
#include "basin/mesh/col_field_transfer.hpp"
#include "basin/physics/easy_ro.hpp"
#include "basin/physics/thermal_properties.hpp"
#include "numerics/time/time_integrator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr double kSecondsPerMa = 1.0e6 * 365.25 * 24.0 * 3600.0;
constexpr FluidProperties kFluid{.lambda = 0.7, .rho = 1040.0, .c = 4184.0};

std::vector<double> lithotype_breakpoints(const BasinConfig& cfg0,
                                          const BasinConfig* cfg1 = nullptr) {
    std::vector<double> result;
    const auto append = [&result](const BasinConfig& cfg) {
        for (const auto& layer : cfg.layers) {
            for (std::size_t i = 1; i < layer.x.size(); ++i) {
                if (layer.lithotype[i - 1] != layer.lithotype[i]) {
                    result.push_back(std::midpoint(layer.x[i - 1], layer.x[i]));
                }
            }
        }
    };
    append(cfg0);
    if (cfg1)
        append(*cfg1);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::span<const double> values(const Vector& v) {
    return {v.native().data(), static_cast<std::size_t>(v.size())};
}

std::vector<double> deposition_times(const ColGrid& grid, std::span<const BasinConfig> configs) {
    std::unordered_map<int, std::pair<double, double>> birth;
    for (std::size_t i = 0; i < configs.size(); ++i) {
        for (const auto& layer : configs[i].layers) {
            const double end = (configs.front().age_ma - configs[i].age_ma) * kSecondsPerMa;
            const double begin =
                i == 0 ? end : (configs.front().age_ma - configs[i - 1].age_ma) * kSecondsPerMa;
            birth.try_emplace(layer.id, begin, end);
        }
    }
    std::vector<double> result(grid.cells.size() + grid.columns.size());
    for (const auto& column : grid.columns) {
        auto first = column.begin();
        while (first != column.end()) {
            const auto& top = grid.cells[*first];
            const auto last = std::find_if_not(first, column.end(), [&](std::size_t index) {
                return grid.cells[index].layer_id == top.layer_id;
            });
            const auto& bottom = grid.cells[*(last - 1)];
            const auto [begin, end] = birth.at(top.layer_id);
            for (auto cell = first; cell != last; ++cell) {
                const double depth = grid.cells[*cell].z_top;
                result[*cell + top.column] =
                    std::lerp(end, begin, (depth - top.z_top) / (bottom.z_bottom - top.z_top));
            }
            result[bottom.index + bottom.column + 1] = begin;
            first = last;
        }
    }
    return result;
}

HeatBoundaryData bc(const BasinConfig& cfg0, const BasinConfig& cfg1, double theta) {
    return {
        .T_surf =
            [&cfg0, &cfg1, theta](Point2D point) {
                return std::lerp(cfg0.T_surf.at(point.x), cfg1.T_surf.at(point.x), theta);
            },
        .q_basal =
            [&cfg0, &cfg1, theta](Point2D point) {
                return std::lerp(cfg0.q_basal.at(point.x), cfg1.q_basal.at(point.x), theta);
            },
    };
}

void for_each_layer_cell(const ColGrid& grid, const auto& visit) {
    for (const auto& column : grid.columns) {
        auto first = column.begin();
        while (first != column.end()) {
            const auto& top = grid.cells[*first];
            const auto last = std::find_if_not(first, column.end(), [&](std::size_t index) {
                return grid.cells[index].layer_id == top.layer_id;
            });
            const double height = grid.cells[*(last - 1)].z_bottom - top.z_top;
            for (auto cell = first; cell != last; ++cell) {
                const auto& value = grid.cells[*cell];
                visit(value, (value.center().z - top.z_top) / height);
            }
            first = last;
        }
    }
}

Vector initial_temperature(const ColGrid& grid, const ScalarProfile& surface, int born) {
    double integral = 0.0;
    for (std::size_t i = 1; i < surface.x.size(); ++i)
        integral +=
            0.5 * (surface.values[i - 1] + surface.values[i]) * (surface.x[i] - surface.x[i - 1]);
    const double mean_surface = integral / (surface.x.back() - surface.x.front());
    Vector T(static_cast<Vector::Index>(grid.cells.size()));
    for (const auto& cell : grid.cells) {
        const double top = grid.cells[grid.top_cells[cell.column]].z_top;
        T[static_cast<Vector::Index>(cell.index)] =
            cell.layer_id == born ? surface.at(cell.center().x)
                                  : mean_surface + 0.03 * (cell.center().z - top);
    }
    return T;
}

void observe_state(const ColGrid& grid, const Vector& T, std::span<const double> I,
                   const EasyRoModel& easy_ro, BasinSnapshot::Kind kind, std::size_t epoch,
                   int step, double age_ma, const BasinObserver& observer, BasinTimings& timings) {
    if (!observer)
        return;
    auto start = Clock::now();
    std::vector<double> Ro(grid.cells.size() + grid.columns.size());
    easy_ro.reflectance(I, Ro);
    timings.easy_ro_s += elapsed_seconds(start);
    start = Clock::now();
    observer({.kind = kind,
              .epoch = epoch,
              .step = step,
              .age_ma = age_ma,
              .grid = grid,
              .T = values(T),
              .I = I,
              .Ro = Ro});
    timings.output_s += elapsed_seconds(start);
}

} // namespace

BasinForwardSolver::BasinForwardSolver(BasinInput input, BasinForwardSettings options)
    : input_(std::move(input)), settings_(std::move(options)), easy_ro_(input_.easy_ro()) {
    const auto configs = input_.configs();
    const auto& settings = settings_;
    if (settings.nonlinear.method != NonlinearMethod::picard || configs.size() < 2 ||
        !(settings.dx > 0.0) || !(settings.dz > 0.0) || !(settings.dz_min > 0.0) ||
        !(settings.dt_ma > 0.0) || !(settings.nonlinear.rtol > 0.0) ||
        !(settings.nonlinear.rtol < 1.0)) {
        throw std::invalid_argument("Invalid basin configurations or settings");
    }
    std::vector<BasinGeometry> geometries;
    geometries.reserve(configs.size());
    for (const auto& cfg : configs)
        geometries.push_back(
            simplify_basin_geometry_rdp(make_basin_geometry(cfg), settings.rdp_tol));
    epochs_.reserve(configs.size() - 1);
    for (std::size_t i = 0; i + 1 < configs.size(); ++i) {
        if (!(configs[i].age_ma > configs[i + 1].age_ma))
            throw std::invalid_argument("basin ages must decrease");
        EpochData data;
        data.mesh = make_col_epoch_grid(geometries[i], geometries[i + 1], settings.dx, settings.dz,
                                        settings.dz_min, settings.axis_rdp_tol,
                                        lithotype_breakpoints(configs[i], &configs[i + 1]));
        data.initial_grid = data.mesh.at(0.0);
        // Use the final layer coordinates: the new cells are compressed at the start.
        data.deposition_times = deposition_times(data.mesh.at(1.0), configs);
        const auto count = data.initial_grid.cells.size();
        data.lithotypes.resize(count);
        data.phi0.resize(count);
        data.phi1.resize(count);
        for (const auto& cell : data.initial_grid.cells) {
            const double x = cell.center().x;
            const auto& layer0 = data.mesh.born_layer_id == cell.layer_id
                                     ? configs[i + 1].layer(cell.layer_id)
                                     : configs[i].layer(cell.layer_id);
            data.lithotypes[cell.index] = layer0.lithotype_at(x);
            const auto& layer1 = configs[i + 1].layer(cell.layer_id);
            data.phi0[cell.index] = {layer0.phi_at(x, 0.0), layer0.phi_at(x, 1.0)};
            data.phi1[cell.index] = {layer1.phi_at(x, 0.0), layer1.phi_at(x, 1.0)};
        }
        epochs_.push_back(std::move(data));
    }
    const auto& geometry = geometries.back();
    const auto knots = settings.axis_rdp_tol
                           ? simplify_basin_geometry_rdp(geometry, *settings.axis_rdp_tol).x
                           : geometry.x;
    final_grid_ = make_col_grid(
        geometry, make_column_axis(knots, settings.dx, lithotype_breakpoints(configs.back())),
        settings.dz);
    initial_temperature_ = initial_temperature(epochs_.front().initial_grid, configs.front().T_surf,
                                               epochs_.front().mesh.born_layer_id.value_or(-1));
}

void BasinForwardSolver::set_parameters(const Vector& parameters) {
    if (parameters.size() != static_cast<Vector::Index>(input_.configs().size()) ||
        !parameters.all_finite())
        throw std::invalid_argument("Expected one finite basal flux per configuration");
    parameters_ = parameters;
}

void BasinForwardSolver::set_intermediate_snapshots(bool enabled) noexcept {
    settings_.intermediate_snapshots = enabled;
}

std::span<const BasinConfig> BasinForwardSolver::configs() const noexcept {
    return input_.configs();
}

const ColGrid& BasinForwardSolver::final_grid() const noexcept {
    return final_grid_;
}

void BasinForwardSolver::final_temperature_nodes(std::span<const double> T,
                                                 std::span<double> node_T) const {
    const auto& cfg = input_.configs().back();
    std::vector<double> lambda(T.size());
    for_each_layer_cell(final_grid_, [&](const ColCell& cell, double depth) {
        const auto& layer = cfg.layer(cell.layer_id);
        const double x = cell.center().x;
        lambda[cell.index] = thermal_properties(T[cell.index], layer.phi_at(x, depth),
                                                input_.lithotype(layer.lithotype_at(x)), kFluid)
                                 .lambda;
    });
    auto boundaries = bc(cfg, cfg, 0.0);
    if (!parameters_.empty()) {
        const double q = parameters_[parameters_.size() - 1];
        boundaries.q_basal = [q](Point2D) { return q; };
    }
    interpolate_horizontal_temperature(final_grid_, T, lambda, boundaries, node_T);
}

BasinForwardResult BasinForwardSolver::solve(const BasinObserver& observe) const {
    const auto solve_start = Clock::now();
    auto grid_start = solve_start;
    BasinForwardResult result;
    const auto& input = input_;
    const auto& settings = settings_;
    const auto configs = input.configs();
    const auto& easy_ro = easy_ro_;
    ColGrid grid = epochs_.front().initial_grid;
    Vector T = initial_temperature_;
    auto I = easy_ro.initial_state(epochs_.front().deposition_times.size());
    result.epochs.reserve(configs.size() - 1);
    TimeIntegrator integrator;
    integrator.set_linear_solver(settings.linear_solver);
    integrator.set_preconditioner_reuse(settings.pc_reuse);
    result.timings.grid_s += elapsed_seconds(grid_start);
    if (settings.intermediate_snapshots)
        observe_state(grid, T, I, easy_ro, BasinSnapshot::Kind::initial, 0, 0,
                      configs.front().age_ma, observe, result.timings);

    for (std::size_t epoch_index = 0; epoch_index + 1 < configs.size(); ++epoch_index) {
        const auto epoch_start = Clock::now();
        grid_start = epoch_start;
        const auto& cfg0 = configs[epoch_index];
        const auto& cfg1 = configs[epoch_index + 1];
        BasinIntervalStats epoch_stats{
            .epoch = epoch_index, .age_begin_ma = cfg0.age_ma, .age_end_ma = cfg1.age_ma};
        const double duration = (cfg0.age_ma - cfg1.age_ma) * kSecondsPerMa;
        const double epoch_begin = (configs.front().age_ma - cfg0.age_ma) * kSecondsPerMa;
        const double dt_max = settings.dt_ma * kSecondsPerMa;
        const auto& epoch = epochs_[epoch_index];
        std::vector<double> node_T0(epoch.deposition_times.size()), node_T1(node_T0.size());
        std::vector<const LithotypeProperties*> lithotypes(epoch.lithotypes.size());
        for (std::size_t i = 0; i < lithotypes.size(); ++i)
            lithotypes[i] = &input.lithotype(epoch.lithotypes[i]);
        std::vector<double> phi(grid.cells.size());
        const auto update_porosity = [&](double theta) {
            for_each_layer_cell(grid, [&](const ColCell& cell, double depth) {
                const auto i = cell.index;
                const double top = std::lerp(epoch.phi0[i][0], epoch.phi1[i][0], theta);
                const double bottom = std::lerp(epoch.phi0[i][1], epoch.phi1[i][1], theta);
                phi[i] = std::lerp(top, bottom, depth);
            });
        };
        update_porosity(0.0);
        const auto boundaries_at = [&](double theta) {
            auto boundaries = bc(cfg0, cfg1, theta);
            if (!parameters_.empty()) {
                const auto index = static_cast<Vector::Index>(epoch_index);
                const double q = std::lerp(parameters_[index], parameters_[index + 1], theta);
                boundaries.q_basal = [q](Point2D) { return q; };
            }
            return boundaries;
        };
        {
            std::vector<double> lambda(grid.cells.size());
            for (const auto& cell : grid.cells)
                lambda[cell.index] = thermal_properties(T[cell.index], phi[cell.index],
                                                        *lithotypes[cell.index], kFluid)
                                         .lambda;
            interpolate_horizontal_temperature(grid, values(T), lambda, boundaries_at(0.0),
                                               node_T0);
        }
        double t = 0.0;
        integrator.set_initial_solution(t, T);
        if (epoch_index > 0 && settings.pc_reuse == PcReuse::manual) {
            integrator.reset_linear_solver();
        }
        epoch_stats.timings.grid_s += elapsed_seconds(grid_start);
        while (t < duration) {
            const auto step_start = Clock::now();
            grid_start = step_start;
            const double t_next = std::min(duration, t + dt_max);
            const double dt = t_next - t;
            const double theta = t_next / duration;
            BasinIntervalStats step_stats{.epoch = epoch_index,
                                          .age_begin_ma = cfg0.age_ma - t / kSecondsPerMa,
                                          .age_end_ma = cfg0.age_ma - t_next / kSecondsPerMa};
            grid = epoch.mesh.at(theta);
            update_porosity(theta);
            ColHeatSystem system(grid, phi, lithotypes, kFluid, boundaries_at(theta));
            system.set_flux_scheme(settings.scheme);
            integrator.set_timestep(dt);
            step_stats.timings.grid_s += elapsed_seconds(grid_start);
            const TimeIntegrationResult step =
                integrator.advance_to(system, t_next, settings.nonlinear, settings.linear);
            step_stats.timings.nonlinear = step.timings;
            step_stats.timings.properties = system.property_timings();
            if (!step.completed()) {
                throw std::runtime_error(
                    "basin nonlinear solve failed at age " +
                    std::to_string(cfg0.age_ma - theta * (cfg0.age_ma - cfg1.age_ma)) +
                    ": nonlinear_status=" +
                    std::to_string(step.last_nonlinear_status
                                       ? static_cast<int>(*step.last_nonlinear_status)
                                       : -1) +
                    " linear_status=" +
                    std::to_string(
                        step.last_linear_status ? static_cast<int>(*step.last_linear_status) : -1) +
                    " nonlinear_iterations=" + std::to_string(step.nonlinear_iterations) +
                    " residual=" + std::to_string(step.last_r_norm));
            }
            const Vector& T1 = integrator.current_snapshot().u;
            const auto easy_ro_start = Clock::now();
            system.interpolate_horizontal_temperature(T1, node_T1);
            easy_ro.advance(node_T0, node_T1, epoch_begin + t, epoch_begin + t_next,
                            epoch.deposition_times, I);
            node_T0.swap(node_T1);
            step_stats.timings.easy_ro_s += elapsed_seconds(easy_ro_start);
            step_stats.timesteps = step.accepted_steps;
            step_stats.nonlinear_iterations = step.nonlinear_iterations;
            step_stats.linear_iterations = step.linear_iterations;
            t = t_next;
            const bool at_event = t == duration;
            if (settings.intermediate_snapshots)
                observe_state(
                    grid, T1, I, easy_ro,
                    at_event ? BasinSnapshot::Kind::before_event : BasinSnapshot::Kind::timestep,
                    epoch_index, epoch_stats.timesteps + step_stats.timesteps,
                    at_event ? cfg1.age_ma : step_stats.age_end_ma, observe, step_stats.timings);
            step_stats.timings.total_s = elapsed_seconds(step_start);
            epoch_stats.timesteps += step_stats.timesteps;
            epoch_stats.nonlinear_iterations += step_stats.nonlinear_iterations;
            epoch_stats.linear_iterations += step_stats.linear_iterations;
            epoch_stats.timings += step_stats.timings;
            result.steps.push_back(step_stats);
        }
        grid_start = Clock::now();
        const bool last = epoch_index + 2 == configs.size();
        int born = -1;
        if (!last)
            born = epochs_[epoch_index + 1].mesh.born_layer_id.value_or(-1);
        else if (settings.reset_final_maturity)
            born = epoch.mesh.born_layer_id.value_or(-1);
        ColGrid next_grid = last ? final_grid_ : epochs_[epoch_index + 1].initial_grid;
        epoch_stats.timings.grid_s += elapsed_seconds(grid_start);
        const auto transfer_start = Clock::now();
        T = interpolate_temperature(grid, next_grid, integrator.current_snapshot().u);
        I = interpolate_reaction_integrals(grid, next_grid, I, easy_ro.reaction_count(), born);
        if (!last && born != -1) {
            for (const auto& cell : next_grid.cells) {
                if (cell.layer_id == born)
                    T[static_cast<Vector::Index>(cell.index)] = cfg1.T_surf.at(cell.center().x);
            }
        }
        grid = std::move(next_grid);
        epoch_stats.timings.transfer_s += elapsed_seconds(transfer_start);
        if (settings.intermediate_snapshots || last)
            observe_state(grid, T, I, easy_ro, BasinSnapshot::Kind::after_event, epoch_index + 1, 0,
                          cfg1.age_ma, observe, epoch_stats.timings);
        epoch_stats.timings.total_s = elapsed_seconds(epoch_start);
        result.timesteps += epoch_stats.timesteps;
        result.nonlinear_iterations += epoch_stats.nonlinear_iterations;
        result.linear_iterations += epoch_stats.linear_iterations;
        result.timings += epoch_stats.timings;
        result.epochs.push_back(epoch_stats);
    }
    result.timings.total_s = elapsed_seconds(solve_start);
    return result;
}
