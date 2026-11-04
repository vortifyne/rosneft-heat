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

struct State {
    ColGrid grid;
    Vector T;
    std::vector<double> I;
};

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

Vector initial_temperature(const ColGrid& grid, const ScalarProfile& surface) {
    double integral = 0.0;
    for (std::size_t i = 1; i < surface.x.size(); ++i)
        integral +=
            0.5 * (surface.values[i - 1] + surface.values[i]) * (surface.x[i] - surface.x[i - 1]);
    const double mean_surface = integral / (surface.x.back() - surface.x.front());
    Vector T(static_cast<Vector::Index>(grid.cells.size()));
    for (const auto& cell : grid.cells) {
        const double top = grid.cells[grid.top_cells[cell.column]].z_top;
        T[static_cast<Vector::Index>(cell.index)] = mean_surface + 0.03 * (cell.center().z - top);
    }
    return T;
}

void observe_state(const State& state, const EasyRoModel& easy_ro, BasinSnapshot::Kind kind,
                   std::size_t epoch, int step, double age_ma, const BasinObserver& observer,
                   BasinTimings& timings) {
    if (!observer)
        return;
    auto start = Clock::now();
    std::vector<double> Ro(state.grid.cells.size() + state.grid.columns.size());
    easy_ro.reflectance(state.I, Ro);
    timings.easy_ro_s += elapsed_seconds(start);
    start = Clock::now();
    observer({.kind = kind,
              .epoch = epoch,
              .step = step,
              .age_ma = age_ma,
              .grid = state.grid,
              .T = values(state.T),
              .I = state.I,
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
        data.deposition_times = deposition_times(data.initial_grid, configs);
        const auto count = data.initial_grid.cells.size();
        data.lithotypes.resize(count);
        data.phi0.resize(count);
        data.phi1.resize(count);
        for (const auto& cell : data.initial_grid.cells) {
            const double x = cell.center().x;
            const auto& layer0 = configs[i].layer(cell.layer_id);
            data.lithotypes[cell.index] = layer0.lithotype_at(x);
            data.phi0[cell.index] = layer0.phi_at(x);
            data.phi1[cell.index] = configs[i + 1].layer(cell.layer_id).phi_at(x);
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
    initial_temperature_ =
        initial_temperature(epochs_.front().initial_grid, configs.front().T_surf);
    if (!settings.reset_final_maturity && epochs_.back().mesh.born_layer_id) {
        const int born = *epochs_.back().mesh.born_layer_id;
        const auto source = epochs_.back().mesh.at(1.0);
        const auto depths = horizontal_node_depths(final_grid_);
        const auto times = deposition_times(final_grid_, configs);
        std::vector<double> centers;
        for (std::size_t column = 0; column < source.columns.size(); ++column)
            centers.push_back(std::midpoint(source.x[column], source.x[column + 1]));
        for (const auto& column : final_grid_.columns) {
            const auto first = std::find_if(column.begin(), column.end(), [&](std::size_t index) {
                return final_grid_.cells[index].layer_id == born;
            });
            if (first == column.end())
                continue;
            const auto last = std::find_if_not(first, column.end(), [&](std::size_t index) {
                return final_grid_.cells[index].layer_id == born;
            });
            const auto c = final_grid_.cells[*first].column;
            const double x = std::midpoint(final_grid_.x[c], final_grid_.x[c + 1]);
            auto right = static_cast<std::size_t>(
                std::lower_bound(centers.begin(), centers.end(), x) - centers.begin());
            right = std::min(std::max(right, std::size_t{1}), centers.size() - 1);
            const auto left = right == 0 ? 0 : right - 1;
            const double wx =
                left == right ? 0.0 : (x - centers[left]) / (centers[right] - centers[left]);
            for (std::size_t node = *first + c; node <= *(last - 1) + c + 1; ++node) {
                std::array<std::size_t, 4> cells;
                std::array<double, 4> weights;
                for (std::size_t side = 0; side < 2; ++side) {
                    const auto& indices = source.columns[side == 0 ? left : right];
                    const auto upper =
                        std::lower_bound(indices.begin(), indices.end(), depths[node],
                                         [&](std::size_t index, double z) {
                                             return source.cells[index].center().z < z;
                                         });
                    const auto r = std::min(
                        std::max(std::size_t{1}, static_cast<std::size_t>(upper - indices.begin())),
                        indices.size() - 1);
                    const auto l = r == 0 ? 0 : r - 1;
                    const double z0 = source.cells[indices[l]].center().z;
                    const double z1 = source.cells[indices[r]].center().z;
                    const double wz = l == r ? 0.0 : (depths[node] - z0) / (z1 - z0);
                    cells[2 * side] = indices[l];
                    cells[2 * side + 1] = indices[r];
                    const double weight_x = side == 0 ? 1.0 - wx : wx;
                    weights[2 * side] = weight_x * (1.0 - wz);
                    weights[2 * side + 1] = weight_x * wz;
                }
                final_birth_nodes_.push_back(node);
                final_birth_cells_.push_back(cells);
                final_birth_weights_.push_back(weights);
                final_birth_deposition_.push_back(times[node]);
            }
        }
    }
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

BasinForwardResult BasinForwardSolver::solve(const BasinObserver& observe) const {
    const auto solve_start = Clock::now();
    auto grid_start = solve_start;
    BasinForwardResult result;
    const auto& input = input_;
    const auto& settings = settings_;
    const auto configs = input.configs();
    const auto& easy_ro = easy_ro_;
    State state{.grid = epochs_.front().initial_grid,
                .T = initial_temperature_,
                .I = easy_ro.initial_state(epochs_.front().deposition_times.size())};
    auto final_birth_I = easy_ro.initial_state(final_birth_nodes_.size());
    std::vector<double> birth_T0(final_birth_nodes_.size()), birth_T1(final_birth_nodes_.size());
    result.epochs.reserve(configs.size() - 1);
    TimeIntegrator integrator;
    integrator.set_linear_solver(settings.linear_solver);
    integrator.set_preconditioner_reuse(settings.pc_reuse);
    result.timings.grid_s += elapsed_seconds(grid_start);
    if (settings.intermediate_snapshots)
        observe_state(state, easy_ro, BasinSnapshot::Kind::initial, 0, 0, configs.front().age_ma,
                      observe, result.timings);

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
        interpolate_horizontal_temperature(state.grid, state.T, node_T0);
        std::vector<const LithotypeProperties*> lithotypes(epoch.lithotypes.size());
        for (std::size_t i = 0; i < lithotypes.size(); ++i)
            lithotypes[i] = &input.lithotype(epoch.lithotypes[i]);
        std::vector<double> phi(state.grid.cells.size());
        double t = 0.0;
        integrator.set_initial_solution(t, state.T);
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
            ColGrid grid = epoch.mesh.at(theta);
            const Vector& T0 = state.T;
            for (std::size_t i = 0; i < phi.size(); ++i)
                phi[i] = std::lerp(epoch.phi0[i], epoch.phi1[i], theta);
            auto boundaries = bc(cfg0, cfg1, theta);
            if (!parameters_.empty()) {
                const auto index = static_cast<Vector::Index>(epoch_index);
                const double q = std::lerp(parameters_[index], parameters_[index + 1], theta);
                boundaries.q_basal = [q](Point2D) { return q; };
            }
            ColHeatSystem system(grid, phi, lithotypes, kFluid, std::move(boundaries));
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
            interpolate_horizontal_temperature(grid, T1, node_T1);
            easy_ro.advance(node_T0, node_T1, epoch_begin + t, epoch_begin + t_next,
                            epoch.deposition_times, state.I);
            node_T0.swap(node_T1);
            if (epoch_index + 1 == epochs_.size() && !final_birth_nodes_.empty()) {
                for (std::size_t node = 0; node < final_birth_nodes_.size(); ++node) {
                    double old_value = 0.0, new_value = 0.0;
                    for (std::size_t j = 0; j < 4; ++j) {
                        const auto index = static_cast<Vector::Index>(final_birth_cells_[node][j]);
                        const double weight = final_birth_weights_[node][j];
                        old_value += weight * T0[index];
                        new_value += weight * T1[index];
                    }
                    birth_T0[node] = old_value;
                    birth_T1[node] = new_value;
                }
                easy_ro.advance(birth_T0, birth_T1, epoch_begin + t, epoch_begin + t_next,
                                final_birth_deposition_, final_birth_I);
            }
            step_stats.timings.easy_ro_s += elapsed_seconds(easy_ro_start);
            step_stats.timesteps = step.accepted_steps;
            step_stats.nonlinear_iterations = step.nonlinear_iterations;
            step_stats.linear_iterations = step.linear_iterations;
            state.grid = std::move(grid);
            state.T = T1;
            t = t_next;
            const bool at_event = t == duration;
            if (settings.intermediate_snapshots)
                observe_state(
                    state, easy_ro,
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
        const int born = epoch.mesh.born_layer_id.value_or(-1);
        ColGrid next_grid = last ? final_grid_ : epochs_[epoch_index + 1].initial_grid;
        epoch_stats.timings.grid_s += elapsed_seconds(grid_start);
        const auto transfer_start = Clock::now();
        state.T = interpolate_temperature(state.grid, next_grid, state.T);
        state.I = interpolate_reaction_integrals(state.grid, next_grid, state.I,
                                                 easy_ro.reaction_count(), born);
        if (last) {
            const auto reactions = easy_ro.reaction_count();
            for (std::size_t node = 0; node < final_birth_nodes_.size(); ++node)
                std::copy_n(final_birth_I.begin() + static_cast<std::ptrdiff_t>(node * reactions),
                            reactions,
                            state.I.begin() +
                                static_cast<std::ptrdiff_t>(final_birth_nodes_[node] * reactions));
        }
        state.grid = std::move(next_grid);
        epoch_stats.timings.transfer_s += elapsed_seconds(transfer_start);
        if (settings.intermediate_snapshots || last)
            observe_state(state, easy_ro, BasinSnapshot::Kind::after_event, epoch_index + 1, 0,
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
