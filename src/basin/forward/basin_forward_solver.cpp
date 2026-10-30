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

Vector as_vector(std::span<const double> values) {
    Vector result(static_cast<Vector::Index>(values.size()));
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[static_cast<Vector::Index>(i)] = values[i];
    }
    return result;
}

struct State {
    ColGrid grid;
    Vector T;
    std::vector<double> I;
};

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

State initial_state(const ColEpochGrid& epoch, const BasinConfig& cfg0, const BasinInput& input,
                    const EasyRoModel& easy_ro) {
    State state{.grid = epoch.at(0.0), .T = {}, .I = {}};
    state.T = Vector(static_cast<Vector::Index>(state.grid.cells.size()));
    for (const auto& cell : state.grid.cells) {
        const double x = cell.center().x;
        const double T_surf = cfg0.T_surf.at(x);
        const auto& layer = cfg0.layer(cell.layer_id);
        const auto& lithotype = input.lithotype(layer.lithotype_at(x));
        const double lambda = thermal_properties(T_surf, layer.phi_at(x), lithotype, kFluid).lambda;
        const double top = state.grid.cells[state.grid.top_cells[cell.column]].z_top;
        state.T[static_cast<Vector::Index>(cell.index)] =
            T_surf + cfg0.q_basal.at(x) * (cell.center().z - top) / lambda;
    }
    state.I = easy_ro.initial_state(state.grid.cells.size());
    return state;
}

void observe_state(const State& state, const EasyRoModel& easy_ro, BasinSnapshot::Kind kind,
                   double age_ma, const BasinObserver& observer) {
    if (!observer)
        return;
    std::vector<double> Ro(state.grid.cells.size());
    easy_ro.reflectance(state.I, Ro);
    observer({.kind = kind,
              .age_ma = age_ma,
              .grid = state.grid,
              .T = values(state.T),
              .I = state.I,
              .Ro = Ro});
}

State transfer_state(const State& old, ColGrid new_grid, const EasyRoModel& easy_ro,
                     int born_layer_id) {
    const ColFieldTransfer transfer = make_col_field_transfer(old.grid, new_grid);
    const std::vector<double> T = transfer.interpolate_temperature(values(old.T));
    State state{
        .grid = std::move(new_grid), .T = as_vector(T), .I = easy_ro.initial_state(T.size())};
    const std::size_t reactions = easy_ro.reaction_count();
    for (std::size_t reaction = 0; reaction < reactions; ++reaction) {
        std::vector<double> old_field(old.grid.cells.size());
        for (std::size_t cell = 0; cell < old_field.size(); ++cell) {
            old_field[cell] = old.I[cell * reactions + reaction];
        }
        const std::vector<double> next = transfer.interpolate_same_layer(old_field);
        for (std::size_t cell = 0; cell < next.size(); ++cell) {
            if (state.grid.cells[cell].layer_id == born_layer_id)
                continue;
            if (transfer.layer_stencils[cell].empty()) {
                throw std::runtime_error("Existing layer has no maturity donor");
            }
            state.I[cell * reactions + reaction] = next[cell];
        }
    }
    return state;
}

} // namespace

BasinForwardSolver::BasinForwardSolver(BasinInput input, BasinForwardSettings settings)
    : input_(std::move(input)), settings_(std::move(settings)) {}

BasinForwardResult BasinForwardSolver::solve(const BasinObserver& observe) const {
    const auto& input = input_;
    const auto& settings = settings_;
    const auto configs = input.configs();
    if (settings.nonlinear.method != NonlinearMethod::picard || configs.size() < 2 ||
        !(settings.dx > 0.0) || !(settings.dz > 0.0) || !(settings.dz_min > 0.0) ||
        !(settings.dt_ma > 0.0) || !(settings.nonlinear.rtol > 0.0) ||
        !(settings.nonlinear.rtol < 1.0)) {
        throw std::invalid_argument("Invalid basin configurations or settings");
    }
    const EasyRoModel easy_ro(input.easy_ro());
    std::vector<BasinGeometry> geometries;
    geometries.reserve(configs.size());
    for (const auto& cfg : configs)
        geometries.push_back(
            simplify_basin_geometry_rdp(make_basin_geometry(cfg), settings.rdp_tol));
    const auto prepare_epoch = [&](std::size_t index) {
        return make_col_epoch_grid(geometries[index], geometries[index + 1], settings.dx,
                                   settings.dz, settings.dz_min, settings.axis_rdp_tol,
                                   lithotype_breakpoints(configs[index], &configs[index + 1]));
    };
    auto epoch = prepare_epoch(0);
    State state = initial_state(epoch, configs[0], input, easy_ro);
    BasinForwardResult result;
    TimeIntegrator integrator;
    integrator.set_linear_solver(settings.linear_solver);
    integrator.set_preconditioner_reuse(settings.pc_reuse);
    observe_state(state, easy_ro, BasinSnapshot::Kind::initial, configs.front().age_ma, observe);

    for (std::size_t epoch_index = 0; epoch_index + 1 < configs.size(); ++epoch_index) {
        const auto& cfg0 = configs[epoch_index];
        const auto& cfg1 = configs[epoch_index + 1];
        if (!(cfg0.age_ma > cfg1.age_ma)) {
            throw std::invalid_argument("basin ages must decrease");
        }
        const double duration = (cfg0.age_ma - cfg1.age_ma) * kSecondsPerMa;
        const double dt_max = settings.dt_ma * kSecondsPerMa;
        std::vector<const LithotypeProperties*> lithotypes(state.grid.cells.size());
        for (const auto& cell : state.grid.cells) {
            const auto& layer = cfg0.layer(cell.layer_id);
            lithotypes[cell.index] = &input.lithotype(layer.lithotype_at(cell.center().x));
        }
        std::vector<double> phi(state.grid.cells.size());
        double t = 0.0;
        integrator.set_initial_solution(t, state.T);
        if (epoch_index > 0 && settings.pc_reuse == PcReuse::manual) {
            integrator.reset_linear_solver();
        }
        while (t < duration) {
            const double t_next = std::min(duration, t + dt_max);
            const double dt = t_next - t;
            const double theta = t_next / duration;
            ColGrid grid = epoch.at(theta);
            const Vector& T0 = state.T;
            for (const auto& cell : grid.cells) {
                const double x = cell.center().x;
                phi[cell.index] = std::lerp(cfg0.layer(cell.layer_id).phi_at(x),
                                            cfg1.layer(cell.layer_id).phi_at(x), theta);
            }
            ColHeatSystem system(grid, phi, lithotypes, kFluid, bc(cfg0, cfg1, theta));
            system.set_flux_scheme(settings.scheme);
            integrator.set_timestep(dt);
            const TimeIntegrationResult step =
                integrator.advance_to(system, t_next, settings.nonlinear, settings.linear);
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
            const Vector T1 = integrator.current_snapshot().u;
            easy_ro.advance(values(T0), values(T1), dt, state.I);
            result.timesteps += step.accepted_steps;
            result.nonlinear_iterations += step.nonlinear_iterations;
            result.linear_iterations += step.linear_iterations;
            state.grid = std::move(grid);
            state.T = T1;
            t = t_next;
        }
        observe_state(state, easy_ro, BasinSnapshot::Kind::before_event, cfg1.age_ma, observe);
        const bool last = epoch_index + 2 == configs.size();
        const int born = epoch.born_layer_id.value_or(-1);
        ColGrid next_grid;
        if (last) {
            const auto& geometry = geometries.back();
            const auto knots = settings.axis_rdp_tol
                                   ? simplify_basin_geometry_rdp(geometry, *settings.axis_rdp_tol).x
                                   : geometry.x;
            next_grid = make_col_grid(
                geometry, make_column_axis(knots, settings.dx, lithotype_breakpoints(cfg1)),
                settings.dz);
        } else {
            epoch = prepare_epoch(epoch_index + 1);
            next_grid = epoch.at(0.0);
        }
        state = transfer_state(state, std::move(next_grid), easy_ro, born);
        observe_state(state, easy_ro, BasinSnapshot::Kind::after_event, cfg1.age_ma, observe);
    }
    return result;
}
