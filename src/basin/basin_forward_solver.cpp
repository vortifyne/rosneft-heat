#include "basin/basin_forward_solver.hpp"

#include "basin/basin_binary.hpp"
#include "basin/basin_geometry_rdp.hpp"
#include "basin/column_epoch_grid.hpp"
#include "basin/column_field_transfer.hpp"
#include "discretization/nonmatching_column_transient_system.hpp"
#include "physics/easy_ro.hpp"
#include "physics/thermophysical_properties.hpp"
#include "time/time_integrator.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kSecondsPerMa = 1.0e6 * 365.25 * 24.0 * 3600.0;
constexpr FluidThermophysicalProperties kFluid{
    .thermal_conductivity = 0.7, .density = 1040.0, .specific_heat = 4184.0};

std::vector<double> lithotype_breakpoints(const BasinConfiguration& first,
                                          const BasinConfiguration* second = nullptr) {
    std::vector<double> result;
    const auto append = [&result](const BasinConfiguration& configuration) {
        for (const auto& layer : configuration.layers) {
            for (std::size_t i = 1; i < layer.x.size(); ++i) {
                if (layer.lithotype[i - 1] != layer.lithotype[i]) {
                    result.push_back(std::midpoint(layer.x[i - 1], layer.x[i]));
                }
            }
        }
    };
    append(first);
    if (second)
        append(*second);
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

EasyRoModel make_easy_ro(const BasinInput& input) {
    const auto& kinetics = input.easy_ro();
    return EasyRoModel({.preexponential = kinetics.preexponential,
                        .gas_constant = kinetics.gas_constant,
                        .activation_energies = kinetics.activation_energies,
                        .weights = kinetics.weights});
}

struct State {
    NonmatchingColumnGrid grid;
    Vector temperature;
    std::vector<double> maturity;
};

struct Properties {
    std::vector<double> porosity;
    std::vector<const LithotypeThermophysicalProperties*> lithotypes;
    std::vector<double> conductivity;
    std::vector<double> capacity;
    std::vector<double> production;

    void evaluate(std::span<const double> temperature) {
        evaluate_thermophysical_property_fields(temperature, porosity, lithotypes, kFluid,
                                                conductivity, capacity, production);
    }
};

Properties make_properties(const NonmatchingColumnGrid& grid, const BasinConfiguration& first,
                           const BasinConfiguration& second, const BasinInput& input,
                           double fraction, std::span<const double> temperature) {
    const std::size_t count = grid.cells.size();
    Properties properties{.porosity = std::vector<double>(count),
                          .lithotypes =
                              std::vector<const LithotypeThermophysicalProperties*>(count),
                          .conductivity = std::vector<double>(count),
                          .capacity = std::vector<double>(count),
                          .production = std::vector<double>(count)};
    for (const auto& cell : grid.cells) {
        const double x = cell.center().x;
        const auto& before = first.layer(cell.layer_id);
        const auto& after = second.layer(cell.layer_id);
        properties.porosity[cell.index] =
            std::lerp(before.porosity_at(x), after.porosity_at(x), fraction);
        properties.lithotypes[cell.index] = &input.lithotype(before.lithotype_at(x));
    }
    properties.evaluate(temperature);
    return properties;
}

NonmatchingColumnBoundaryData boundary_data(const BasinConfiguration& first,
                                            const BasinConfiguration& second, double fraction) {
    return {
        .surface_temperature =
            [&first, &second, fraction](Point2D point) {
                return std::lerp(first.surface_temperature.at(point.x),
                                 second.surface_temperature.at(point.x), fraction);
            },
        .basal_inward_flux =
            [&first, &second, fraction](Point2D point) {
                return std::lerp(first.basal_heat_flux.at(point.x),
                                 second.basal_heat_flux.at(point.x), fraction);
            },
    };
}

State initial_state(const ColumnEpochGrid& epoch, const BasinConfiguration& first,
                    const BasinInput& input, const EasyRoModel& easy_ro) {
    State state{.grid = epoch.at(0.0), .temperature = {}, .maturity = {}};
    state.temperature = Vector(static_cast<Vector::Index>(state.grid.cells.size()));
    for (const auto& cell : state.grid.cells) {
        const double x = cell.center().x;
        const double surface = first.surface_temperature.at(x);
        const auto& layer = first.layer(cell.layer_id);
        const auto& lithotype = input.lithotype(layer.lithotype_at(x));
        const double conductivity =
            evaluate_thermophysical_properties(surface, layer.porosity_at(x), lithotype, kFluid)
                .thermal_conductivity;
        const double top = state.grid.cells[state.grid.top_cells[cell.column]].z_top;
        state.temperature[static_cast<Vector::Index>(cell.index)] =
            surface + first.basal_heat_flux.at(x) * (cell.center().z - top) / conductivity;
    }
    state.maturity = easy_ro.initial_state(state.grid.cells.size());
    return state;
}

void observe_state(const State& state, const EasyRoModel& easy_ro, BasinForwardSnapshot::Kind kind,
                   double age_ma, const BasinForwardObserver& observer) {
    if (!observer)
        return;
    std::vector<double> reflectance(state.grid.cells.size());
    easy_ro.reflectance(state.maturity, reflectance);
    observer({.kind = kind,
              .age_ma = age_ma,
              .grid = state.grid,
              .temperature = values(state.temperature),
              .maturity_integrals = state.maturity,
              .reflectance_percent = reflectance});
}

State transfer_state(const State& old, NonmatchingColumnGrid new_grid, const EasyRoModel& easy_ro,
                     int born_layer_id) {
    const ColumnFieldTransfer transfer = make_column_field_transfer(old.grid, new_grid);
    const std::vector<double> temperature =
        transfer.interpolate_temperature(values(old.temperature));
    State state{.grid = std::move(new_grid),
                .temperature = as_vector(temperature),
                .maturity = easy_ro.initial_state(temperature.size())};
    const std::size_t reactions = easy_ro.reaction_count();
    for (std::size_t reaction = 0; reaction < reactions; ++reaction) {
        std::vector<double> old_field(old.grid.cells.size());
        for (std::size_t cell = 0; cell < old_field.size(); ++cell) {
            old_field[cell] = old.maturity[cell * reactions + reaction];
        }
        const std::vector<double> next = transfer.interpolate_same_layer(old_field);
        for (std::size_t cell = 0; cell < next.size(); ++cell) {
            if (state.grid.cells[cell].layer_id == born_layer_id)
                continue;
            if (transfer.same_layer[cell].empty()) {
                throw std::runtime_error("Existing layer has no maturity donor");
            }
            state.maturity[cell * reactions + reaction] = next[cell];
        }
    }
    return state;
}

NonlinearSolveRequest nonlinear_request(double relative_tolerance) {
    return {.nonlinear_method = NonlinearMethod::picard,
            .relative_tolerance = relative_tolerance,
            .absolute_tolerance = 1.0e-8,
            .max_iterations = 200};
}

} // namespace

BasinForwardSolver::BasinForwardSolver(BasinInput input, BasinForwardSettings settings)
    : input_(std::move(input)), settings_(std::move(settings)) {}

BasinForwardResult BasinForwardSolver::solve(const BasinForwardObserver& observe) const {
    const auto& input = input_;
    const auto& settings = settings_;
    const auto configurations = input.configurations();
    if (configurations.size() < 2 || !(settings.horizontal_step > 0.0) ||
        !(settings.vertical_step > 0.0) || !(settings.minimum_cell_height > 0.0) ||
        !(settings.timestep_ma > 0.0) || !(settings.picard_relative_tolerance > 0.0) ||
        !(settings.picard_relative_tolerance < 1.0)) {
        throw std::invalid_argument("Invalid basin configurations or settings");
    }
    const EasyRoModel easy_ro = make_easy_ro(input);
    const double minimum_height = settings.pinchout == BasinForwardSettings::Pinchout::exact_zero
                                      ? 0.0
                                      : settings.minimum_cell_height;
    const auto first_material_x = lithotype_breakpoints(configurations[0], &configurations[1]);
    const auto first_epoch = make_column_epoch_grid(
        make_column_epoch_geometry(configurations[0], configurations[1], settings.rdp_tolerance),
        settings.horizontal_step, settings.vertical_step, minimum_height,
        settings.axis_rdp_tolerance, first_material_x);
    State state = initial_state(first_epoch, configurations[0], input, easy_ro);
    BasinForwardResult result;
    TimeIntegrator integrator;
    integrator.set_linear_solver(settings.linear_solver);
    observe_state(state, easy_ro, BasinForwardSnapshot::Kind::initial,
                  configurations.front().age_ma, observe);

    for (std::size_t epoch_index = 0; epoch_index + 1 < configurations.size(); ++epoch_index) {
        const auto& first = configurations[epoch_index];
        const auto& second = configurations[epoch_index + 1];
        if (!(first.age_ma > second.age_ma)) {
            throw std::invalid_argument("basin ages must decrease");
        }
        const auto material_x = lithotype_breakpoints(first, &second);
        const ColumnEpochGrid epoch =
            epoch_index == 0
                ? first_epoch
                : make_column_epoch_grid(
                      make_column_epoch_geometry(first, second, settings.rdp_tolerance),
                      settings.horizontal_step, settings.vertical_step, minimum_height,
                      settings.axis_rdp_tolerance, material_x);
        const double duration = (first.age_ma - second.age_ma) * kSecondsPerMa;
        const double timestep = settings.timestep_ma * kSecondsPerMa;
        double elapsed = 0.0;
        integrator.set_initial_solution(elapsed, state.temperature);
        if (epoch_index > 0) {
            integrator.reset_linear_solver();
        }
        while (elapsed < duration) {
            const double target = std::min(duration, elapsed + timestep);
            const double dt = target - elapsed;
            const double fraction = target / duration;
            ColumnEpochGrid::Active active;
            if (settings.pinchout == BasinForwardSettings::Pinchout::exact_zero) {
                active = epoch.at_active(fraction);
            } else {
                active.grid = epoch.at(fraction);
                active.full_indices.resize(active.grid.cells.size());
                for (std::size_t i = 0; i < active.full_indices.size(); ++i) {
                    active.full_indices[i] = i;
                }
            }
            NonmatchingColumnGrid current_grid = std::move(active.grid);
            Vector old_temperature(static_cast<Vector::Index>(current_grid.cells.size()));
            std::vector<double> old_maturity(current_grid.cells.size() * easy_ro.reaction_count());
            for (std::size_t cell = 0; cell < current_grid.cells.size(); ++cell) {
                const std::size_t previous = active.full_indices[cell];
                if (previous >= static_cast<std::size_t>(state.temperature.size())) {
                    throw std::runtime_error("basin persistent cell index changed within epoch");
                }
                old_temperature[static_cast<Vector::Index>(cell)] =
                    state.temperature[static_cast<Vector::Index>(previous)];
                for (std::size_t reaction = 0; reaction < easy_ro.reaction_count(); ++reaction) {
                    old_maturity[cell * easy_ro.reaction_count() + reaction] =
                        state.maturity[previous * easy_ro.reaction_count() + reaction];
                }
            }
            Properties properties = make_properties(current_grid, first, second, input, fraction,
                                                    values(old_temperature));
            auto property_updater = [&properties](std::span<const double> temperature,
                                                  std::span<double> conductivity,
                                                  std::span<double> production) {
                properties.evaluate(temperature);
                std::copy(properties.conductivity.begin(), properties.conductivity.end(),
                          conductivity.begin());
                std::copy(properties.production.begin(), properties.production.end(),
                          production.begin());
            };
            auto capacity_updater = [&properties](double, std::span<const double> temperature,
                                                  std::span<double> capacity) {
                properties.evaluate(temperature);
                std::copy(properties.capacity.begin(), properties.capacity.end(), capacity.begin());
            };
            NonmatchingColumnTransientSystem system(
                current_grid, properties.conductivity, properties.production, properties.capacity,
                boundary_data(first, second, fraction), property_updater, capacity_updater);
            system.set_flux_scheme(settings.scheme);
            if (integrator.current_snapshot().solution.size() != old_temperature.size()) {
                integrator.set_initial_solution(elapsed, old_temperature);
                integrator.reset_linear_solver();
            }
            integrator.set_timestep(dt);
            const TimeIntegrationResult step = integrator.advance_to(
                system, target, nonlinear_request(settings.picard_relative_tolerance),
                settings.linear);
            if (!step.completed()) {
                throw std::runtime_error(
                    "basin nonlinear solve failed at age " +
                    std::to_string(first.age_ma - fraction * (first.age_ma - second.age_ma)) +
                    ": nonlinear_status=" +
                    std::to_string(step.last_nonlinear_status
                                       ? static_cast<int>(*step.last_nonlinear_status)
                                       : -1) +
                    " linear_status=" +
                    std::to_string(
                        step.last_linear_status ? static_cast<int>(*step.last_linear_status) : -1) +
                    " nonlinear_iterations=" + std::to_string(step.nonlinear_iterations) +
                    " residual=" + std::to_string(step.last_residual_norm));
            }
            const Vector new_temperature = integrator.current_snapshot().solution;
            easy_ro.advance(values(old_temperature), values(new_temperature), dt, old_maturity);
            Vector derivative(new_temperature.size());
            for (Vector::Index cell = 0; cell < derivative.size(); ++cell) {
                derivative[cell] = (new_temperature[cell] - old_temperature[cell]) / dt;
            }
            Vector residual;
            system.assemble_residual(dt, new_temperature, derivative, residual);
            result.maximum_cell_residual =
                std::max(result.maximum_cell_residual, residual.infinity_norm());
            const auto rates = system.spatial_system().energy_rates(new_temperature);
            const auto flux_diagnostics = system.spatial_system().flux_diagnostics(new_temperature);
            result.maximum_internal_flux_defect =
                std::max(result.maximum_internal_flux_defect,
                         flux_diagnostics.internal_conservation_defect_linf);
            properties.evaluate(values(new_temperature));
            double accumulation = 0.0;
            for (const auto& cell : current_grid.cells) {
                accumulation += cell.area() * properties.capacity[cell.index] *
                                derivative[static_cast<Vector::Index>(cell.index)];
            }
            const double global_residual = accumulation + rates.surface_outflow -
                                           rates.basal_inflow - rates.lateral_inflow -
                                           rates.heat_production;
            const double scale =
                std::max({1.0, std::abs(rates.surface_outflow), std::abs(rates.basal_inflow),
                          std::abs(rates.heat_production)});
            result.maximum_global_balance =
                std::max(result.maximum_global_balance, std::abs(global_residual) / scale);
            result.maximum_cells = std::max(result.maximum_cells, current_grid.cells.size());
            result.accepted_steps += step.accepted_steps;
            result.nonlinear_iterations += step.nonlinear_iterations;
            result.linear_iterations += step.linear_iterations;
            state.grid = std::move(current_grid);
            state.temperature = new_temperature;
            state.maturity = std::move(old_maturity);
            elapsed = target;
        }
        observe_state(state, easy_ro, BasinForwardSnapshot::Kind::before_event, second.age_ma,
                      observe);
        const bool last = epoch_index + 2 == configurations.size();
        NonmatchingColumnGrid after_grid;
        if (last) {
            const BasinGeometry after_geometry =
                simplify_basin_geometry_rdp(make_basin_geometry(second), settings.rdp_tolerance);
            const BasinGeometry axis_geometry =
                settings.axis_rdp_tolerance
                    ? simplify_basin_geometry_rdp(after_geometry, *settings.axis_rdp_tolerance)
                    : after_geometry;
            const ColumnEpochGeometry final_axis_geometry{.start = axis_geometry,
                                                          .after_event = axis_geometry,
                                                          .born_layer_id = std::nullopt};
            const auto final_material_x = lithotype_breakpoints(second);
            const LayeredMeshLayout layout{.x = make_column_epoch_axis(final_axis_geometry,
                                                                       settings.horizontal_step,
                                                                       final_material_x),
                                           .cell_size = settings.vertical_step};
            after_grid = make_nonmatching_column_grid(after_geometry, layout);
        } else {
            after_grid = make_column_epoch_grid(
                             make_column_epoch_geometry(second, configurations[epoch_index + 2],
                                                        settings.rdp_tolerance),
                             settings.horizontal_step, settings.vertical_step, minimum_height,
                             settings.axis_rdp_tolerance,
                             lithotype_breakpoints(second, &configurations[epoch_index + 2]))
                             .at(0.0);
        }
        const int born = epoch.geometry.born_layer_id.value_or(-1);
        state = transfer_state(state, std::move(after_grid), easy_ro, born);
        observe_state(state, easy_ro, BasinForwardSnapshot::Kind::after_event, second.age_ma,
                      observe);
        result.final_age_ma = second.age_ma;
    }
    return result;
}

BasinForwardResult BasinForwardSolver::solve(const std::filesystem::path& output,
                                             const BasinForwardObserver& observe) const {
    std::filesystem::create_directories(output);
    std::size_t snapshot_number = 0;
    const BasinForwardResult result = solve([&](const BasinForwardSnapshot& snapshot) {
        std::ostringstream name;
        name << "snapshot_" << std::setw(4) << std::setfill('0') << snapshot_number++ << ".cgrid";
        write_basin_binary(output / name.str(), snapshot);
        if (observe)
            observe(snapshot);
    });
    std::ofstream summary(output / "summary.txt");
    summary << std::setprecision(17) << "dx=" << settings_.horizontal_step << '\n'
            << "dz=" << settings_.vertical_step << '\n'
            << "dt_ma=" << settings_.timestep_ma << '\n'
            << "rdp_tol=" << settings_.rdp_tolerance << '\n'
            << "axis_rdp_tol=";
    if (settings_.axis_rdp_tolerance)
        summary << *settings_.axis_rdp_tolerance;
    else
        summary << "none";
    summary << '\n'
            << "min_height=" << settings_.minimum_cell_height << '\n'
            << "pinchout="
            << (settings_.pinchout == BasinForwardSettings::Pinchout::exact_zero ? "zero"
                                                                                 : "positive")
            << '\n'
            << "scheme=" << (settings_.scheme == MultipointFluxScheme::linear ? "linear" : "nmpfa")
            << '\n'
            << "linear_solver=";
    switch (settings_.linear_solver) {
    case LinearSolverKind::umfpack_lu:
        summary << "umfpack";
        break;
    case LinearSolverKind::petsc_gmres_gamg:
        summary << "gamg";
        break;
    case LinearSolverKind::petsc_gmres_boomeramg:
        summary << "boomeramg";
        break;
    case LinearSolverKind::petsc_gmres_ilu:
        summary << "gmres-ilu";
        break;
    case LinearSolverKind::petsc_gmres_ilu2:
        summary << "gmres-ilu2";
        break;
    case LinearSolverKind::petsc_bicgstab_ilu:
        summary << "bicgstab-ilu";
        break;
    case LinearSolverKind::petsc_bicgstab_ilu2:
        summary << "bicgstab-ilu2";
        break;
    case LinearSolverKind::petsc_bicgstab_gamg:
        summary << "bicgstab-gamg";
        break;
    case LinearSolverKind::petsc_bicgstab_boomeramg:
        summary << "bicgstab-boomeramg";
        break;
    case LinearSolverKind::petsc_gmres_gamg_reuse:
        summary << "gmres-gamg-reuse";
        break;
    case LinearSolverKind::petsc_gmres_boomeramg_reuse:
        summary << "gmres-boomeramg-reuse";
        break;
    case LinearSolverKind::petsc_bicgstab_gamg_reuse:
        summary << "bicgstab-gamg-reuse";
        break;
    case LinearSolverKind::petsc_bicgstab_boomeramg_reuse:
        summary << "bicgstab-boomeramg-reuse";
        break;
    default:
        throw std::logic_error("Unknown linear solver kind");
    }
    summary << '\n'
            << "picard_rtol=" << settings_.picard_relative_tolerance << '\n'
            << "accepted_steps=" << result.accepted_steps << '\n'
            << "nonlinear_iterations=" << result.nonlinear_iterations << '\n'
            << "linear_iterations=" << result.linear_iterations << '\n'
            << "maximum_cells=" << result.maximum_cells << '\n'
            << "maximum_cell_residual=" << result.maximum_cell_residual << '\n'
            << "maximum_global_balance=" << result.maximum_global_balance << '\n'
            << "maximum_internal_flux_defect=" << result.maximum_internal_flux_defect << '\n'
            << "final_age_ma=" << result.final_age_ma << '\n';
    if (!summary)
        throw std::runtime_error("Failed writing basin summary");
    summary.close();
    if (!summary)
        throw std::runtime_error("Failed closing basin summary");
    return result;
}
