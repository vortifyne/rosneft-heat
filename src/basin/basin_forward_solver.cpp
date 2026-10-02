#include "basin/basin_forward_solver.hpp"

#include "basin/basin_state.hpp"
#include "basin/forward_output.hpp"
#include "basin/layered_mesh.hpp"
#include "basin/mesh_motion.hpp"
#include "time/time_integrator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr double kSecondsPerYear = 365.25 * 24.0 * 3600.0;
constexpr double kSecondsPerMa = 1.0e6 * kSecondsPerYear;

HeatBoundaryConditions boundary_conditions(const ScalarProfile& first_surface_temperature,
                                           const ScalarProfile& second_surface_temperature,
                                           const ScalarProfile& first_basal_heat_flux,
                                           const ScalarProfile& second_basal_heat_flux,
                                           const double start_time, const double duration) {
    const auto fraction = [start_time, duration](const double time) {
        return std::clamp((time - start_time) / duration, 0.0, 1.0);
    };
    return {
        .surface_temperature =
            [&first_surface_temperature, &second_surface_temperature, fraction](const Point2D point,
                                                                                const double time) {
                const double alpha = fraction(time);
                return ((1.0 - alpha) * first_surface_temperature.at(point.x)) +
                       (alpha * second_surface_temperature.at(point.x));
            },
        .basal_heat_flux =
            [&first_basal_heat_flux, &second_basal_heat_flux, fraction](const Point2D point,
                                                                        const double time) {
                const double alpha = fraction(time);
                return ((1.0 - alpha) * first_basal_heat_flux.at(point.x)) +
                       (alpha * second_basal_heat_flux.at(point.x));
            },
    };
}

NonlinearSolveRequest nonlinear_request() {
    return {
        .nonlinear_method = NonlinearMethod::picard,
        .relative_tolerance = 1.0e-6,
        .absolute_tolerance = 0.0,
        .step_relative_tolerance = 1.0e-8,
        .max_iterations = 200,
        .use_backtracking = true,
        .max_backtracking_steps = 20,
        .backtracking_reduction = 0.5,
    };
}

LinearSolveRequest linear_request() {
    return {.relative_tolerance = 1.0e-12, .absolute_tolerance = 1.0e-12, .max_iterations = 1};
}

void accumulate_result(BasinForwardResult& total, const TimeIntegrationResult& step) {
    total.accepted_steps += step.accepted_steps;
    total.nonlinear_iterations += step.nonlinear_iterations;
    total.maximum_nonlinear_iterations_per_step =
        std::max(total.maximum_nonlinear_iterations_per_step, step.nonlinear_iterations);
    total.linear_iterations += step.linear_iterations;
    if (!step.completed()) {
        throw std::runtime_error(
            "Nonlinear solution did not converge: nonlinear_status=" +
            std::to_string(
                step.last_nonlinear_status ? static_cast<int>(*step.last_nonlinear_status) : -1) +
            " nonlinear_iterations=" + std::to_string(step.nonlinear_iterations) +
            " residual=" + std::to_string(step.last_residual_norm));
    }
}

double maximum_cell_diameter(const Mesh2D& mesh) {
    double result = 0.0;
    for (const TQMesh::Facet* cell : mesh.cells()) {
        for (std::size_t first = 0; first < cell->n_vertices(); ++first) {
            for (std::size_t second = first + 1; second < cell->n_vertices(); ++second) {
                result =
                    std::max(result, (cell->vertex(first).xy() - cell->vertex(second).xy()).norm());
            }
        }
    }
    return result;
}

void update_mesh_statistics(BasinForwardResult& result, const Mesh2D& mesh) {
    const std::size_t cells = mesh.cells().size();
    result.minimum_cells =
        result.minimum_cells == 0 ? cells : std::min(result.minimum_cells, cells);
    result.maximum_cells = std::max(result.maximum_cells, cells);
    result.maximum_cell_diameter =
        std::max(result.maximum_cell_diameter, maximum_cell_diameter(mesh));
}

EasyRoModel make_easy_ro(const BasinInput& input) {
    const EasyRoInput& parameters = input.easy_ro();
    return EasyRoModel({.preexponential = parameters.preexponential,
                        .gas_constant = parameters.gas_constant,
                        .activation_energies = parameters.activation_energies,
                        .weights = parameters.weights});
}

void validate_parameters(const BasinInput& input, const BasinForwardParameters& parameters) {
    const std::size_t count = input.configurations().size();
    if (parameters.surface_temperature.size() != count ||
        parameters.basal_heat_flux.size() != count) {
        throw std::invalid_argument("Basin boundary parameters must match the configuration count");
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (parameters.surface_temperature[index].x.empty() ||
            parameters.basal_heat_flux[index].x.empty()) {
            throw std::invalid_argument("Basin boundary profiles must not be empty");
        }
    }
}

} // namespace

struct BasinForwardSolver::Impl {
    Impl(const std::filesystem::path& input_directory, const BasinForwardSettings solver_settings)
        : input(BasinInput::read(input_directory)), settings(solver_settings),
          easy_ro(make_easy_ro(input)) {
        if (!(settings.cell_size > 0.0) || !(settings.timestep_ma > 0.0)) {
            throw std::invalid_argument("Invalid forward-solver settings");
        }
        const std::span<const BasinConfiguration> configurations = input.configurations();
        if (configurations.size() < 2) {
            throw std::runtime_error("At least two basin configurations are required");
        }
        for (std::size_t index = 1; index < configurations.size(); ++index) {
            if (!(configurations[index - 1].age_ma > configurations[index].age_ma)) {
                throw std::runtime_error(
                    "Basin configurations must be ordered from oldest to newest");
            }
        }
        layout = make_layered_mesh_layout(configurations, settings.cell_size);
        default_parameters.surface_temperature.reserve(configurations.size());
        default_parameters.basal_heat_flux.reserve(configurations.size());
        for (const BasinConfiguration& configuration : configurations) {
            default_parameters.surface_temperature.push_back(configuration.surface_temperature);
            default_parameters.basal_heat_flux.push_back(configuration.basal_heat_flux);
        }
    }

    BasinInput input;
    BasinForwardSettings settings;
    LayeredMeshLayout layout;
    BasinForwardParameters default_parameters;
    EasyRoModel easy_ro;
};

BasinForwardSolver::BasinForwardSolver(const std::filesystem::path& input_directory,
                                       const BasinForwardSettings settings)
    : impl_(std::make_unique<Impl>(input_directory, settings)) {}

BasinForwardSolver::~BasinForwardSolver() = default;
BasinForwardSolver::BasinForwardSolver(BasinForwardSolver&&) noexcept = default;
BasinForwardSolver& BasinForwardSolver::operator=(BasinForwardSolver&&) noexcept = default;

const BasinForwardParameters& BasinForwardSolver::default_parameters() const noexcept {
    return impl_->default_parameters;
}

BasinForwardResult BasinForwardSolver::solve(const BasinForwardParameters& parameters,
                                             const std::filesystem::path& output_directory) const {
    const auto wall_start = std::chrono::steady_clock::now();
    const BasinInput& input = impl_->input;
    const BasinForwardSettings& settings = impl_->settings;
    const EasyRoModel& easy_ro = impl_->easy_ro;
    validate_parameters(input, parameters);
    const std::span<const BasinConfiguration> configurations = input.configurations();
    const double timestep = settings.timestep_ma * kSecondsPerMa;

    BasinForwardOutput output(output_directory);
    const BasinConfiguration& oldest = configurations.front();
    BasinState state = make_initial_basin_state(
        oldest, input, make_layered_mesh(oldest, impl_->layout), easy_ro,
        parameters.surface_temperature.front(), parameters.basal_heat_flux.front());

    BasinForwardResult result;
    result.configurations = static_cast<int>(configurations.size());
    update_mesh_statistics(result, state.mesh);
    output.save_state(state, easy_ro, 0.0, oldest.age_ma, "initial", 0);
    double elapsed = 0.0;
    double accumulated_energy_imbalance = 0.0;
    double accumulated_basal_energy = 0.0;

    for (std::size_t epoch = 0; epoch + 1 < configurations.size(); ++epoch) {
        const auto epoch_wall_start = std::chrono::steady_clock::now();
        const BasinConfiguration& first = configurations[epoch];
        const BasinConfiguration& second = configurations[epoch + 1];
        const double duration = (first.age_ma - second.age_ma) * kSecondsPerMa;
        const double epoch_end = elapsed + duration;
        const int steps_before = result.accepted_steps;
        const int nonlinear_before = result.nonlinear_iterations;
        const int linear_before = result.linear_iterations;
        const std::size_t old_cell_count = state.mesh.cells().size();

        bool topology_regularized = false;
        std::unique_ptr<BasinMeshMotion> motion;
        try {
            motion = std::make_unique<BasinMeshMotion>(state.mesh, first, second, duration);
        } catch (const std::exception&) {
            motion = std::make_unique<BasinMeshMotion>(state.mesh, first, second, duration, true);
            topology_regularized = true;
            ++result.topology_regularized_epochs;
        }
        state.fields.velocity_z.assign(motion->velocity_z().begin(), motion->velocity_z().end());
        std::unique_ptr<HeatSystem> system = make_heat_system(
            state.mesh, state.fields,
            boundary_conditions(parameters.surface_temperature[epoch],
                                parameters.surface_temperature[epoch + 1],
                                parameters.basal_heat_flux[epoch],
                                parameters.basal_heat_flux[epoch + 1], elapsed, duration));

        double current_time = elapsed;
        while (current_time < epoch_end) {
            const double old_time = current_time;
            const double target = std::min(epoch_end, old_time + timestep);
            Vector old_temperature = state.temperature;
            const double energy_before = total_energy(state);
            const double fraction = (target - elapsed) / duration;
            motion->set_position(fraction);
            motion->set_porosity(fraction, state.fields.porosity);
            system->update_geometry(state.mesh);

            TimeIntegrator integrator;
            integrator.set_initial_solution(old_time, state.temperature);
            integrator.set_timestep(target - old_time);
            const TimeIntegrationResult step =
                integrator.advance_to(*system, target, nonlinear_request(), linear_request());
            try {
                accumulate_result(result, step);
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    "Epoch " + std::to_string(first.age_ma) + " -> " +
                    std::to_string(second.age_ma) + " Ma at age " +
                    std::to_string(first.age_ma - ((target - elapsed) / kSecondsPerMa)) +
                    " Ma: " + error.what());
            }
            state.temperature = integrator.current_snapshot().solution;
            const HeatEnergyRates rates = system->energy_rates(target, state.temperature);
            const double energy_after = total_energy(state);
            const double step_duration = target - old_time;
            accumulated_energy_imbalance +=
                energy_after - energy_before +
                (step_duration *
                 (rates.surface_outflow - rates.basal_inflow - rates.heat_production));
            accumulated_basal_energy += step_duration * rates.basal_inflow;
            result.global_energy_balance = std::abs(accumulated_energy_imbalance) /
                                           std::max(std::abs(accumulated_basal_energy), 1.0);
            output.save_energy_balance(target / kSecondsPerMa,
                                       first.age_ma - ((target - elapsed) / kSecondsPerMa), "step",
                                       result.global_energy_balance);
            easy_ro.advance(vector_values(old_temperature), vector_values(state.temperature),
                            step_duration, state.maturity);
            current_time = target;

            if (target == epoch_end) {
                output.save_state(state, easy_ro, target / kSecondsPerMa,
                                  first.age_ma - ((target - elapsed) / kSecondsPerMa), "epoch",
                                  step.nonlinear_iterations);
            }
        }

        elapsed = epoch_end;
        result.final_age_ma = second.age_ma;
        output.save_state(state, easy_ro, elapsed / kSecondsPerMa, second.age_ma,
                          "before_transition", 0);

        BasinStateTransfer transferred = transfer_basin_state(
            state, make_layered_mesh(second, impl_->layout), first, second, input, easy_ro,
            parameters.surface_temperature[epoch + 1], settings.cell_size);
        accumulated_energy_imbalance += transferred.energy_imbalance;
        result.global_energy_balance = std::abs(accumulated_energy_imbalance) /
                                       std::max(std::abs(accumulated_basal_energy), 1.0);
        output.save_transition(second.age_ma, state.mesh.cells().size(),
                               transferred.state.mesh.cells().size());
        output.save_energy_balance(elapsed / kSecondsPerMa, second.age_ma, "transition",
                                   result.global_energy_balance);

        state = std::move(transferred.state);
        update_thermophysical_properties(vector_values(state.temperature), state.fields);
        update_mesh_statistics(result, state.mesh);
        output.save_state(state, easy_ro, elapsed / kSecondsPerMa, second.age_ma,
                          "after_transition", 0);

        const double epoch_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_wall_start)
                .count();
        output.save_epoch(first.age_ma, second.age_ma, result.accepted_steps - steps_before,
                          result.nonlinear_iterations - nonlinear_before,
                          result.linear_iterations - linear_before, old_cell_count,
                          state.mesh.cells().size(), topology_regularized, epoch_seconds);
    }

    result.wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    output.finish(result);
    return result;
}
