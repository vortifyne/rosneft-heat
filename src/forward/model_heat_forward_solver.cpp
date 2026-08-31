#include "forward/model_heat_forward_solver.hpp"

#include "discretization/model_heat_system.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

void check_finite(double value, const char* message) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(message);
    }
}

void check_parameters(const ModelHeatParameters& parameters) {
    check_finite(parameters.surface_temperature, "Surface temperature must be finite");
    check_finite(parameters.basal_heat_flux, "Basal heat flux must be finite");
}

struct InterpolationAxis {
    int lower;
    int upper;
    double upper_weight;
};

InterpolationAxis make_axis(double coordinate, double first_cell_center, double spacing,
                            int cell_count) {
    const double centre_coordinate = (coordinate - first_cell_center) / spacing;
    const double clamped_coordinate =
        std::clamp(centre_coordinate, 0.0, static_cast<double>(cell_count - 1));
    const int lower = static_cast<int>(std::floor(clamped_coordinate));
    const int upper = std::min(lower + 1, cell_count - 1);
    return {
        .lower = lower,
        .upper = upper,
        .upper_weight = clamped_coordinate - static_cast<double>(lower),
    };
}

} // namespace

ModelHeatForwardSolver::ModelHeatForwardSolver(RegularGrid grid, ModelHeatProblem problem,
                                               ModelHeatForwardSettings settings,
                                               ModelHeatParameters initial_parameters)
    : grid_(grid), problem_(std::move(problem)), settings_(settings),
      parameters_(initial_parameters), initial_temperature_(grid_.cell_count()) {
    check_finite(settings_.initial_time, "Initial time must be finite");
    check_finite(settings_.final_time, "Final time must be finite");
    if (settings_.final_time < settings_.initial_time) {
        throw std::invalid_argument("Final time must not be less than initial time");
    }
    if (!(settings_.time_step > 0.0) || !std::isfinite(settings_.time_step)) {
        throw std::invalid_argument("Time step must be finite and positive");
    }
    check_finite(problem_.initial_condition.initial_temperature,
                 "Initial temperature must be finite");
    check_parameters(parameters_);

    initial_temperature_.set_constant(problem_.initial_condition.initial_temperature);
    observation_stencils_.reserve(problem_.observation_points.size());
    for (const ObservationPoint& point : problem_.observation_points) {
        observation_stencils_.push_back(make_observation_stencil(point));
    }
}

void ModelHeatForwardSolver::set_surface_temperature(double value) {
    check_finite(value, "Surface temperature must be finite");
    parameters_.surface_temperature = value;
}

void ModelHeatForwardSolver::set_basal_heat_flux(double value) {
    check_finite(value, "Basal heat flux must be finite");
    parameters_.basal_heat_flux = value;
}

void ModelHeatForwardSolver::set_parameters(const ModelHeatParameters& parameters) {
    check_parameters(parameters);
    parameters_ = parameters;
}

const ModelHeatParameters& ModelHeatForwardSolver::parameters() const noexcept {
    return parameters_;
}

std::size_t ModelHeatForwardSolver::observation_count() const noexcept {
    return observation_stencils_.size();
}

ModelHeatForwardResult ModelHeatForwardSolver::solve() const {
    const auto start_time = std::chrono::steady_clock::now();
    const ModelHeatSystem system(grid_, problem_.lithotype, parameters_);
    TimeIntegrator integrator;
    integrator.set_initial_solution(settings_.initial_time, initial_temperature_);
    integrator.set_timestep(settings_.time_step);
    integrator.set_linear_solver(settings_.linear_solver);

    const TimeIntegrationResult integration =
        integrator.advance_to(system, settings_.final_time, settings_.nonlinear, settings_.linear);
    const TimeSnapshot& snapshot = integrator.current_snapshot();

    ModelHeatForwardResult result{
        .integration = integration,
        .final_state = {.time = snapshot.time, .temperature = snapshot.solution},
        .calculated_temperature = interpolate_observations(snapshot.solution),
        .elapsed_time_seconds = 0.0,
    };
    const auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    return result;
}

ModelHeatForwardSolver::ObservationStencil
ModelHeatForwardSolver::make_observation_stencil(const ObservationPoint& point) const {
    check_finite(point.x, "Observation x coordinate must be finite");
    check_finite(point.z, "Observation z coordinate must be finite");
    if (point.x < 0.0 || point.x > grid_.W || point.z < grid_.z_surf || point.z > grid_.L) {
        throw std::invalid_argument("Observation point must lie inside the computational domain");
    }

    const InterpolationAxis x_axis = make_axis(point.x, 0.5 * grid_.dx, grid_.dx, grid_.nx);
    const InterpolationAxis z_axis =
        make_axis(point.z, grid_.z_surf + (0.5 * grid_.dz), grid_.dz, grid_.nz);
    const auto cell_index = [&](int ix, int iz) {
        return (static_cast<Vector::Index>(iz) * static_cast<Vector::Index>(grid_.nx)) +
               static_cast<Vector::Index>(ix);
    };
    const double lower_x_weight = 1.0 - x_axis.upper_weight;
    const double lower_z_weight = 1.0 - z_axis.upper_weight;

    return {
        .indices =
            {
                cell_index(x_axis.lower, z_axis.lower),
                cell_index(x_axis.upper, z_axis.lower),
                cell_index(x_axis.lower, z_axis.upper),
                cell_index(x_axis.upper, z_axis.upper),
            },
        .weights =
            {
                lower_x_weight * lower_z_weight,
                x_axis.upper_weight * lower_z_weight,
                lower_x_weight * z_axis.upper_weight,
                x_axis.upper_weight * z_axis.upper_weight,
            },
    };
}

Vector ModelHeatForwardSolver::interpolate_observations(const Vector& temperature) const {
    Vector calculated_temperature(static_cast<Vector::Index>(observation_stencils_.size()));
    for (std::size_t observation = 0; observation < observation_stencils_.size(); ++observation) {
        const ObservationStencil& stencil = observation_stencils_[observation];
        double value = 0.0;
        for (std::size_t entry = 0; entry < stencil.indices.size(); ++entry) {
            value += stencil.weights[entry] * temperature[stencil.indices[entry]];
        }
        calculated_temperature[static_cast<Vector::Index>(observation)] = value;
    }
    return calculated_temperature;
}
