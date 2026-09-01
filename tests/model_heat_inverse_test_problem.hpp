#pragma once

#include "inverse/model_heat_inverse_solver.hpp"

#include <stdexcept>
#include <utility>

namespace model_heat_inverse_test {

inline constexpr NonlinearSolveRequest kNewtonRequest{
    .nonlinear_method = NonlinearMethod::newton,
    .relative_tolerance = 1e-12,
    .absolute_tolerance = 1e-12,
    .step_relative_tolerance = 1e-12,
    .max_iterations = 5,
};

inline constexpr LinearSolveRequest kDirectLinearRequest{
    .relative_tolerance = 1e-12,
    .absolute_tolerance = 1e-12,
    .max_iterations = 1,
};

inline constexpr PoundersSolveRequest kOptimizationRequest{
    .gradient_absolute_tolerance = 1e-8,
    .gradient_relative_tolerance = 1e-8,
    .gradient_reduction_tolerance = 1e-8,
    .max_iterations = 100,
    .max_function_evaluations = 500,
};

inline RegularGrid make_grid() {
    return RegularGrid(3, 3, 3.0, 0.0, 3.0);
}

inline ModelHeatProblem make_problem() {
    return {
        .lithotype =
            {
                .thermal_conductivity = 1.0,
                .density = 1.0,
                .specific_heat = 1.0,
                .heat_production = 0.1,
            },
        .initial_condition = {.initial_temperature = 10.0},
        .observation_points =
            {
                {.x = 1.5, .z = 0.5},
                {.x = 1.5, .z = 1.5},
                {.x = 1.5, .z = 2.5},
            },
    };
}

inline ModelHeatForwardSettings make_settings() {
    return {
        .initial_time = 0.0,
        .final_time = 0.3,
        .time_step = 0.1,
        .nonlinear = kNewtonRequest,
        .linear = kDirectLinearRequest,
        .linear_solver = LinearSolverKind::umfpack_lu,
    };
}

inline ModelHeatParameterSpace make_parameter_space() {
    return ModelHeatParameterSpace({
        .lower = {.surface_temperature = 5.0, .basal_heat_flux = 0.0},
        .upper = {.surface_temperature = 20.0, .basal_heat_flux = 1.0},
    });
}

inline ModelHeatForwardSolver
make_forward_solver(ModelHeatForwardSettings settings = make_settings()) {
    return ModelHeatForwardSolver(make_grid(), make_problem(), settings);
}

inline ModelHeatForwardResult solve_forward(const ModelHeatParameters& parameters,
                                            ModelHeatForwardSettings settings = make_settings()) {
    ModelHeatForwardSolver solver(make_grid(), make_problem(), settings);
    ModelHeatForwardResult result = solver.solve(parameters);
    if (!result.completed()) {
        throw std::runtime_error("Failed to solve synthetic model heat problem");
    }
    return result;
}

inline ModelHeatObservationData make_synthetic_observations(const ModelHeatParameters& parameters) {
    ModelHeatForwardResult result = solve_forward(parameters);
    return ModelHeatObservationData(std::move(result.calculated_temperature),
                                    Vector{0.1, 0.1, 0.1});
}

inline ModelHeatInverseSolver
make_inverse_solver(ModelHeatObservationData observations,
                    ModelHeatForwardSettings settings = make_settings()) {
    return ModelHeatInverseSolver(ModelHeatResidualEvaluator(
        make_forward_solver(settings), make_parameter_space(), std::move(observations)));
}

} // namespace model_heat_inverse_test
