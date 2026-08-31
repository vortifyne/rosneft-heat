#include "inverse/model_heat_inverse_solver.hpp"
#include "petsc_test_session.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <utility>

namespace {

constexpr NonlinearSolveRequest kNewtonRequest{
    .nonlinear_method = NonlinearMethod::newton,
    .relative_tolerance = 1e-12,
    .absolute_tolerance = 1e-12,
    .step_relative_tolerance = 1e-12,
    .max_iterations = 5,
};

constexpr LinearSolveRequest kDirectLinearRequest{
    .relative_tolerance = 1e-12,
    .absolute_tolerance = 1e-12,
    .max_iterations = 1,
};

constexpr PoundersSolveRequest kOptimizationRequest{
    .gradient_absolute_tolerance = 1e-8,
    .gradient_relative_tolerance = 1e-8,
    .gradient_reduction_tolerance = 1e-8,
    .max_iterations = 100,
    .max_function_evaluations = 500,
};

RegularGrid make_grid() {
    return RegularGrid(3, 3, 3.0, 0.0, 3.0);
}

ModelHeatProblem make_problem() {
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

ModelHeatForwardSettings make_settings() {
    return {
        .initial_time = 0.0,
        .final_time = 0.3,
        .time_step = 0.1,
        .nonlinear = kNewtonRequest,
        .linear = kDirectLinearRequest,
        .linear_solver = LinearSolverKind::umfpack_lu,
    };
}

ModelHeatParameterSpace make_parameter_space() {
    return ModelHeatParameterSpace({
        .lower = {.surface_temperature = 5.0, .basal_heat_flux = 0.0},
        .upper = {.surface_temperature = 20.0, .basal_heat_flux = 1.0},
    });
}

ModelHeatForwardSolver make_forward_solver(ModelHeatForwardSettings settings = make_settings()) {
    return ModelHeatForwardSolver(make_grid(), make_problem(), settings,
                                  {.surface_temperature = 10.0, .basal_heat_flux = 0.0});
}

ModelHeatObservationData make_synthetic_observations(const ModelHeatParameters& parameters) {
    ModelHeatForwardSolver generator(make_grid(), make_problem(), make_settings(), parameters);
    ModelHeatForwardResult result = generator.solve();
    if (!result.completed()) {
        throw std::runtime_error("Failed to generate synthetic model heat observations");
    }
    return ModelHeatObservationData(std::move(result.calculated_temperature),
                                    Vector{0.1, 0.1, 0.1});
}

ModelHeatInverseSolver make_inverse_solver(ModelHeatObservationData observations,
                                           ModelHeatForwardSettings settings = make_settings()) {
    return ModelHeatInverseSolver(ModelHeatResidualEvaluator(
        make_forward_solver(settings), make_parameter_space(), std::move(observations)));
}

void expect_vector_near(const Vector& actual, const Vector& expected, double tolerance = 1e-6) {
    ASSERT_EQ(actual.size(), expected.size());
    for (Vector::Index index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance);
    }
}

TEST(ModelHeatInverseSolverTest, ConnectsModelEvaluatorToPoundersAndReturnsConsistentResult) {
    ensure_petsc_test_session();
    const ModelHeatParameters hidden_parameters{
        .surface_temperature = 12.0,
        .basal_heat_flux = 0.3,
    };
    const ModelHeatObservationData observations = make_synthetic_observations(hidden_parameters);
    const Vector observed_temperatures = observations.observed_temperatures();
    ModelHeatInverseSolver solver = make_inverse_solver(observations);

    const ModelHeatInverseResult result = solver.solve({
        .initial_parameters = hidden_parameters,
        .optimization = kOptimizationRequest,
    });

    EXPECT_TRUE(result.converged());
    EXPECT_NEAR(result.parameters.surface_temperature, hidden_parameters.surface_temperature, 1e-5);
    EXPECT_NEAR(result.parameters.basal_heat_flux, hidden_parameters.basal_heat_flux, 1e-5);
    expect_vector_near(result.calculated_temperatures, observed_temperatures);
    expect_vector_near(result.normalized_residuals, Vector{0.0, 0.0, 0.0});
    EXPECT_NEAR(result.objective, result.normalized_residuals.squared_norm(), 1e-14);
    EXPECT_EQ(result.forward_evaluations, result.optimization_function_evaluations + 1);
    EXPECT_GE(result.optimization_iterations, 0);
    EXPECT_GE(result.optimization_function_evaluations, 1);
    EXPECT_TRUE(std::isfinite(result.forward_elapsed_time_seconds));
    EXPECT_GE(result.forward_elapsed_time_seconds, 0.0);
}

TEST(ModelHeatInverseSolverTest, ReturnsBestPointAfterFunctionEvaluationLimit) {
    ensure_petsc_test_session();
    const ModelHeatParameters hidden_parameters{
        .surface_temperature = 18.0,
        .basal_heat_flux = 0.8,
    };
    ModelHeatInverseSolver solver =
        make_inverse_solver(make_synthetic_observations(hidden_parameters));
    PoundersSolveRequest limited_request = kOptimizationRequest;
    limited_request.max_function_evaluations = 3;

    const ModelHeatInverseResult result = solver.solve({
        .initial_parameters = {.surface_temperature = 6.0, .basal_heat_flux = 0.1},
        .optimization = limited_request,
    });

    EXPECT_EQ(result.status, PoundersSolveStatus::maximum_function_evaluations);
    EXPECT_FALSE(result.converged());
    EXPECT_GE(result.optimization_function_evaluations, 3);
    EXPECT_EQ(result.forward_evaluations, result.optimization_function_evaluations + 1);
    EXPECT_TRUE(result.normalized_parameters.all_finite());
    EXPECT_TRUE(result.normalized_residuals.all_finite());
    EXPECT_NEAR(result.objective, result.normalized_residuals.squared_norm(), 1e-14);
}

TEST(ModelHeatInverseSolverTest, ValidatesPhysicalInitialParameters) {
    ensure_petsc_test_session();
    ModelHeatInverseSolver solver = make_inverse_solver(
        ModelHeatObservationData(Vector{10.0, 10.0, 10.0}, Vector{0.1, 0.1, 0.1}));

    EXPECT_THROW(solver.solve({
                     .initial_parameters = {.surface_temperature = 21.0, .basal_heat_flux = 0.5},
                     .optimization = kOptimizationRequest,
                 }),
                 std::invalid_argument);
}

TEST(ModelHeatInverseSolverTest, PropagatesForwardFailureUntilPenaltyPolicyIsImplemented) {
    ensure_petsc_test_session();
    ModelHeatForwardSettings failing_settings = make_settings();
    failing_settings.nonlinear.max_iterations = 0;
    ModelHeatInverseSolver solver = make_inverse_solver(
        ModelHeatObservationData(Vector{10.0, 10.0, 10.0}, Vector{0.1, 0.1, 0.1}),
        failing_settings);

    EXPECT_THROW(solver.solve({
                     .initial_parameters = {.surface_temperature = 12.0, .basal_heat_flux = 0.3},
                     .optimization = kOptimizationRequest,
                 }),
                 std::runtime_error);
}

} // namespace
