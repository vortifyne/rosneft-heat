#include "model_heat_inverse_test_problem.hpp"
#include "petsc_test_session.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using namespace model_heat_inverse_test;

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

    EXPECT_THROW(static_cast<void>(solver.solve({
                     .initial_parameters = {.surface_temperature = 21.0, .basal_heat_flux = 0.5},
                     .optimization = kOptimizationRequest,
                 })),
                 std::invalid_argument);
}

TEST(ModelHeatInverseSolverTest, PropagatesForwardFailureUntilPenaltyPolicyIsImplemented) {
    ensure_petsc_test_session();
    ModelHeatForwardSettings failing_settings = make_settings();
    failing_settings.nonlinear.max_iterations = 0;
    ModelHeatInverseSolver solver = make_inverse_solver(
        ModelHeatObservationData(Vector{10.0, 10.0, 10.0}, Vector{0.1, 0.1, 0.1}),
        failing_settings);

    EXPECT_THROW(static_cast<void>(solver.solve({
                     .initial_parameters = {.surface_temperature = 12.0, .basal_heat_flux = 0.3},
                     .optimization = kOptimizationRequest,
                 })),
                 std::runtime_error);
}

} // namespace
