#include "inverse/model_heat_residual_evaluator.hpp"

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

RegularGrid make_grid() {
    return RegularGrid(3, 3, 1500.0, 0.0, 1500.0);
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
                {.x = 750.0, .z = 250.0},
                {.x = 750.0, .z = 750.0},
                {.x = 750.0, .z = 1250.0},
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
        .lower = {.surface_temperature = 10.0, .basal_heat_flux = 0.0},
        .upper = {.surface_temperature = 30.0, .basal_heat_flux = 0.4},
    });
}

ModelHeatObservationData make_observations() {
    return ModelHeatObservationData(Vector{12.0, 13.0, 14.0}, Vector{0.5, 1.0, 2.0});
}

ModelHeatForwardSolver make_forward_solver(ModelHeatForwardSettings settings = make_settings()) {
    return ModelHeatForwardSolver(make_grid(), make_problem(), settings);
}

void expect_vector_near(const Vector& actual, const Vector& expected, double tolerance = 1e-12) {
    ASSERT_EQ(actual.size(), expected.size());
    for (Vector::Index index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance);
    }
}

TEST(ModelHeatResidualEvaluatorTest, EvaluatesNormalizedParametersAndReturnsResidualData) {
    ModelHeatResidualEvaluator evaluator(make_forward_solver(), make_parameter_space(),
                                         make_observations());
    const Vector normalized_parameters{0.25, 0.75};
    const ModelHeatParameters physical_parameters =
        evaluator.parameter_space().to_physical(normalized_parameters);
    ModelHeatForwardSolver direct_solver(make_grid(), make_problem(), make_settings());

    const ModelHeatForwardResult direct_result = direct_solver.solve(physical_parameters);
    const ModelHeatResidualResult result = evaluator.evaluate(normalized_parameters);
    const Vector expected_residuals =
        evaluator.observations().normalized_residuals(direct_result.calculated_temperature);

    ASSERT_TRUE(result.completed());
    expect_vector_near(result.calculated_temperatures, direct_result.calculated_temperature);
    expect_vector_near(result.normalized_residuals, expected_residuals);
    EXPECT_NEAR(result.objective, expected_residuals.squared_norm(), 1e-12);
    EXPECT_EQ(result.time_integration.accepted_steps, direct_result.integration.accepted_steps);
    EXPECT_EQ(result.time_integration.nonlinear_iterations,
              direct_result.integration.nonlinear_iterations);
    EXPECT_EQ(result.time_integration.linear_iterations,
              direct_result.integration.linear_iterations);
    EXPECT_TRUE(std::isfinite(result.forward_elapsed_time_seconds));
    EXPECT_GE(result.forward_elapsed_time_seconds, 0.0);
}

TEST(ModelHeatResidualEvaluatorTest, RepeatsEveryEvaluationFromTheInitialState) {
    ModelHeatResidualEvaluator evaluator(make_forward_solver(), make_parameter_space(),
                                         make_observations());

    const ModelHeatResidualResult first = evaluator.evaluate(Vector{0.2, 0.3});
    const ModelHeatResidualResult changed = evaluator.evaluate(Vector{0.8, 0.7});
    const ModelHeatResidualResult repeated = evaluator.evaluate(Vector{0.2, 0.3});

    ASSERT_TRUE(first.completed());
    ASSERT_TRUE(changed.completed());
    ASSERT_TRUE(repeated.completed());
    EXPECT_NE(changed.objective, first.objective);
    expect_vector_near(repeated.calculated_temperatures, first.calculated_temperatures);
    expect_vector_near(repeated.normalized_residuals, first.normalized_residuals);
    EXPECT_NEAR(repeated.objective, first.objective, 1e-12);
}

TEST(ModelHeatResidualEvaluatorTest, ReportsForwardFailureWithoutComputingResiduals) {
    ModelHeatForwardSettings settings = make_settings();
    settings.nonlinear.max_iterations = 0;
    ModelHeatResidualEvaluator evaluator(make_forward_solver(settings), make_parameter_space(),
                                         make_observations());

    const ModelHeatResidualResult result = evaluator.evaluate(Vector{0.5, 0.5});

    EXPECT_FALSE(result.completed());
    EXPECT_EQ(result.time_integration.status, TimeIntegrationStatus::nonlinear_solve_failed);
    EXPECT_EQ(result.calculated_temperatures.size(), evaluator.residual_count());
    EXPECT_TRUE(result.normalized_residuals.empty());
    EXPECT_TRUE(std::isinf(result.objective));
    EXPECT_TRUE(std::isfinite(result.forward_elapsed_time_seconds));
}

TEST(ModelHeatResidualEvaluatorTest, ValidatesObservationCountAndNormalizedParameters) {
    EXPECT_THROW(ModelHeatResidualEvaluator(make_forward_solver(), make_parameter_space(),
                                            ModelHeatObservationData(Vector{12.0}, Vector{1.0})),
                 std::invalid_argument);

    ModelHeatResidualEvaluator evaluator(make_forward_solver(), make_parameter_space(),
                                         make_observations());
    EXPECT_EQ(evaluator.parameter_count(), 2);
    EXPECT_EQ(evaluator.residual_count(), 3);
    EXPECT_THROW(static_cast<void>(evaluator.evaluate(Vector{0.5})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(evaluator.evaluate(Vector{0.5, 1.1})), std::invalid_argument);
}

} // namespace
