#include "forward/model_heat_forward_solver.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

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

ModelHeatProblem make_problem(std::vector<ObservationPoint> observation_points) {
    return {
        .lithotype =
            {
                .thermal_conductivity = 1.0,
                .density = 1.0,
                .specific_heat = 1.0,
                .heat_production = 0.1,
            },
        .initial_condition = {.initial_temperature = 10.0},
        .observation_points = std::move(observation_points),
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

std::vector<ObservationPoint> make_well_observations() {
    std::vector<ObservationPoint> points;
    for (int depth = 0; depth <= 3000; depth += 500) {
        points.push_back({.x = 1000.0, .z = static_cast<double>(depth)});
    }
    return points;
}

void expect_vector_near(const Vector& actual, const Vector& expected, double tolerance = 1e-12) {
    ASSERT_EQ(actual.size(), expected.size());
    for (Vector::Index index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance);
    }
}

TEST(ModelHeatForwardSolverTest,
     RunsCompleteForwardCalculationAndSamplesWellEveryFiveHundredMeters) {
    const RegularGrid grid(4, 6, 2000.0, 0.0, 3000.0);
    ModelHeatForwardSolver solver(grid, make_problem(make_well_observations()), make_settings());

    EXPECT_EQ(solver.observation_count(), 7);
    const ModelHeatForwardResult result =
        solver.solve({.surface_temperature = 10.0, .basal_heat_flux = 0.2});

    ASSERT_TRUE(result.completed());
    EXPECT_EQ(result.integration.accepted_steps, 3);
    EXPECT_EQ(result.integration.rejected_steps, 0);
    EXPECT_EQ(result.integration.nonlinear_iterations, 3);
    EXPECT_EQ(result.integration.linear_iterations, 3);
    EXPECT_TRUE(std::isfinite(result.elapsed_time_seconds));
    EXPECT_GE(result.elapsed_time_seconds, 0.0);
    EXPECT_DOUBLE_EQ(result.final_state.time, 0.3);
    EXPECT_EQ(result.final_state.temperature.size(), grid.cell_count());
    ASSERT_EQ(result.calculated_temperature.size(), 7);

    const auto temperature = [&](int ix, int iz) {
        return result.final_state.temperature[(iz * grid.nx) + ix];
    };
    for (int iz = 0; iz < grid.nz; ++iz) {
        EXPECT_NEAR(temperature(0, iz), temperature(3, iz), 1e-12);
    }

    // x = width/2 lies between the two middle columns, but symmetry makes the vertical
    // interpolation sufficient below. Observations above/below the outer centres use their
    // constant values.
    EXPECT_NEAR(result.calculated_temperature[0], temperature(1, 0), 1e-12);
    EXPECT_NEAR(result.calculated_temperature[1], 0.5 * (temperature(1, 0) + temperature(1, 1)),
                1e-12);
    EXPECT_NEAR(result.calculated_temperature[2], 0.5 * (temperature(1, 1) + temperature(1, 2)),
                1e-12);
    EXPECT_NEAR(result.calculated_temperature[5], 0.5 * (temperature(1, 4) + temperature(1, 5)),
                1e-12);
    EXPECT_NEAR(result.calculated_temperature[6], temperature(1, 5), 1e-12);
}

TEST(ModelHeatForwardSolverTest, ReusesFixedProblemAndStartsEveryParameterRunFromInitialState) {
    const RegularGrid grid(4, 6, 2000.0, 0.0, 3000.0);
    const ModelHeatParameters base_parameters{
        .surface_temperature = 10.0,
        .basal_heat_flux = 0.2,
    };
    ModelHeatForwardSolver solver(grid, make_problem(make_well_observations()), make_settings());

    const ModelHeatForwardResult first = solver.solve(base_parameters);
    const ModelHeatForwardResult changed =
        solver.solve({.surface_temperature = 20.0, .basal_heat_flux = 0.4});
    const ModelHeatForwardResult repeated = solver.solve(base_parameters);

    ASSERT_TRUE(first.completed());
    ASSERT_TRUE(changed.completed());
    ASSERT_TRUE(repeated.completed());
    EXPECT_GT(changed.calculated_temperature[0], first.calculated_temperature[0]);
    EXPECT_GT(changed.calculated_temperature[6], first.calculated_temperature[6]);
    expect_vector_near(repeated.final_state.temperature, first.final_state.temperature);
    expect_vector_near(repeated.calculated_temperature, first.calculated_temperature);
}

TEST(ModelHeatForwardSolverTest, KeepsLastAcceptedStateWhenCalculationFails) {
    const RegularGrid grid(3, 3, 1500.0, 0.0, 1500.0);
    ModelHeatForwardSettings settings = make_settings();
    settings.nonlinear.max_iterations = 0;
    ModelHeatForwardSolver solver(grid, make_problem({{.x = 750.0, .z = 750.0}}), settings);

    const ModelHeatForwardResult result =
        solver.solve({.surface_temperature = 20.0, .basal_heat_flux = 0.2});

    EXPECT_FALSE(result.completed());
    EXPECT_EQ(result.integration.status, TimeIntegrationStatus::nonlinear_solve_failed);
    EXPECT_DOUBLE_EQ(result.final_state.time, settings.initial_time);
    for (Vector::Index index = 0; index < result.final_state.temperature.size(); ++index) {
        EXPECT_DOUBLE_EQ(result.final_state.temperature[index], 10.0);
    }
    ASSERT_EQ(result.calculated_temperature.size(), 1);
    EXPECT_DOUBLE_EQ(result.calculated_temperature[0], 10.0);
}

TEST(ModelHeatForwardSolverTest, SupportsSingleCellGridAndConstantBoundaryExtrapolation) {
    const RegularGrid grid(1, 1, 1000.0, 0.0, 1000.0);
    ModelHeatForwardSettings settings = make_settings();
    settings.final_time = settings.initial_time;
    ModelHeatForwardSolver solver(
        grid, make_problem({{.x = 0.0, .z = 0.0}, {.x = 1000.0, .z = 1000.0}}), settings);

    const ModelHeatForwardResult result =
        solver.solve({.surface_temperature = 10.0, .basal_heat_flux = 0.0});

    ASSERT_TRUE(result.completed());
    expect_vector_near(result.calculated_temperature, Vector{10.0, 10.0});
}

TEST(ModelHeatForwardSolverTest, ValidatesParametersSettingsAndObservationDomain) {
    const RegularGrid grid(2, 2, 1000.0, 0.0, 1000.0);
    EXPECT_THROW(
        ModelHeatForwardSolver(grid, make_problem({{.x = -1.0, .z = 500.0}}), make_settings()),
        std::invalid_argument);

    ModelHeatForwardSettings invalid_settings = make_settings();
    invalid_settings.time_step = 0.0;
    EXPECT_THROW(ModelHeatForwardSolver(grid, make_problem({}), invalid_settings),
                 std::invalid_argument);

    ModelHeatForwardSolver solver(grid, make_problem({}), make_settings());
    EXPECT_THROW(static_cast<void>(
                     solver.solve({.surface_temperature = std::numeric_limits<double>::infinity(),
                                   .basal_heat_flux = 0.2})),
                 std::invalid_argument);
}

} // namespace
