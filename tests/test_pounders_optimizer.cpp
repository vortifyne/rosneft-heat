#include "optimization/pounders_optimizer.hpp"
#include "petsc_test_session.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {

constexpr PoundersSolveRequest kSolveRequest{
    .gradient_absolute_tolerance = 1e-8,
    .gradient_relative_tolerance = 1e-8,
    .gradient_reduction_tolerance = 1e-8,
    .max_iterations = 100,
    .max_function_evaluations = 500,
};

void expect_vector_near(const Vector& actual, const Vector& expected, double tolerance = 1e-5) {
    ASSERT_EQ(actual.size(), expected.size());
    for (Vector::Index index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance);
    }
}

TEST(PoundersOptimizerTest, SolvesBoundedLeastSquaresProblem) {
    ensure_petsc_test_session();
    const BoundedLeastSquaresProblem problem{
        .lower_bounds = Vector{-5.0, -5.0},
        .upper_bounds = Vector{5.0, 5.0},
        .residual_count = 3,
        .residual_function =
            [](const Vector& parameters) {
                return Vector{parameters[0] - 1.0, parameters[1] - 2.0,
                              parameters[0] + parameters[1] - 3.0};
            },
    };

    const PoundersSolveResult result =
        PoundersOptimizer{}.solve(problem, Vector{0.0, 0.0}, kSolveRequest);

    EXPECT_TRUE(result.converged());
    expect_vector_near(result.parameters, Vector{1.0, 2.0});
    expect_vector_near(result.residuals, Vector{0.0, 0.0, 0.0});
    EXPECT_NEAR(result.objective, result.residuals.squared_norm(), 1e-14);
    EXPECT_GE(result.iterations, 0);
    EXPECT_GE(result.function_evaluations, 1);
}

TEST(PoundersOptimizerTest, RespectsActiveParameterBound) {
    ensure_petsc_test_session();
    const BoundedLeastSquaresProblem problem{
        .lower_bounds = Vector{-1.0},
        .upper_bounds = Vector{1.0},
        .residual_count = 1,
        .residual_function = [](const Vector& parameters) { return Vector{parameters[0] - 2.0}; },
    };

    const PoundersSolveResult result =
        PoundersOptimizer{}.solve(problem, Vector{0.0}, kSolveRequest);

    EXPECT_TRUE(result.converged());
    expect_vector_near(result.parameters, Vector{1.0});
    expect_vector_near(result.residuals, Vector{-1.0});
    EXPECT_NEAR(result.objective, 1.0, 1e-5);
}

TEST(PoundersOptimizerTest, SupportsNonUnitBounds) {
    ensure_petsc_test_session();
    const BoundedLeastSquaresProblem problem{
        .lower_bounds = Vector{2.0, -4.0},
        .upper_bounds = Vector{6.0, 0.0},
        .residual_count = 2,
        .residual_function =
            [](const Vector& parameters) {
                return Vector{parameters[0] - 4.0, parameters[1] + 2.0};
            },
    };

    const PoundersSolveResult result =
        PoundersOptimizer{}.solve(problem, Vector{3.0, -1.0}, kSolveRequest);

    EXPECT_TRUE(result.converged());
    expect_vector_near(result.parameters, Vector{4.0, -2.0});
    EXPECT_NEAR(result.objective, 0.0, 1e-10);
}

TEST(PoundersOptimizerTest, ReportsFunctionEvaluationLimit) {
    ensure_petsc_test_session();
    PoundersSolveRequest limited_request = kSolveRequest;
    limited_request.max_function_evaluations = 3;
    const BoundedLeastSquaresProblem problem{
        .lower_bounds = Vector{-5.0, -5.0},
        .upper_bounds = Vector{5.0, 5.0},
        .residual_count = 2,
        .residual_function =
            [](const Vector& parameters) {
                return Vector{parameters[0] - 4.0, parameters[1] - 4.0};
            },
    };

    const PoundersSolveResult result =
        PoundersOptimizer{}.solve(problem, Vector{0.0, 0.0}, limited_request);

    EXPECT_EQ(result.status, PoundersSolveStatus::maximum_function_evaluations);
    EXPECT_FALSE(result.converged());
    EXPECT_EQ(result.parameters.size(), 2);
    EXPECT_EQ(result.residuals.size(), 2);
    expect_vector_near(result.residuals,
                       Vector{result.parameters[0] - 4.0, result.parameters[1] - 4.0});
    EXPECT_NEAR(result.objective, result.residuals.squared_norm(), 1e-14);
    EXPECT_TRUE(std::isfinite(result.objective));
}

TEST(PoundersOptimizerTest, ValidatesProblemAndRequest) {
    ensure_petsc_test_session();
    const auto residual = [](const Vector& parameters) { return Vector{parameters[0]}; };
    const BoundedLeastSquaresProblem valid_problem{
        .lower_bounds = Vector{-1.0},
        .upper_bounds = Vector{1.0},
        .residual_count = 1,
        .residual_function = residual,
    };

    EXPECT_THROW(PoundersOptimizer{}.solve({.lower_bounds = Vector{},
                                            .upper_bounds = Vector{},
                                            .residual_count = 1,
                                            .residual_function = residual},
                                           Vector{}, kSolveRequest),
                 std::invalid_argument);
    EXPECT_THROW(PoundersOptimizer{}.solve({.lower_bounds = Vector{-1.0},
                                            .upper_bounds = Vector{1.0, 2.0},
                                            .residual_count = 1,
                                            .residual_function = residual},
                                           Vector{0.0}, kSolveRequest),
                 std::invalid_argument);
    EXPECT_THROW(PoundersOptimizer{}.solve({.lower_bounds = Vector{1.0},
                                            .upper_bounds = Vector{1.0},
                                            .residual_count = 1,
                                            .residual_function = residual},
                                           Vector{1.0}, kSolveRequest),
                 std::invalid_argument);
    EXPECT_THROW(PoundersOptimizer{}.solve(valid_problem, Vector{2.0}, kSolveRequest),
                 std::invalid_argument);
    EXPECT_THROW(PoundersOptimizer{}.solve({.lower_bounds = Vector{-1.0},
                                            .upper_bounds = Vector{1.0},
                                            .residual_count = 0,
                                            .residual_function = residual},
                                           Vector{0.0}, kSolveRequest),
                 std::invalid_argument);
    EXPECT_THROW(PoundersOptimizer{}.solve({.lower_bounds = Vector{-1.0},
                                            .upper_bounds = Vector{1.0},
                                            .residual_count = 1,
                                            .residual_function = {}},
                                           Vector{0.0}, kSolveRequest),
                 std::invalid_argument);

    PoundersSolveRequest invalid_request = kSolveRequest;
    invalid_request.gradient_absolute_tolerance = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(PoundersOptimizer{}.solve(valid_problem, Vector{0.0}, invalid_request),
                 std::invalid_argument);
    invalid_request = kSolveRequest;
    invalid_request.max_iterations = 0;
    EXPECT_THROW(PoundersOptimizer{}.solve(valid_problem, Vector{0.0}, invalid_request),
                 std::invalid_argument);
    invalid_request = kSolveRequest;
    invalid_request.max_function_evaluations = 0;
    EXPECT_THROW(PoundersOptimizer{}.solve(valid_problem, Vector{0.0}, invalid_request),
                 std::invalid_argument);
}

TEST(PoundersOptimizerTest, RejectsInvalidResidualVector) {
    ensure_petsc_test_session();
    const BoundedLeastSquaresProblem wrong_size{
        .lower_bounds = Vector{-1.0},
        .upper_bounds = Vector{1.0},
        .residual_count = 1,
        .residual_function = [](const Vector&) { return Vector{0.0, 0.0}; },
    };
    const BoundedLeastSquaresProblem nonfinite{
        .lower_bounds = Vector{-1.0},
        .upper_bounds = Vector{1.0},
        .residual_count = 1,
        .residual_function =
            [](const Vector&) { return Vector{std::numeric_limits<double>::infinity()}; },
    };

    EXPECT_THROW(PoundersOptimizer{}.solve(wrong_size, Vector{0.0}, kSolveRequest),
                 std::runtime_error);
    EXPECT_THROW(PoundersOptimizer{}.solve(nonfinite, Vector{0.0}, kSolveRequest),
                 std::runtime_error);
}

TEST(PoundersOptimizerTest, PropagatesResidualFunctionExceptionSafely) {
    ensure_petsc_test_session();
    const BoundedLeastSquaresProblem problem{
        .lower_bounds = Vector{-1.0},
        .upper_bounds = Vector{1.0},
        .residual_count = 1,
        .residual_function = [](const Vector&) -> Vector {
            throw std::domain_error("forward evaluation failed");
        },
    };

    EXPECT_THROW(PoundersOptimizer{}.solve(problem, Vector{0.0}, kSolveRequest), std::domain_error);
}

} // namespace
