#include "discretization/semi_discrete_system.hpp"
#include "time/time_integrator.hpp"

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
    .max_iterations = 10,
};

constexpr LinearSolveRequest kDirectLinearRequest{
    .relative_tolerance = 1e-8,
    .absolute_tolerance = 1e-12,
    .max_iterations = 100,
};

class DecaySystem final : public SemiDiscreteSystem {
public:
    explicit DecaySystem(double rate) : rate_(rate) {}

    void assemble_residual(double time, const Vector& solution, const Vector& solution_derivative,
                           Vector& residual) const override {
        evaluated_times_.push_back(time);
        residual = solution_derivative + rate_ * solution;
    }

    void assemble_matrix(NonlinearMethod method, double time, const Vector& solution,
                         const Vector& solution_derivative, double derivative_shift,
                         SparseMatrix& matrix) const override {
        static_cast<void>(method);
        static_cast<void>(time);
        static_cast<void>(solution);
        static_cast<void>(solution_derivative);
        const std::vector<MatrixTriplet> triplets = {
            {0, 0, derivative_shift + rate_},
        };
        matrix.set_from_triplets(triplets);
    }

    const std::vector<double>& evaluated_times() const {
        return evaluated_times_;
    }

private:
    double rate_;
    mutable std::vector<double> evaluated_times_;
};

class FailingSystem final : public SemiDiscreteSystem {
public:
    void assemble_residual(double time, const Vector& solution, const Vector& solution_derivative,
                           Vector& residual) const override {
        static_cast<void>(time);
        static_cast<void>(solution);
        static_cast<void>(solution_derivative);
        residual = Vector{1.0};
    }

    void assemble_matrix(NonlinearMethod method, double time, const Vector& solution,
                         const Vector& solution_derivative, double derivative_shift,
                         SparseMatrix& matrix) const override {
        static_cast<void>(method);
        static_cast<void>(time);
        static_cast<void>(solution);
        static_cast<void>(solution_derivative);
        static_cast<void>(derivative_shift);
        matrix.set_zero();
    }
};

TEST(TimeIntegratorTest, AdvancesWithFixedStepsAndMatchesFinalTimeExactly) {
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, Vector{1.0});
    integrator.set_timestep(0.4);
    DecaySystem system(1.0);

    const TimeIntegrationResult result =
        integrator.advance_to(system, 1.0, kNewtonRequest, kDirectLinearRequest);

    EXPECT_TRUE(result.completed());
    EXPECT_EQ(result.accepted_steps, 3);
    EXPECT_EQ(result.rejected_steps, 0);
    EXPECT_EQ(result.nonlinear_iterations, 3);
    EXPECT_EQ(result.linear_iterations, 3);
    ASSERT_TRUE(result.last_nonlinear_status.has_value());
    EXPECT_EQ(*result.last_nonlinear_status, NonlinearSolveStatus::converged_residual_absolute);
    ASSERT_TRUE(result.last_linear_status.has_value());
    EXPECT_EQ(*result.last_linear_status, LinearSolveStatus::converged);
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().time, 1.0);
    ASSERT_EQ(integrator.current_snapshot().solution.size(), 1);
    EXPECT_NEAR(integrator.current_snapshot().solution[0], 1.0 / (1.4 * 1.4 * 1.2), 1e-12);
    ASSERT_FALSE(system.evaluated_times().empty());
    EXPECT_DOUBLE_EQ(system.evaluated_times().back(), 1.0);
}

TEST(TimeIntegratorTest, ReturnsImmediatelyWhenAlreadyAtFinalTime) {
    TimeIntegrator integrator;
    integrator.set_initial_solution(2.0, Vector{3.0});
    integrator.set_timestep(0.5);
    DecaySystem system(1.0);

    const TimeIntegrationResult result =
        integrator.advance_to(system, 2.0, kNewtonRequest, kDirectLinearRequest);

    EXPECT_TRUE(result.completed());
    EXPECT_EQ(result.accepted_steps, 0);
    EXPECT_EQ(result.nonlinear_iterations, 0);
    EXPECT_EQ(result.linear_iterations, 0);
    EXPECT_FALSE(result.last_nonlinear_status.has_value());
    EXPECT_FALSE(result.last_linear_status.has_value());
    EXPECT_TRUE(system.evaluated_times().empty());
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().solution[0], 3.0);
}

TEST(TimeIntegratorTest, CanUseSuccessiveAdvanceTargetsAsExactEventTimes) {
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, Vector{1.0});
    integrator.set_timestep(0.4);
    DecaySystem system(1.0);

    const TimeIntegrationResult event_result =
        integrator.advance_to(system, 0.3, kNewtonRequest, kDirectLinearRequest);
    ASSERT_TRUE(event_result.completed());
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().time, 0.3);

    const TimeIntegrationResult final_result =
        integrator.advance_to(system, 0.8, kNewtonRequest, kDirectLinearRequest);

    EXPECT_TRUE(final_result.completed());
    EXPECT_EQ(final_result.accepted_steps, 2);
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().time, 0.8);
}

TEST(TimeIntegratorTest, StopsAtFirstNonlinearFailureAndKeepsAcceptedState) {
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, Vector{2.0});
    integrator.set_timestep(0.5);
    const FailingSystem system;

    const TimeIntegrationResult result =
        integrator.advance_to(system, 1.0, kNewtonRequest, kDirectLinearRequest);

    EXPECT_EQ(result.status, TimeIntegrationStatus::nonlinear_solve_failed);
    EXPECT_FALSE(result.completed());
    EXPECT_EQ(result.accepted_steps, 0);
    EXPECT_EQ(result.rejected_steps, 1);
    EXPECT_EQ(result.nonlinear_iterations, 0);
    EXPECT_EQ(result.linear_iterations, 0);
    ASSERT_TRUE(result.last_nonlinear_status.has_value());
    EXPECT_EQ(*result.last_nonlinear_status, NonlinearSolveStatus::linear_solve_failed);
    ASSERT_TRUE(result.last_linear_status.has_value());
    EXPECT_EQ(*result.last_linear_status, LinearSolveStatus::factorization_failed);
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().time, 0.0);
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().solution[0], 2.0);
}

TEST(TimeIntegratorTest, ReplacesInitialStateAndExposesCurrentSnapshot) {
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, Vector{1.0});
    integrator.set_initial_solution(3.0, Vector{4.0});

    EXPECT_DOUBLE_EQ(integrator.current_snapshot().time, 3.0);
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().solution[0], 4.0);
}

TEST(TimeIntegratorTest, RejectsInvalidConfigurationAndTimeDirection) {
    TimeIntegrator integrator;
    const DecaySystem system(1.0);

    EXPECT_THROW(integrator.current_snapshot(), std::logic_error);
    EXPECT_THROW(integrator.set_timestep(0.0), std::invalid_argument);
    EXPECT_THROW(integrator.set_timestep(-1.0), std::invalid_argument);
    EXPECT_THROW(integrator.set_timestep(std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    EXPECT_THROW(
        integrator.set_initial_solution(std::numeric_limits<double>::quiet_NaN(), Vector{1.0}),
        std::invalid_argument);
    EXPECT_THROW(integrator.advance_to(system, 1.0, kNewtonRequest, kDirectLinearRequest),
                 std::logic_error);

    integrator.set_initial_solution(2.0, Vector{1.0});
    integrator.set_timestep(0.5);
    EXPECT_THROW(integrator.advance_to(system, 1.0, kNewtonRequest, kDirectLinearRequest),
                 std::invalid_argument);
}

TEST(TimeIntegratorTest, ReportsWhenTimeStepCannotAdvanceFloatingPointTime) {
    TimeIntegrator integrator;
    const double current_time = 1e20;
    integrator.set_initial_solution(current_time, Vector{1.0});
    integrator.set_timestep(1.0);
    const DecaySystem system(1.0);

    const TimeIntegrationResult result = integrator.advance_to(
        system, std::nextafter(current_time, std::numeric_limits<double>::infinity()),
        kNewtonRequest, kDirectLinearRequest);

    EXPECT_EQ(result.status, TimeIntegrationStatus::time_step_too_small);
    EXPECT_EQ(result.accepted_steps, 0);
}

TEST(TimeIntegratorTest, DelegatesLinearSolverSelection) {
    TimeIntegrator integrator;

    EXPECT_NO_THROW(integrator.set_linear_solver(LinearSolverKind::umfpack_lu));
}

} // namespace
