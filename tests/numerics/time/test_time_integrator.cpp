#include "numerics/time/semi_discrete_system.hpp"
#include "numerics/time/time_integrator.hpp"
#include "support/petsc_test_session.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr NonlinearSolveRequest kNewtonRequest{
    .method = NonlinearMethod::newton,
    .rtol = 1e-12,
    .atol = 1e-12,
    .maxiter = 10,
};

constexpr LinearSolveRequest kDirectLinearRequest{
    .rtol = 1e-8,
    .atol = 1e-12,
    .maxiter = 100,
};

class DecaySystem final : public SemiDiscreteSystem {
public:
    explicit DecaySystem(double rate) : rate_(rate) {}

    void assemble_residual(double t, const Vector& u, const Vector& du_dt,
                           Vector& r) const override {
        evaluated_times_.push_back(t);
        r = du_dt + rate_ * u;
    }

    void assemble_matrix(NonlinearMethod method, double t, const Vector& u, const Vector& du_dt,
                         double alpha, SparseMatrix& A) const override {
        static_cast<void>(method);
        static_cast<void>(t);
        static_cast<void>(u);
        static_cast<void>(du_dt);
        const std::vector<MatrixTriplet> triplets = {
            {0, 0, alpha + rate_},
        };
        A.set_from_triplets(triplets);
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
    void assemble_residual(double t, const Vector& u, const Vector& du_dt,
                           Vector& r) const override {
        static_cast<void>(t);
        static_cast<void>(u);
        static_cast<void>(du_dt);
        r = Vector{1.0};
    }

    void assemble_matrix(NonlinearMethod method, double t, const Vector& u, const Vector& du_dt,
                         double alpha, SparseMatrix& A) const override {
        static_cast<void>(method);
        static_cast<void>(t);
        static_cast<void>(u);
        static_cast<void>(du_dt);
        static_cast<void>(alpha);
        A.set_zero();
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
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().t, 1.0);
    ASSERT_EQ(integrator.current_snapshot().u.size(), 1);
    EXPECT_NEAR(integrator.current_snapshot().u[0], 1.0 / (1.4 * 1.4 * 1.2), 1e-12);
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
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().u[0], 3.0);
}

TEST(TimeIntegratorTest, CanUseSuccessiveAdvanceTargetsAsExactEventTimes) {
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, Vector{1.0});
    integrator.set_timestep(0.4);
    DecaySystem system(1.0);

    const TimeIntegrationResult event_result =
        integrator.advance_to(system, 0.3, kNewtonRequest, kDirectLinearRequest);
    ASSERT_TRUE(event_result.completed());
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().t, 0.3);

    const TimeIntegrationResult final_result =
        integrator.advance_to(system, 0.8, kNewtonRequest, kDirectLinearRequest);

    EXPECT_TRUE(final_result.completed());
    EXPECT_EQ(final_result.accepted_steps, 2);
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().t, 0.8);
}

TEST(TimeIntegratorTest, DoesNotCreateRoundoffSizedFinalStep) {
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, Vector{1.0});
    integrator.set_timestep(0.025);
    DecaySystem system(1.0);

    const TimeIntegrationResult result =
        integrator.advance_to(system, 0.2, kNewtonRequest, kDirectLinearRequest);

    EXPECT_TRUE(result.completed());
    EXPECT_EQ(result.accepted_steps, 8);
    EXPECT_EQ(result.rejected_steps, 0);
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().t, 0.2);
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
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().t, 0.0);
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().u[0], 2.0);
}

TEST(TimeIntegratorTest, ReplacesInitialStateAndExposesCurrentSnapshot) {
    TimeIntegrator integrator;
    const auto* nonlinear_solver = &integrator.nonlinear_solver();
    const auto* linear_solver = &nonlinear_solver->linear_solver();
    integrator.set_initial_solution(0.0, Vector{1.0});
    integrator.set_initial_solution(3.0, Vector{4.0});
    integrator.reset_linear_solver();

    EXPECT_DOUBLE_EQ(integrator.current_snapshot().t, 3.0);
    EXPECT_DOUBLE_EQ(integrator.current_snapshot().u[0], 4.0);
    EXPECT_EQ(&integrator.nonlinear_solver(), nonlinear_solver);
    EXPECT_EQ(&integrator.nonlinear_solver().linear_solver(), linear_solver);
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
    EXPECT_THROW(
        static_cast<void>(integrator.advance_to(system, 1.0, kNewtonRequest, kDirectLinearRequest)),
        std::logic_error);

    integrator.set_initial_solution(2.0, Vector{1.0});
    integrator.set_timestep(0.5);
    EXPECT_THROW(
        static_cast<void>(integrator.advance_to(system, 1.0, kNewtonRequest, kDirectLinearRequest)),
        std::invalid_argument);
}

TEST(TimeIntegratorTest, ReportsWhenTimeStepCannotAdvanceFloatingPointTime) {
    TimeIntegrator integrator;
    const double t = 1e20;
    integrator.set_initial_solution(t, Vector{1.0});
    integrator.set_timestep(1.0);
    const DecaySystem system(1.0);

    const TimeIntegrationResult result =
        integrator.advance_to(system, std::nextafter(t, std::numeric_limits<double>::infinity()),
                              kNewtonRequest, kDirectLinearRequest);

    EXPECT_EQ(result.status, TimeIntegrationStatus::time_step_too_small);
    EXPECT_EQ(result.accepted_steps, 0);
}

} // namespace

namespace {

class ChangingDiagonalSystem final : public SemiDiscreteSystem {
public:
    void assemble_residual(double t, const Vector& u, const Vector& derivative,
                           Vector& r) const override {
        r = Vector{derivative[0] + (1.0 + t) * u[0], derivative[1] + (2.0 + 3.0 * t) * u[1]};
    }
    void assemble_matrix(NonlinearMethod, double t, const Vector&, const Vector&, double shift,
                         SparseMatrix& A) const override {
        const std::vector<MatrixTriplet> entries{{0, 0, shift + 1.0 + t},
                                                 {1, 1, shift + 2.0 + 3.0 * t}};
        A.set_from_triplets(entries);
    }
};

TEST(TimeIntegratorTest, ResetsStepPreconditionerButRetainsManuallyManagedPreconditioner) {
    ensure_petsc_test_session();
    ChangingDiagonalSystem system;
    for (const auto reuse : {PcReuse::none, PcReuse::step, PcReuse::manual}) {
        TimeIntegrator integrator;
        integrator.set_initial_solution(0.0, Vector{1.0, 1.0});
        integrator.set_timestep(1.0);
        integrator.set_linear_solver({.method = LinearMethod::gmres});
        integrator.set_preconditioner_reuse(reuse);
        const LinearSolveRequest linear{.rtol = 1e-10, .atol = 1e-12, .maxiter = 1};
        const auto result = integrator.advance_to(system, 2.0, kNewtonRequest, linear);
        if (reuse == PcReuse::manual) {
            EXPECT_EQ(result.status, TimeIntegrationStatus::nonlinear_solve_failed);
            EXPECT_EQ(result.accepted_steps, 1);
            EXPECT_DOUBLE_EQ(integrator.current_snapshot().t, 1.0);
            integrator.reset_linear_solver();
            EXPECT_TRUE(integrator.advance_to(system, 2.0, kNewtonRequest, linear).completed());
        } else {
            EXPECT_TRUE(result.completed());
            EXPECT_EQ(result.accepted_steps, 2);
        }
        EXPECT_NEAR(integrator.current_snapshot().u[0], 1.0 / 12.0, 1e-12);
        EXPECT_NEAR(integrator.current_snapshot().u[1], 1.0 / 54.0, 1e-12);
    }
}

} // namespace
