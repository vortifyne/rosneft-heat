#include "nonlinear/nonlinear_solver.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

constexpr NonlinearSolveRequest kNewtonRequest{
    .nonlinear_method = NonlinearMethod::newton,
    .relative_tolerance = 1e-12,
    .absolute_tolerance = 1e-12,
    .step_relative_tolerance = 1e-12,
    .max_iterations = 20,
};

constexpr LinearSolveRequest kDirectLinearRequest{
    .relative_tolerance = 1e-8,
    .absolute_tolerance = 1e-12,
    .max_iterations = 100,
};

class ScalarNonlinearSystem final : public NonlinearSystem {
public:
    enum class ResidualKind {
        quadratic,
        constant,
        nonfinite_after_update,
    };

    explicit ScalarNonlinearSystem(ResidualKind residual_kind = ResidualKind::quadratic,
                                   double matrix_coefficient = 1.0)
        : residual_kind_(residual_kind), matrix_coefficient_(matrix_coefficient) {}

    void assemble_residual(const Vector& x, Vector& F) const override {
        F.resize(1);
        switch (residual_kind_) {
        case ResidualKind::quadratic:
            F[0] = x[0] * x[0] - 2.0;
            break;
        case ResidualKind::constant:
            F[0] = 1.0;
            break;
        case ResidualKind::nonfinite_after_update:
            F[0] = x[0] == 0.0 ? 1.0 : std::numeric_limits<double>::quiet_NaN();
            break;
        }
    }

    mutable int picard_assemblies = 0;
    mutable int newton_assemblies = 0;
    mutable SparseStorageOrder last_storage_order = SparseStorageOrder::csr;

private:
    void assemble_picard_matrix(const Vector& x, SparseMatrix& A) const override {
        static_cast<void>(x);
        ++picard_assemblies;
        assemble_scalar_matrix(A, matrix_coefficient_);
    }

    void assemble_newton_matrix(const Vector& x, SparseMatrix& A) const override {
        ++newton_assemblies;
        const double coefficient =
            residual_kind_ == ResidualKind::quadratic ? 2.0 * x[0] : matrix_coefficient_;
        assemble_scalar_matrix(A, coefficient);
    }

    void assemble_scalar_matrix(SparseMatrix& A, double coefficient) const {
        last_storage_order = A.storage_order();
        const std::vector<MatrixTriplet> triplets = {{0, 0, coefficient}};
        A.set_from_triplets(triplets);
    }

    ResidualKind residual_kind_;
    double matrix_coefficient_;
};

TEST(NonlinearSolverTest, UsesUmfpackByDefaultAndCanSelectItAtRuntime) {
    NonlinearSolver solver;

    EXPECT_EQ(solver.linear_solver_kind(), LinearSolverKind::umfpack_lu);
    EXPECT_EQ(solver.linear_solver_type(), LinearSolverType::direct);

    solver.set_linear_solver(LinearSolverKind::umfpack_lu);

    EXPECT_EQ(solver.linear_solver_kind(), LinearSolverKind::umfpack_lu);
    EXPECT_EQ(solver.linear_solver_type(), LinearSolverType::direct);
}

TEST(NonlinearSolverTest, ConvergesWithoutIterationWhenInitialResidualIsSmall) {
    ScalarNonlinearSystem system;
    Vector x{1.0};
    NonlinearSolver solver;
    NonlinearSolveRequest request = kNewtonRequest;
    request.absolute_tolerance = 2.0;

    const NonlinearSolveResult result = solver.solve(system, x, request, kDirectLinearRequest);

    EXPECT_EQ(result.status, NonlinearSolveStatus::converged_residual_absolute);
    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.linear_iterations, 0);
    EXPECT_FALSE(result.last_linear_status.has_value());
    EXPECT_EQ(system.newton_assemblies, 0);
    EXPECT_DOUBLE_EQ(x[0], 1.0);
}

TEST(NonlinearSolverTest, NewtonMethodConvergesAndBuildsCscMatrices) {
    ScalarNonlinearSystem system;
    Vector x{1.0};
    NonlinearSolver solver;

    const NonlinearSolveResult result =
        solver.solve(system, x, kNewtonRequest, kDirectLinearRequest);

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.status, NonlinearSolveStatus::converged_residual_absolute);
    EXPECT_GT(result.iterations, 1);
    EXPECT_EQ(result.linear_iterations, result.iterations);
    ASSERT_TRUE(result.last_linear_status.has_value());
    EXPECT_EQ(*result.last_linear_status, LinearSolveStatus::converged);
    EXPECT_NEAR(x[0], 1.4142135623730951, 1e-12);
    EXPECT_EQ(system.picard_assemblies, 0);
    EXPECT_EQ(system.newton_assemblies, result.iterations);
    EXPECT_EQ(system.last_storage_order, SparseStorageOrder::csc);
}

TEST(NonlinearSolverTest, UsesRequestedPicardMatrix) {
    ScalarNonlinearSystem system(ScalarNonlinearSystem::ResidualKind::constant, 2.0);
    Vector x{1.0};
    NonlinearSolver solver;
    NonlinearSolveRequest request = kNewtonRequest;
    request.nonlinear_method = NonlinearMethod::picard;
    request.relative_tolerance = 0.0;
    request.absolute_tolerance = 0.0;
    request.step_relative_tolerance = 2.0;

    const NonlinearSolveResult result = solver.solve(system, x, request, kDirectLinearRequest);

    EXPECT_EQ(result.status, NonlinearSolveStatus::converged_step);
    EXPECT_EQ(system.picard_assemblies, 1);
    EXPECT_EQ(system.newton_assemblies, 0);
}

TEST(NonlinearSolverTest, DetectsRelativeResidualConvergence) {
    ScalarNonlinearSystem system;
    Vector x{1.0};
    NonlinearSolver solver;
    NonlinearSolveRequest request = kNewtonRequest;
    request.relative_tolerance = 0.3;
    request.absolute_tolerance = 0.0;
    request.step_relative_tolerance = 0.0;

    const NonlinearSolveResult result = solver.solve(system, x, request, kDirectLinearRequest);

    EXPECT_EQ(result.status, NonlinearSolveStatus::converged_residual_relative);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_DOUBLE_EQ(result.final_residual_norm, 0.25);
}

TEST(NonlinearSolverTest, DetectsSmallStepConvergence) {
    ScalarNonlinearSystem system(ScalarNonlinearSystem::ResidualKind::constant, 1e12);
    Vector x{1.0};
    NonlinearSolver solver;
    NonlinearSolveRequest request = kNewtonRequest;
    request.relative_tolerance = 0.0;
    request.absolute_tolerance = 0.0;
    request.step_relative_tolerance = 1e-6;

    const NonlinearSolveResult result = solver.solve(system, x, request, kDirectLinearRequest);

    EXPECT_EQ(result.status, NonlinearSolveStatus::converged_step);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_DOUBLE_EQ(result.final_residual_norm, 1.0);
}

TEST(NonlinearSolverTest, ReportsMaximumIterationsAndKeepsLastIterate) {
    ScalarNonlinearSystem system(ScalarNonlinearSystem::ResidualKind::constant);
    Vector x{0.0};
    NonlinearSolver solver;
    NonlinearSolveRequest request = kNewtonRequest;
    request.relative_tolerance = 0.0;
    request.absolute_tolerance = 0.0;
    request.step_relative_tolerance = 0.0;
    request.max_iterations = 2;

    const NonlinearSolveResult result = solver.solve(system, x, request, kDirectLinearRequest);

    EXPECT_EQ(result.status, NonlinearSolveStatus::max_iterations);
    EXPECT_FALSE(result.converged());
    EXPECT_EQ(result.iterations, 2);
    EXPECT_DOUBLE_EQ(x[0], -2.0);
}

TEST(NonlinearSolverTest, ReportsInitialNonfiniteResidual) {
    ScalarNonlinearSystem system;
    Vector x{std::numeric_limits<double>::quiet_NaN()};
    NonlinearSolver solver;

    const NonlinearSolveResult result =
        solver.solve(system, x, kNewtonRequest, kDirectLinearRequest);

    EXPECT_EQ(result.status, NonlinearSolveStatus::nonfinite_residual);
    EXPECT_EQ(result.iterations, 0);
}

TEST(NonlinearSolverTest, DoesNotAcceptCandidateWithNonfiniteResidual) {
    ScalarNonlinearSystem system(ScalarNonlinearSystem::ResidualKind::nonfinite_after_update);
    Vector x{0.0};
    NonlinearSolver solver;

    const NonlinearSolveResult result =
        solver.solve(system, x, kNewtonRequest, kDirectLinearRequest);

    EXPECT_EQ(result.status, NonlinearSolveStatus::nonfinite_residual);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_DOUBLE_EQ(x[0], 0.0);
}

TEST(NonlinearSolverTest, ReportsLinearSolveFailure) {
    ScalarNonlinearSystem system(ScalarNonlinearSystem::ResidualKind::constant, 0.0);
    Vector x{0.0};
    NonlinearSolver solver;

    const NonlinearSolveResult result =
        solver.solve(system, x, kNewtonRequest, kDirectLinearRequest);

    EXPECT_EQ(result.status, NonlinearSolveStatus::linear_solve_failed);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.linear_iterations, 0);
    ASSERT_TRUE(result.last_linear_status.has_value());
    EXPECT_EQ(*result.last_linear_status, LinearSolveStatus::factorization_failed);
    EXPECT_DOUBLE_EQ(x[0], 0.0);
}

} // namespace
