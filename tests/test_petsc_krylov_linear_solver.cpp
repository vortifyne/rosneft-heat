#include "linear/petsc_krylov_linear_solver.hpp"
#include "nonlinear/nonlinear_solver.hpp"
#include "petsc_test_session.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(PetscKrylovLinearSolver, TransfersCsrAndVectorsAndRebuildsForNewMatrix) {
    ensure_petsc_test_session();
    for (const auto kind :
         {LinearSolverKind::petsc_gmres_ilu, LinearSolverKind::petsc_gmres_ilu2,
          LinearSolverKind::petsc_bicgstab_ilu, LinearSolverKind::petsc_bicgstab_ilu2}) {
        NonlinearSolver nonlinear;
        nonlinear.set_linear_solver(kind);
        auto& solver = nonlinear.linear_solver();
        ASSERT_EQ(solver.type(), LinearSolverType::iterative);
        ASSERT_EQ(solver.required_storage_order(), SparseStorageOrder::csr);
        for (const double diagonal : {4.0, 8.0}) {
            SparseMatrix matrix(2, 2, SparseStorageOrder::csr);
            const std::vector<MatrixTriplet> entries{
                {0, 0, diagonal}, {0, 1, -1.0}, {1, 0, -2.0}, {1, 1, 5.0}};
            matrix.set_from_triplets(entries.begin(), entries.end());
            matrix.make_compressed();
            Vector expected(2);
            expected[0] = 2.0;
            expected[1] = -3.0;
            Vector solution;
            const auto result = solver.solve(matrix, matrix * expected, solution, {});
            ASSERT_EQ(result.status, LinearSolveStatus::converged);
            EXPECT_LT((solution - expected).norm(), 1.0e-7);
        }
        Vector invalid_rhs(1);
        invalid_rhs.set_zero();
        Vector unchanged(1);
        unchanged[0] = 42.0;
        SparseMatrix invalid(2, 2, SparseStorageOrder::csc);
        EXPECT_EQ(solver.solve(invalid, invalid_rhs, unchanged, {}).status,
                  LinearSolveStatus::invalid_input);
        EXPECT_DOUBLE_EQ(unchanged[0], 42.0);
    }
}

TEST(PetscKrylovLinearSolver, ReusedAmgSolvesTheCurrentMatrix) {
    ensure_petsc_test_session();
    for (const auto kind : {LinearSolverKind::petsc_gmres_gamg_reuse,
                            LinearSolverKind::petsc_gmres_boomeramg_reuse}) {
        NonlinearSolver nonlinear;
        nonlinear.set_linear_solver(kind);
        for (const double diagonal : {4.0, 8.0}) {
            SparseMatrix matrix(2, 2, SparseStorageOrder::csr);
            const std::vector<MatrixTriplet> entries{
                {0, 0, diagonal}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 5.0}};
            matrix.set_from_triplets(entries.begin(), entries.end());
            matrix.make_compressed();
            Vector expected(2);
            expected[0] = 2.0;
            expected[1] = -3.0;
            Vector solution;
            const auto result =
                nonlinear.linear_solver().solve(matrix, matrix * expected, solution, {});
            ASSERT_EQ(result.status, LinearSolveStatus::converged);
            EXPECT_LT((solution - expected).norm(), 1.0e-7);
        }
    }
}

TEST(PetscKrylovLinearSolver, ExplicitResetRebuildsAmgForChangedCoefficients) {
    ensure_petsc_test_session();
    PetscKrylovLinearSolver solver(PetscKrylovMethod::gmres, PetscPreconditioner::gamg, true);
    const auto solve_diagonal = [&solver](double first, double second) {
        SparseMatrix matrix(2, 2, SparseStorageOrder::csr);
        const std::vector<MatrixTriplet> entries{{0, 0, first}, {1, 1, second}};
        matrix.set_from_triplets(entries);
        matrix.make_compressed();
        const Vector expected{2.0, -3.0};
        Vector solution;
        const auto result = solver.solve(
            matrix, matrix * expected, solution,
            {.relative_tolerance = 1.0e-10, .absolute_tolerance = 1.0e-12, .max_iterations = 1});
        if (result.status == LinearSolveStatus::converged) {
            EXPECT_LT((solution - expected).norm(), 1.0e-8);
        }
        return result.status;
    };
    ASSERT_EQ(solve_diagonal(2.0, 3.0), LinearSolveStatus::converged);
    EXPECT_EQ(solve_diagonal(7.0, 11.0), LinearSolveStatus::max_iterations);
    solver.reset();
    EXPECT_EQ(solve_diagonal(7.0, 11.0), LinearSolveStatus::converged);
}
