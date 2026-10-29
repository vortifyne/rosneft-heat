#include "linear/petsc_krylov_linear_solver.hpp"
#include "nonlinear/nonlinear_solver.hpp"
#include "petsc_test_session.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(PetscKrylovLinearSolver, SolvesCurrentMatrixForEveryMethodAndPreconditioner) {
    ensure_petsc_test_session();
    for (const auto method : {LinearMethod::gmres, LinearMethod::bicgstab}) {
        for (const auto pc : {Preconditioner::ilu, Preconditioner::ilu2, Preconditioner::gamg,
                              Preconditioner::boomeramg}) {
            for (const bool reuse : {false, true}) {
                SCOPED_TRACE(testing::Message() << "method=" << static_cast<int>(method) << " pc="
                                                << static_cast<int>(pc) << " reuse=" << reuse);
                NonlinearSolver nonlinear;
                nonlinear.set_linear_solver({.method = method, .preconditioner = pc});
                auto& solver = nonlinear.linear_solver();
                solver.set_preconditioner_reuse(reuse);
                ASSERT_EQ(solver.type(), LinearSolverType::iterative);
                ASSERT_EQ(solver.required_storage_order(), SparseStorageOrder::csr);
                for (const double diagonal : {4.0, 8.0}) {
                    SparseMatrix matrix(2, 2, SparseStorageOrder::csr);
                    const std::vector<MatrixTriplet> entries{
                        {0, 0, diagonal}, {0, 1, -1.0}, {1, 0, -2.0}, {1, 1, 5.0}};
                    matrix.set_from_triplets(entries);
                    matrix.make_compressed();
                    const Vector expected{2.0, -3.0};
                    Vector solution;
                    const auto result = solver.solve(matrix, matrix * expected, solution, {});
                    ASSERT_EQ(result.status, LinearSolveStatus::converged);
                    EXPECT_LT((solution - expected).norm(), 1.0e-7);
                }
                Vector unchanged{42.0};
                SparseMatrix invalid(2, 2, SparseStorageOrder::csc);
                EXPECT_EQ(solver.solve(invalid, Vector{0.0}, unchanged, {}).status,
                          LinearSolveStatus::invalid_input);
                EXPECT_DOUBLE_EQ(unchanged[0], 42.0);
            }
        }
    }
}

TEST(PetscKrylovLinearSolver, ExplicitResetRebuildsPreconditionerForChangedCoefficients) {
    ensure_petsc_test_session();
    for (const auto pc : {Preconditioner::ilu, Preconditioner::ilu2, Preconditioner::gamg,
                          Preconditioner::boomeramg}) {
        PetscKrylovLinearSolver solver(LinearMethod::gmres, pc);
        solver.set_preconditioner_reuse(true);
        const auto solve_diagonal = [&solver](double first, double second) {
            SparseMatrix matrix(2, 2, SparseStorageOrder::csr);
            const std::vector<MatrixTriplet> entries{{0, 0, first}, {1, 1, second}};
            matrix.set_from_triplets(entries);
            matrix.make_compressed();
            const Vector expected{2.0, -3.0};
            Vector solution;
            const auto result = solver.solve(matrix, matrix * expected, solution,
                                             {.relative_tolerance = 1.0e-10,
                                              .absolute_tolerance = 1.0e-12,
                                              .max_iterations = 1});
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
}
