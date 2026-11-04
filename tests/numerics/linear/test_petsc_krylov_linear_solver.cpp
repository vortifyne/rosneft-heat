#include "numerics/linear/petsc_krylov_linear_solver.hpp"
#include "numerics/nonlinear/nonlinear_solver.hpp"
#include "support/petsc_test_session.hpp"

#include <gtest/gtest.h>
#include <petscsys.h>
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
                                             {.rtol = 1.0e-10, .atol = 1.0e-12, .maxiter = 1});
            if (result.status == LinearSolveStatus::converged) {
                EXPECT_LT((solution - expected).norm(), 1.0e-8);
            }
            return result.status;
        };
        ASSERT_EQ(solve_diagonal(2.0, 3.0), LinearSolveStatus::converged);
        EXPECT_EQ(solve_diagonal(7.0, 11.0), LinearSolveStatus::maxiter);
        solver.reset();
        EXPECT_EQ(solve_diagonal(7.0, 11.0), LinearSolveStatus::converged);
    }
}

TEST(PetscKrylovLinearSolver, UpdatesValuesAndReusesObjectsAtExplicitBoundaries) {
    ensure_petsc_test_session();
    PetscKrylovLinearSolver solver;
    Vector solution;
    for (int step = 0; step < 3; ++step) {
        const int size = step == 2 ? 3 : 2;
        SparseMatrix matrix(size, size, SparseStorageOrder::csr);
        std::vector<MatrixTriplet> entries;
        for (int i = 0; i < size; ++i)
            entries.emplace_back(i, i, 2.0 + step + i);
        matrix.set_from_triplets(entries);
        Vector expected(size);
        expected.set_constant(3.0);
        const Vector rhs = matrix * expected;
        const auto result =
            solver.solve(matrix, rhs, solution, {.new_pattern = step != 1, .new_solve = step != 1});
        ASSERT_EQ(result.status, LinearSolveStatus::converged);
        EXPECT_LT((solution - expected).norm(), 1.0e-10);
        EXPECT_EQ((rhs - matrix * expected).norm(), 0.0);
        EXPECT_EQ(result.timings.ksp_builds, step == 0 ? 1 : 0);
        EXPECT_EQ(result.timings.vector_builds, step == 1 ? 0 : 2);
        EXPECT_EQ(result.timings.matrix_builds, step == 1 ? 0 : 1);
        EXPECT_EQ(result.timings.pc_updates, 1);
    }
}

TEST(PetscKrylovLinearSolver, NativeOptionsOverrideApplicationDefaults) {
    ensure_petsc_test_session();
    PetscOptionsSetValue(nullptr, "-pc_type", "none");
    PetscOptionsSetValue(nullptr, "-ksp_max_it", "1");
    PetscKrylovLinearSolver solver;
    SparseMatrix matrix(2, 2, SparseStorageOrder::csr);
    const std::vector<MatrixTriplet> entries{{0, 0, 2.0}, {1, 1, 3.0}};
    matrix.set_from_triplets(entries);
    Vector solution{42.0};
    const auto result = solver.solve(matrix, Vector{2.0, 3.0}, solution, {.maxiter = 10});
    PetscOptionsClearValue(nullptr, "-ksp_max_it");
    PetscOptionsClearValue(nullptr, "-pc_type");
    EXPECT_EQ(result.status, LinearSolveStatus::maxiter);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_DOUBLE_EQ(solution[0], 42.0);
}
