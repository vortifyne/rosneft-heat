#include "linear/umfpack_linear_solver.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

constexpr LinearSolveRequest kUnusedDirectSolveRequest{
    .relative_tolerance = 1e-8,
    .absolute_tolerance = 1e-12,
    .max_iterations = 100,
};

SparseMatrix make_test_matrix(SparseStorageOrder storage_order = SparseStorageOrder::csc) {
    SparseMatrix matrix(3, 3, storage_order);
    const std::vector<MatrixTriplet> triplets = {
        {0, 0, 4.0},  {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 4.0},
        {1, 2, -1.0}, {2, 1, -1.0}, {2, 2, 3.0},
    };
    matrix.set_from_triplets(triplets);
    return matrix;
}

TEST(UmfpackLinearSolverTest, IsDirectSolver) {
    const UmfpackLinearSolver solver;

    EXPECT_EQ(solver.type(), LinearSolverType::direct);
}

TEST(UmfpackLinearSolverTest, SolvesCscSystemAndReportsResidual) {
    const SparseMatrix matrix = make_test_matrix();
    const Vector right_hand_side{15.0, 10.0, 10.0};
    Vector solution;
    UmfpackLinearSolver solver;

    const LinearSolveResult result =
        solver.solve(matrix, right_hand_side, solution, kUnusedDirectSolveRequest);

    EXPECT_EQ(result.status, LinearSolveStatus::converged);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_LE(result.final_residual_norm, 1e-12);
    ASSERT_EQ(solution.size(), 3);
    EXPECT_NEAR(solution[0], 5.0, 1e-12);
    EXPECT_NEAR(solution[1], 5.0, 1e-12);
    EXPECT_NEAR(solution[2], 5.0, 1e-12);
}

TEST(UmfpackLinearSolverTest, RejectsCsrWithoutImplicitConversion) {
    const SparseMatrix matrix = make_test_matrix(SparseStorageOrder::csr);
    const Vector right_hand_side{15.0, 10.0, 10.0};
    Vector solution{7.0};
    UmfpackLinearSolver solver;

    const LinearSolveResult result =
        solver.solve(matrix, right_hand_side, solution, kUnusedDirectSolveRequest);

    EXPECT_EQ(result.status, LinearSolveStatus::invalid_input);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_TRUE(std::isinf(result.final_residual_norm));
    ASSERT_EQ(solution.size(), 1);
    EXPECT_DOUBLE_EQ(solution[0], 7.0);
}

TEST(UmfpackLinearSolverTest, RejectsUncompressedCscMatrix) {
    SparseMatrix matrix = make_test_matrix();
    matrix.native_csc().coeffRef(0, 2) = 1.0;
    const Vector right_hand_side{15.0, 10.0, 10.0};
    Vector solution;
    UmfpackLinearSolver solver;

    ASSERT_FALSE(matrix.is_compressed());
    const LinearSolveResult result =
        solver.solve(matrix, right_hand_side, solution, kUnusedDirectSolveRequest);

    EXPECT_EQ(result.status, LinearSolveStatus::invalid_input);
}

TEST(UmfpackLinearSolverTest, RejectsNonSquareMatrix) {
    const SparseMatrix matrix(2, 3, SparseStorageOrder::csc);
    const Vector right_hand_side{1.0, 2.0};
    Vector solution;
    UmfpackLinearSolver solver;

    const LinearSolveResult result =
        solver.solve(matrix, right_hand_side, solution, kUnusedDirectSolveRequest);

    EXPECT_EQ(result.status, LinearSolveStatus::invalid_input);
}

TEST(UmfpackLinearSolverTest, RejectsRightHandSideSizeMismatch) {
    const SparseMatrix matrix = make_test_matrix();
    const Vector right_hand_side{1.0, 2.0};
    Vector solution;
    UmfpackLinearSolver solver;

    const LinearSolveResult result =
        solver.solve(matrix, right_hand_side, solution, kUnusedDirectSolveRequest);

    EXPECT_EQ(result.status, LinearSolveStatus::invalid_input);
}

TEST(UmfpackLinearSolverTest, RejectsNonfiniteInput) {
    SparseMatrix matrix(1, 1, SparseStorageOrder::csc);
    const std::vector<MatrixTriplet> triplets = {
        {0, 0, std::numeric_limits<double>::infinity()},
    };
    matrix.set_from_triplets(triplets);
    const Vector right_hand_side{1.0};
    Vector solution;
    UmfpackLinearSolver solver;

    const LinearSolveResult result =
        solver.solve(matrix, right_hand_side, solution, kUnusedDirectSolveRequest);

    EXPECT_EQ(result.status, LinearSolveStatus::invalid_input);
}

TEST(UmfpackLinearSolverTest, ReportsFactorizationFailureForSingularMatrix) {
    SparseMatrix matrix(2, 2, SparseStorageOrder::csc);
    const std::vector<MatrixTriplet> triplets = {
        {0, 0, 1.0},
        {0, 1, 1.0},
        {1, 0, 2.0},
        {1, 1, 2.0},
    };
    matrix.set_from_triplets(triplets);
    const Vector right_hand_side{1.0, 2.0};
    Vector solution;
    UmfpackLinearSolver solver;

    const LinearSolveResult result =
        solver.solve(matrix, right_hand_side, solution, kUnusedDirectSolveRequest);

    EXPECT_EQ(result.status, LinearSolveStatus::factorization_failed);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_TRUE(std::isinf(result.final_residual_norm));
}

} // namespace
