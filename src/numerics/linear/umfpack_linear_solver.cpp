#include "numerics/linear/umfpack_linear_solver.hpp"

#include <Eigen/Core>
#include <Eigen/UmfPackSupport>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr int kDirectSolveIterations = 1;

LinearSolveResult failure(LinearSolveStatus status, int iterations = 0) {
    return {
        .status = status,
        .iterations = iterations,
        .r_norm = std::numeric_limits<double>::infinity(),
    };
}

bool matrix_values_are_finite(const SparseMatrix::CscNativeType& matrix) {
    for (SparseMatrix::Index index = 0; index < matrix.nonZeros(); ++index) {
        if (!std::isfinite(matrix.valuePtr()[index])) {
            return false;
        }
    }
    return true;
}

template <typename Dense>
LinearSolveResult solve_columns_in_place(const SparseMatrix& A, Dense& rhs) {
    if (A.storage_order() != SparseStorageOrder::csc || !A.is_compressed() || A.rows() == 0 ||
        A.rows() != A.cols() || A.rows() != rhs.rows() || rhs.cols() == 0 || !rhs.allFinite()) {
        return failure(LinearSolveStatus::invalid_input);
    }

    const auto& native_matrix = A.native_csc();
    if (!matrix_values_are_finite(native_matrix)) {
        return failure(LinearSolveStatus::invalid_input);
    }

    Eigen::UmfPackLU<SparseMatrix::CscNativeType> solver;
    solver.compute(native_matrix);

    if (solver.info() != Eigen::Success) {
        return failure(LinearSolveStatus::factorization_failed);
    }

    double residual_norm = 0.0;
    Eigen::VectorXd solution(A.rows());
    for (Eigen::Index j = 0; j < rhs.cols(); ++j) {
        // Eigen's solve expression discards the per-column UMFPACK status.
        if (!solver._solve_impl(rhs.col(j), solution))
            return failure(LinearSolveStatus::solve_failed, kDirectSolveIterations);
        const double norm = (native_matrix * solution - rhs.col(j)).norm();
        if (!std::isfinite(norm) || !solution.allFinite())
            return failure(LinearSolveStatus::nonfinite_solution, kDirectSolveIterations);
        residual_norm = std::max(residual_norm, norm);
        rhs.col(j) = solution;
    }
    return {
        .status = LinearSolveStatus::converged,
        .iterations = kDirectSolveIterations,
        .r_norm = residual_norm,
    };
}
} // namespace

LinearSolveResult UmfpackLinearSolver::solve(const SparseMatrix& A, const Vector& b, Vector& x,
                                             const LinearSolveRequest&) {
    Vector candidate = b;
    const auto result = solve_columns_in_place(A, candidate.native());
    if (result.status == LinearSolveStatus::converged)
        x = std::move(candidate);
    return result;
}

Eigen::MatrixXd UmfpackLinearSolver::solve_columns(const SparseMatrix& A, Eigen::MatrixXd rhs) {
    const auto result = solve_columns_in_place(A, rhs);
    if (result.status != LinearSolveStatus::converged)
        throw std::runtime_error("UMFPACK multiple-RHS solve failed, status " +
                                 std::to_string(static_cast<int>(result.status)));
    return rhs;
}
