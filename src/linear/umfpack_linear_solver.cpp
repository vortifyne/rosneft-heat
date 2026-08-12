#include "linear/umfpack_linear_solver.hpp"

#include <Eigen/Core>
#include <Eigen/UmfPackSupport>
#include <cmath>
#include <limits>
#include <utility>

namespace {

constexpr int kDirectSolveIterations = 1;

LinearSolveResult failure(LinearSolveStatus status, int iterations = 0) {
    return {
        .status = status,
        .iterations = iterations,
        .final_residual_norm = std::numeric_limits<double>::infinity(),
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

} // namespace

LinearSolveResult UmfpackLinearSolver::solve(const SparseMatrix& A, const Vector& b, Vector& x,
                                             const LinearSolveRequest& request) {
    static_cast<void>(request);

    if (A.storage_order() != SparseStorageOrder::csc || !A.is_compressed() || A.rows() == 0 ||
        A.rows() != A.cols() || A.rows() != b.size() || !b.all_finite()) {
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

    Vector::NativeType native_solution = solver.solve(b.native());
    if (solver.info() != Eigen::Success) {
        return failure(LinearSolveStatus::solve_failed, kDirectSolveIterations);
    }

    Vector x_candidate(std::move(native_solution));
    if (!x_candidate.all_finite()) {
        return failure(LinearSolveStatus::nonfinite_solution, kDirectSolveIterations);
    }

    const double residual_norm = (A * x_candidate - b).norm();
    if (!std::isfinite(residual_norm)) {
        return failure(LinearSolveStatus::nonfinite_solution, kDirectSolveIterations);
    }

    x = std::move(x_candidate);
    return {
        .status = LinearSolveStatus::converged,
        .iterations = kDirectSolveIterations,
        .final_residual_norm = residual_norm,
    };
}
