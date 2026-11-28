#include "numerics/linear/tdma_linear_solver.hpp"

#include <limits>
#include <tdma/tdma.hpp>

namespace {
template <class Dense> LinearSolveResult solve_in_place(const SparseMatrix& A, Dense& rhs) {
    LinearSolveResult result{.status = LinearSolveStatus::invalid_input,
                             .iterations = 0,
                             .r_norm = std::numeric_limits<double>::infinity()};
    if (A.storage_order() != MatrixStorageFormat::tridiagonal || A.rows() == 0 ||
        rhs.rows() != A.rows() || rhs.cols() == 0 || !rhs.allFinite())
        return result;

    const auto& bands = A.native_tridiagonal();
    tdma::PivotingSolver<double> solver;
    try {
        solver.compute({bands.lower(), bands.diagonal(), bands.upper()});
    } catch (const std::invalid_argument&) {
        return result;
    } catch (const std::runtime_error&) {
        result.status = LinearSolveStatus::factorization_failed;
        return result;
    }
    result.iterations = 1;
    result.r_norm = 0.;
    Eigen::VectorXd x(rhs.rows());
    for (Eigen::Index j = 0; j < rhs.cols(); ++j) {
        auto column = rhs.col(j);
        const auto n = static_cast<std::size_t>(x.size());
        solver.solve({column.data(), n}, {x.data(), n});
        const double norm = (bands * x - column).norm();
        if (!std::isfinite(norm)) {
            result.r_norm = norm;
            result.status = LinearSolveStatus::nonfinite_solution;
            return result;
        }
        result.r_norm = std::max(result.r_norm, norm);
        column = x;
    }
    result.status = LinearSolveStatus::converged;
    return result;
}
} // namespace

LinearSolveResult TdmaLinearSolver::solve(const SparseMatrix& A, const Vector& b, Vector& x,
                                          const LinearSolveRequest&) {
    Vector candidate = b;
    const auto result = solve_in_place(A, candidate.native());
    if (result.status == LinearSolveStatus::converged)
        x = std::move(candidate);
    return result;
}

Eigen::MatrixXd TdmaLinearSolver::solve_columns(const SparseMatrix& A, Eigen::MatrixXd rhs) {
    const auto result = solve_in_place(A, rhs);
    if (result.status != LinearSolveStatus::converged)
        throw std::runtime_error("TDMA multiple-RHS solve failed, status " +
                                 std::to_string(static_cast<int>(result.status)));
    return rhs;
}
