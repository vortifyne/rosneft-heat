#include "numerics/linear/fgmres.hpp"

#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

LinearSolveResult FGMRES::solve(const SparseMatrix& A, const Vector& b, Vector& x,
                                const LinearSolveRequest& request) {
    const auto n = b.size();
    if (n == 0 || A.rows() != n || A.cols() != n || A.storage_order() != SparseStorageOrder::csr ||
        x.size() != n || !x.all_finite() || request.maxiter < 0 || !std::isfinite(request.rtol) ||
        request.rtol < 0 || !std::isfinite(request.atol) || request.atol < 0) {
        return {LinearSolveStatus::invalid_input, 0, std::numeric_limits<double>::infinity()};
    }

    const auto& matrix = A.native_csr();
    Eigen::VectorXd solution = x.native();
    Eigen::VectorXd residual = b.native() - matrix * solution;
    double beta = residual.norm();
    const double tolerance = std::max(request.atol, request.rtol * b.norm());

    constexpr int restart = 30;
    const int m = static_cast<int>(std::min<Eigen::Index>({restart, n, request.maxiter}));
    Eigen::MatrixXd V(n, m + 1), Z(n, m);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m + 1, m);
    Eigen::VectorXd w(n), y;
    Eigen::VectorXd g = Eigen::VectorXd::Zero(m + 1);

    LinearTimings timings;
    if (pc_ && beta > tolerance && m > 0) {
        const auto start = Clock::now();
        try {
            pc_->setup(A);
        } catch (const std::runtime_error&) {
            timings.setup_s = elapsed_seconds(start);
            return {LinearSolveStatus::factorization_failed, 0, beta, timings};
        }
        timings.setup_s = elapsed_seconds(start);
        timings.pc_updates = 1;
    }
    const auto start = Clock::now();
    int iterations = 0;
    while (beta > tolerance && iterations < request.maxiter) {
        V.col(0) = residual / beta;
        g(0) = beta;
        for (int j = 0; j < m && iterations < request.maxiter; ++j) {
            ++iterations;
            if (pc_) {
                try {
                    pc_->apply(A, V.col(j), Z.col(j));
                } catch (const std::runtime_error&) {
                    timings.solve_s = elapsed_seconds(start);
                    return {LinearSolveStatus::factorization_failed, iterations, beta, timings};
                }
            } else {
                Z.col(j) = V.col(j);
            }
            w = matrix * Z.col(j);

            for (int i = 0; i <= j; ++i) {
                H(i, j) = V.col(i).dot(w);
                w -= H(i, j) * V.col(i);
            }
            H(j + 1, j) = w.norm();

            const int k = j + 1;
            const auto Hk = H.topLeftCorner(k + 1, k);
            const auto gk = g.head(k + 1);
            const auto qr = Hk.colPivHouseholderQr();
            if (qr.rank() == 0) {
                timings.solve_s = elapsed_seconds(start);
                return {LinearSolveStatus::solve_failed, iterations, beta, timings};
            }
            y = qr.solve(gk);
            if ((gk - Hk * y).norm() <= tolerance || H(j + 1, j) == 0.0) {
                break;
            }
            V.col(j + 1) = w / H(j + 1, j);
        }

        solution += Z.leftCols(y.size()) * y;
        residual = b.native() - matrix * solution;
        beta = residual.norm();
    }

    timings.solve_s = elapsed_seconds(start);
    if (!std::isfinite(beta)) {
        return {LinearSolveStatus::nonfinite_solution, iterations, beta, timings};
    }
    if (beta > tolerance) {
        return {LinearSolveStatus::maxiter, iterations, beta, timings};
    }
    x.native().swap(solution);
    return {LinearSolveStatus::converged, iterations, beta, timings};
}
