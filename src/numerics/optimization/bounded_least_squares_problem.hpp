#pragma once

#include "numerics/algebra/vector.hpp"

#include <functional>

using LeastSquaresResidualFunction = std::function<Vector(const Vector&)>;

[[nodiscard]] inline double least_squares_objective(const Vector& residuals) {
    return 0.5 * residuals.squared_norm();
}

struct LeastSquaresEvaluation {
    Vector residuals;
    bool within_errors;
};

struct BoundedLeastSquaresProblem {
    Vector lower_bounds;
    Vector upper_bounds;
    Vector::Index residual_count;
    LeastSquaresResidualFunction residual_function{};
    // Joint evaluation with the measurement tolerance and an optional weighted Jacobian.
    std::function<LeastSquaresEvaluation(const Vector&, double, Eigen::MatrixXd*)> objective{};

    [[nodiscard]] LeastSquaresEvaluation evaluate(const Vector& x, double tolerance,
                                                  Eigen::MatrixXd* jacobian = nullptr) const;

    void validate(const Vector& initial) const;
};

struct LeastSquaresSolveRequest {
    double gatol;
    double grtol;
    double gttol;
    int maxiter;
    int maxfev;
    double residual_tolerance = 0.0; // Stop when every residual is within this bound; 0 disables.

    void validate() const;
};

enum class LeastSquaresStatus {
    failed,
    converged,
    within_errors,
    limit_reached,
};

[[nodiscard]] constexpr bool is_least_squares_converged(LeastSquaresStatus status) noexcept {
    return status == LeastSquaresStatus::converged || status == LeastSquaresStatus::within_errors;
}

struct [[nodiscard]] LeastSquaresSolveResult {
    LeastSquaresStatus status;
    Vector parameters;
    Vector residuals;
    double objective; // Half the squared norm of the residual vector.
    int iterations;
    int function_evaluations;

    [[nodiscard]] constexpr bool converged() const noexcept {
        return is_least_squares_converged(status);
    }
};

// Optimizer iteration, residual evaluation, parameters, residuals.
using LeastSquaresObserver = std::function<void(int, int, const Vector&, const Vector&)>;
