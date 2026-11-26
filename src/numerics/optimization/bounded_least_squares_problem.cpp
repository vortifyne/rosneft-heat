#include "numerics/optimization/bounded_least_squares_problem.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

void LeastSquaresSolveRequest::validate() const {
    if (!std::isfinite(gatol) || gatol < 0.0) {
        throw std::invalid_argument("Absolute gradient tolerance must be finite and non-negative");
    }
    if (!std::isfinite(grtol) || grtol < 0.0) {
        throw std::invalid_argument("Relative gradient tolerance must be finite and non-negative");
    }
    if (!std::isfinite(gttol) || gttol < 0.0) {
        throw std::invalid_argument("Gradient reduction tolerance must be finite and non-negative");
    }
    if (maxiter <= 0) {
        throw std::invalid_argument("Maximum iteration count must be positive");
    }
    if (maxfev <= 0) {
        throw std::invalid_argument("Maximum function evaluation count must be positive");
    }
    if (!std::isfinite(residual_tolerance) || residual_tolerance < 0.0)
        throw std::invalid_argument("Residual tolerance must be finite and non-negative");
}

void BoundedLeastSquaresProblem::validate(const Vector& x0) const {
    if (lower_bounds.empty()) {
        throw std::invalid_argument("Least-squares problem must have at least one parameter");
    }
    if (lower_bounds.size() != upper_bounds.size() || lower_bounds.size() != x0.size()) {
        throw std::invalid_argument("Parameter vectors and bound vectors must have the same size");
    }
    if (!lower_bounds.all_finite() || !upper_bounds.all_finite() || !x0.all_finite()) {
        throw std::invalid_argument("Parameters and bounds must be finite");
    }
    if (residual_count <= 0) {
        throw std::invalid_argument("Least-squares problem must have at least one residual");
    }

    for (Vector::Index index = 0; index < x0.size(); ++index) {
        if (!(lower_bounds[index] < upper_bounds[index])) {
            throw std::invalid_argument("Parameter bounds must be strictly increasing");
        }
        if (x0[index] < lower_bounds[index] || x0[index] > upper_bounds[index]) {
            throw std::invalid_argument("Initial parameters must lie within their bounds");
        }
    }
}

LeastSquaresEvaluation BoundedLeastSquaresProblem::evaluate(const Vector& x, double tolerance,
                                                            Eigen::MatrixXd* jacobian) const {
    if (objective)
        return objective(x, tolerance, jacobian);
    if (jacobian || !residual_function)
        throw std::invalid_argument("Missing least-squares evaluation for this optimizer");
    Vector r = residual_function(x);
    if (r.size() != residual_count)
        throw std::runtime_error("Invalid least-squares residual count");
    const bool within_errors = tolerance > 0.0 && r.infinity_norm() <= tolerance;
    return {std::move(r), within_errors};
}
