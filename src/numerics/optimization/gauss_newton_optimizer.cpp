#include "numerics/optimization/gauss_newton_optimizer.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

LeastSquaresSolveResult GaussNewtonOptimizer::solve(const BoundedLeastSquaresProblem& problem,
                                                    const Vector& initial,
                                                    const LeastSquaresSolveRequest& request,
                                                    const LeastSquaresObserver& observe) {
    problem.validate(initial);
    request.validate();

    Vector x = initial;
    Eigen::MatrixXd J;
    double initial_gradient_norm = 0.0;
    for (int iteration = 0;; ++iteration) {
        auto [r, within_errors] = problem.evaluate(x, request.residual_tolerance, &J);
        if (r.size() != problem.residual_count || J.rows() != r.size() || J.cols() != x.size())
            throw std::runtime_error("Invalid residual or Jacobian dimensions");
        const double objective = least_squares_objective(r);
        if (!std::isfinite(objective) || !J.allFinite())
            throw std::runtime_error("Non-finite least-squares evaluation");
        const int evaluations = iteration + 1;
        if (observe)
            observe(iteration, evaluations, x, r);
        const auto finish = [&](LeastSquaresStatus status) {
            return LeastSquaresSolveResult{status,    std::move(x), std::move(r),
                                           objective, iteration,    evaluations};
        };
        if (within_errors)
            return finish(LeastSquaresStatus::within_errors);

        const Eigen::VectorXd g = J.transpose() * r.native();
        if (!g.allFinite())
            return finish(LeastSquaresStatus::failed);
        Eigen::VectorXd projected = g;
        for (Vector::Index i = 0; i < x.size(); ++i)
            if ((x[i] == problem.lower_bounds[i] && g[i] > 0.0) ||
                (x[i] == problem.upper_bounds[i] && g[i] < 0.0))
                projected[i] = 0.0;
        const double norm = projected.norm();
        if (!std::isfinite(norm))
            return finish(LeastSquaresStatus::failed);
        if (iteration == 0)
            initial_gradient_norm = norm;
        if (norm <= request.gatol || norm <= request.grtol * objective ||
            norm <= request.gttol * initial_gradient_norm)
            return finish(LeastSquaresStatus::converged);
        if (iteration >= request.maxiter || evaluations >= request.maxfev)
            return finish(LeastSquaresStatus::limit_reached);

        const Eigen::MatrixXd H = J.transpose() * J;
        if (!H.allFinite())
            return finish(LeastSquaresStatus::failed);
        linear_solver_.compute(H);
        Eigen::VectorXd next = x.native() + linear_solver_.solve(-g);
        if (!next.allFinite())
            return finish(LeastSquaresStatus::failed);
        next = next.cwiseMax(problem.lower_bounds.native()).cwiseMin(problem.upper_bounds.native());
        if (next == x.native())
            return finish(LeastSquaresStatus::failed);
        x.native().swap(next);
    }
}
