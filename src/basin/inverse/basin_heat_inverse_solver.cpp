#include "basin/inverse/basin_heat_inverse_solver.hpp"

#include "numerics/optimization/gauss_newton_optimizer.hpp"
#include "numerics/optimization/pounders_optimizer.hpp"

#include <utility>

BasinHeatInverseSolver::BasinHeatInverseSolver(BasinObjective objective)
    : objective_(std::move(objective)) {}

Vector BasinHeatInverseSolver::solve(const BasinHeatInverseSolveRequest& request,
                                     const BasinInverseObserver& observe) {
    const auto& space = objective_.parameter_space();
    const BoundedLeastSquaresProblem problem{
        .lower_bounds = space.normalized_lower_bounds(),
        .upper_bounds = space.normalized_upper_bounds(),
        .residual_count =
            objective_.observations().size() + space.gradient_penalty_size(request.regularization),
        .objective =
            [&](const Vector& eta, double tolerance, Eigen::MatrixXd* jacobian) {
                return objective_.evaluate(eta, tolerance, jacobian, request.regularization);
            },
    };
    const Vector initial = space.to_normalized(request.initial_parameters);
    const auto report = [&](int iteration, int evaluation, const Vector& eta,
                            const Vector& residuals) {
        if (observe)
            observe(iteration, evaluation, space.to_physical(eta), residuals);
    };
    if (request.method == BasinInverseMethod::gauss_newton)
        result_ = GaussNewtonOptimizer{}.solve(problem, initial, request.optimization, report);
    else
        result_ = PoundersOptimizer{}.solve(problem, initial, request.optimization, report);
    return space.to_physical(result_.parameters);
}

const LeastSquaresSolveResult& BasinHeatInverseSolver::optimization_result() const noexcept {
    return result_;
}

BasinObjective& BasinHeatInverseSolver::objective() noexcept {
    return objective_;
}
