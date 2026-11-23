#include "basin/inverse/basin_heat_inverse_solver.hpp"

#include <utility>

BasinHeatInverseSolver::BasinHeatInverseSolver(BasinHeatResidualEvaluator residual_evaluator)
    : residual_evaluator_(std::move(residual_evaluator)) {}

Vector BasinHeatInverseSolver::solve(const BasinHeatInverseSolveRequest& request,
                                     const BasinInverseObserver& observe) {
    const auto& space = residual_evaluator_.parameter_space();
    const BoundedLeastSquaresProblem problem{
        .lower_bounds = space.normalized_lower_bounds(),
        .upper_bounds = space.normalized_upper_bounds(),
        .residual_count = residual_evaluator_.observations().size(),
        .residual_function =
            [&](const Vector& normalized) { return residual_evaluator_.evaluate(normalized); },
    };
    result_ = optimizer_.solve(
        problem, space.to_normalized(request.initial_parameters), request.optimization,
        [&](int iteration, int evaluation, const Vector& normalized, const Vector& residuals) {
            if (observe)
                observe(iteration, evaluation, space.to_physical(normalized), residuals);
        });
    return space.to_physical(result_.parameters);
}

const PoundersSolveResult& BasinHeatInverseSolver::optimization_result() const noexcept {
    return result_;
}

BasinHeatResidualEvaluator& BasinHeatInverseSolver::residual_evaluator() noexcept {
    return residual_evaluator_;
}
