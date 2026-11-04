#include "basin/inverse/basin_heat_inverse_solver.hpp"

#include <utility>

BasinHeatInverseSolver::BasinHeatInverseSolver(BasinHeatResidualEvaluator residual_evaluator)
    : residual_evaluator_(std::move(residual_evaluator)) {}

Vector BasinHeatInverseSolver::solve(const BasinHeatInverseSolveRequest& request,
                                     const BasinInverseObserver& observe) {
    const auto& space = residual_evaluator_.parameter_space();
    int evaluations = 0;
    const BoundedLeastSquaresProblem problem{
        .lower_bounds = space.normalized_lower_bounds(),
        .upper_bounds = space.normalized_upper_bounds(),
        .residual_count = residual_evaluator_.observations().size(),
        .residual_function =
            [&](const Vector& normalized) {
                Vector residuals = residual_evaluator_.evaluate(normalized);
                ++evaluations;
                if (observe)
                    observe(evaluations, space.to_physical(normalized), residuals);
                return residuals;
            },
    };
    result_ = optimizer_.solve(problem, space.to_normalized(request.initial_parameters),
                               request.optimization);
    return space.to_physical(result_.parameters);
}

const PoundersSolveResult& BasinHeatInverseSolver::optimization_result() const noexcept {
    return result_;
}

BasinHeatResidualEvaluator& BasinHeatInverseSolver::residual_evaluator() noexcept {
    return residual_evaluator_;
}
