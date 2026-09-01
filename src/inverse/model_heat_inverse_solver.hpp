#pragma once

#include "inverse/model_heat_residual_evaluator.hpp"
#include "optimization/pounders_optimizer.hpp"

struct ModelHeatInverseSolveRequest {
    ModelHeatParameters initial_parameters;
    PoundersSolveRequest optimization;
};

struct [[nodiscard]] ModelHeatInverseResult {
    PoundersSolveStatus status;

    ModelHeatParameters parameters;
    Vector normalized_parameters;

    Vector calculated_temperatures;
    Vector normalized_residuals;
    double objective;

    int optimization_iterations;
    int optimization_function_evaluations;
    int forward_evaluations;
    double forward_elapsed_time_seconds;

    [[nodiscard]] constexpr bool converged() const noexcept {
        return is_pounders_converged(status);
    }
};

class ModelHeatInverseSolver {
public:
    explicit ModelHeatInverseSolver(ModelHeatResidualEvaluator residual_evaluator);

    ModelHeatInverseResult solve(const ModelHeatInverseSolveRequest& request);

private:
    ModelHeatResidualEvaluator residual_evaluator_;
    PoundersOptimizer optimizer_;
};
