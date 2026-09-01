#include "inverse/model_heat_inverse_solver.hpp"

#include <stdexcept>
#include <utility>

namespace {

struct ForwardEvaluationStatistics {
    int evaluations = 0;
    double elapsed_time_seconds = 0.0;
};

void record_evaluation(ForwardEvaluationStatistics& statistics,
                       const ModelHeatResidualResult& evaluation) {
    ++statistics.evaluations;
    statistics.elapsed_time_seconds += evaluation.forward_elapsed_time_seconds;
}

void require_completed(const ModelHeatResidualResult& evaluation) {
    if (!evaluation.completed()) {
        throw std::runtime_error("Forward solve failed during model heat inverse optimization");
    }
}

} // namespace

ModelHeatInverseSolver::ModelHeatInverseSolver(ModelHeatResidualEvaluator residual_evaluator)
    : residual_evaluator_(std::move(residual_evaluator)) {}

ModelHeatInverseResult ModelHeatInverseSolver::solve(const ModelHeatInverseSolveRequest& request) {
    const ModelHeatParameterSpace& parameter_space = residual_evaluator_.parameter_space();
    const Vector initial_parameters = parameter_space.to_normalized(request.initial_parameters);
    ForwardEvaluationStatistics forward_statistics;

    const BoundedLeastSquaresProblem problem{
        .lower_bounds = parameter_space.normalized_lower_bounds(),
        .upper_bounds = parameter_space.normalized_upper_bounds(),
        .residual_count = residual_evaluator_.residual_count(),
        .residual_function =
            [this, &forward_statistics](const Vector& parameters) {
                ModelHeatResidualResult evaluation = residual_evaluator_.evaluate(parameters);
                record_evaluation(forward_statistics, evaluation);
                require_completed(evaluation);
                return std::move(evaluation.normalized_residuals);
            },
    };

    PoundersSolveResult optimization =
        optimizer_.solve(problem, initial_parameters, request.optimization);

    ModelHeatResidualResult final_evaluation =
        residual_evaluator_.evaluate(optimization.parameters);
    record_evaluation(forward_statistics, final_evaluation);
    require_completed(final_evaluation);

    ModelHeatParameters physical_parameters = parameter_space.to_physical(optimization.parameters);
    return {
        .status = optimization.status,
        .parameters = physical_parameters,
        .normalized_parameters = std::move(optimization.parameters),
        .calculated_temperatures = std::move(final_evaluation.calculated_temperatures),
        .normalized_residuals = std::move(final_evaluation.normalized_residuals),
        .objective = final_evaluation.objective,
        .optimization_iterations = optimization.iterations,
        .optimization_function_evaluations = optimization.function_evaluations,
        .forward_evaluations = forward_statistics.evaluations,
        .forward_elapsed_time_seconds = forward_statistics.elapsed_time_seconds,
    };
}
