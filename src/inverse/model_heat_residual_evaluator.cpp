#include "inverse/model_heat_residual_evaluator.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

ModelHeatResidualEvaluator::ModelHeatResidualEvaluator(ModelHeatForwardSolver forward_solver,
                                                       ModelHeatParameterSpace parameter_space,
                                                       ModelHeatObservationData observations)
    : forward_solver_(std::move(forward_solver)), parameter_space_(std::move(parameter_space)),
      observations_(std::move(observations)) {
    if (forward_solver_.observation_count() != static_cast<std::size_t>(observations_.size())) {
        throw std::invalid_argument(
            "Forward solver observation count must match the observed temperature count");
    }
}

Vector::Index ModelHeatResidualEvaluator::parameter_count() const noexcept {
    return ModelHeatParameterSpace::kParameterCount;
}

Vector::Index ModelHeatResidualEvaluator::residual_count() const noexcept {
    return observations_.size();
}

const ModelHeatParameterSpace& ModelHeatResidualEvaluator::parameter_space() const noexcept {
    return parameter_space_;
}

const ModelHeatObservationData& ModelHeatResidualEvaluator::observations() const noexcept {
    return observations_;
}

ModelHeatResidualResult ModelHeatResidualEvaluator::evaluate(const Vector& normalized_parameters) {
    ModelHeatForwardResult forward_result =
        forward_solver_.solve(parameter_space_.to_physical(normalized_parameters));

    if (!forward_result.completed()) {
        return {
            .time_integration = forward_result.integration,
            .calculated_temperatures = std::move(forward_result.calculated_temperature),
            .normalized_residuals = {},
            .objective = std::numeric_limits<double>::infinity(),
            .forward_elapsed_time_seconds = forward_result.elapsed_time_seconds,
        };
    }

    Vector residuals = observations_.normalized_residuals(forward_result.calculated_temperature);
    const double objective = residuals.squared_norm();
    return {
        .time_integration = forward_result.integration,
        .calculated_temperatures = std::move(forward_result.calculated_temperature),
        .normalized_residuals = std::move(residuals),
        .objective = objective,
        .forward_elapsed_time_seconds = forward_result.elapsed_time_seconds,
    };
}
