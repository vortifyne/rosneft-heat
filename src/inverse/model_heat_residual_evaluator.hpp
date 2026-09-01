#pragma once

#include "forward/model_heat_forward_solver.hpp"
#include "inverse/model_heat_observation_data.hpp"
#include "inverse/model_heat_parameter_space.hpp"

struct [[nodiscard]] ModelHeatResidualResult {
    TimeIntegrationResult time_integration;
    Vector calculated_temperatures;
    Vector normalized_residuals;
    double objective;
    double forward_elapsed_time_seconds;

    [[nodiscard]] constexpr bool completed() const noexcept {
        return time_integration.completed();
    }
};

class ModelHeatResidualEvaluator {
public:
    ModelHeatResidualEvaluator(ModelHeatForwardSolver forward_solver,
                               ModelHeatParameterSpace parameter_space,
                               ModelHeatObservationData observations);

    Vector::Index parameter_count() const noexcept;
    Vector::Index residual_count() const noexcept;

    const ModelHeatParameterSpace& parameter_space() const noexcept;
    const ModelHeatObservationData& observations() const noexcept;

    ModelHeatResidualResult evaluate(const Vector& normalized_parameters);

private:
    ModelHeatForwardSolver forward_solver_;
    ModelHeatParameterSpace parameter_space_;
    ModelHeatObservationData observations_;
};
