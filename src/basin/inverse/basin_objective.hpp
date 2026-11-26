#pragma once

#include "basin/forward/basin_forward_solver.hpp"
#include "basin/inverse/basin_heat_observation_data.hpp"
#include "numerics/optimization/bounded_least_squares_problem.hpp"

class BasinObjective {
public:
    BasinObjective(BasinForwardSolver forward, BasinHeatObservationData observations);

    [[nodiscard]] const BasinHeatParameterSpace& parameter_space() const noexcept;
    [[nodiscard]] const BasinHeatObservationData& observations() const noexcept;
    [[nodiscard]] BasinForwardSolver& forward_solver() noexcept;
    [[nodiscard]] Vector residuals_in_error_units(Vector weighted) const;
    [[nodiscard]] LeastSquaresEvaluation evaluate(const Vector& normalized, double tolerance,
                                                  Eigen::MatrixXd* jacobian = nullptr);

private:
    BasinForwardSolver forward_;
    BasinHeatObservationData observations_;
    std::vector<BasinWellSample> interpolation_;
};
