#pragma once

#include "basin/forward/basin_forward_solver.hpp"
#include "basin/inverse/basin_heat_observation_data.hpp"
#include "basin/inverse/basin_heat_parameter_space.hpp"

#include <array>

class BasinHeatResidualEvaluator {
public:
    BasinHeatResidualEvaluator(BasinForwardSolver forward, BasinHeatObservationData observations);

    [[nodiscard]] const BasinHeatParameterSpace& parameter_space() const noexcept;
    [[nodiscard]] const BasinHeatObservationData& observations() const noexcept;
    [[nodiscard]] BasinForwardSolver& forward_solver() noexcept;
    [[nodiscard]] Vector evaluate(const Vector& normalized);

private:
    struct Interpolation {
        std::array<std::size_t, 4> cells;
        std::array<double, 4> weights;
    };

    BasinForwardSolver forward_;
    BasinHeatObservationData observations_;
    BasinHeatParameterSpace parameter_space_;
    std::vector<Interpolation> interpolation_;
};
