#pragma once

#include "basin/inverse/basin_heat_residual_evaluator.hpp"
#include "numerics/optimization/pounders_optimizer.hpp"

#include <functional>

struct BasinHeatInverseSolveRequest {
    Vector initial_parameters; // Basal flux [W/m²], oldest to youngest.
    PoundersSolveRequest optimization{.gatol = 1e-6,
                                      .grtol = 1e-6,
                                      .gttol = 1e-6,
                                      .maxiter = 10,
                                      .maxfev = 100,
                                      .residual_tolerance = 1.0};
};

// Optimizer iteration, forward evaluation, physical parameters, normalized residuals.
using BasinInverseObserver = std::function<void(int, int, const Vector&, const Vector&)>;

class BasinHeatInverseSolver {
public:
    explicit BasinHeatInverseSolver(BasinHeatResidualEvaluator residual_evaluator);

    [[nodiscard]] Vector solve(const BasinHeatInverseSolveRequest& request,
                               const BasinInverseObserver& observe = {});
    [[nodiscard]] const PoundersSolveResult& optimization_result() const noexcept;
    [[nodiscard]] BasinHeatResidualEvaluator& residual_evaluator() noexcept;

private:
    BasinHeatResidualEvaluator residual_evaluator_;
    PoundersOptimizer optimizer_;
    PoundersSolveResult result_{};
};
