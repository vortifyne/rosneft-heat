#pragma once

#include "basin/inverse/basin_objective.hpp"
#include "numerics/optimization/bounded_least_squares_problem.hpp"

#include <functional>

enum class BasinInverseMethod {
    pounders,
    gauss_newton
};

struct BasinHeatInverseSolveRequest {
    Vector initial_parameters; // Basal flux [W/m²], oldest to youngest.
    LeastSquaresSolveRequest optimization{.gatol = 1e-6,
                                          .grtol = 1e-6,
                                          .gttol = 1e-6,
                                          .maxiter = 10,
                                          .maxfev = 100,
                                          .residual_tolerance = 1.0};
    BasinInverseMethod method = BasinInverseMethod::pounders;
};

// Optimizer iteration, forward evaluation, physical parameters, normalized residuals.
using BasinInverseObserver = std::function<void(int, int, const Vector&, const Vector&)>;

class BasinHeatInverseSolver {
public:
    explicit BasinHeatInverseSolver(BasinObjective objective);

    [[nodiscard]] Vector solve(const BasinHeatInverseSolveRequest& request,
                               const BasinInverseObserver& observe = {});
    [[nodiscard]] const LeastSquaresSolveResult& optimization_result() const noexcept;
    [[nodiscard]] BasinObjective& objective() noexcept;

private:
    BasinObjective objective_;
    LeastSquaresSolveResult result_{};
};
