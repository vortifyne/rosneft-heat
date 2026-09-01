#pragma once

#include "optimization/bounded_least_squares_problem.hpp"

struct PoundersSolveRequest {
    double gradient_absolute_tolerance;
    double gradient_relative_tolerance;
    double gradient_reduction_tolerance;
    int max_iterations;
    int max_function_evaluations;
};

enum class PoundersSolveStatus {
    converged_gradient_absolute,
    converged_gradient_relative,
    converged_gradient_reduction,
    converged_step,
    converged_objective,
    converged_user,
    maximum_iterations,
    maximum_function_evaluations,
    nonfinite_objective,
    line_search_failed,
    trust_region_failed,
    diverged_user,
    unknown,
};

[[nodiscard]] constexpr bool is_pounders_converged(PoundersSolveStatus status) noexcept {
    switch (status) {
    case PoundersSolveStatus::converged_gradient_absolute:
    case PoundersSolveStatus::converged_gradient_relative:
    case PoundersSolveStatus::converged_gradient_reduction:
    case PoundersSolveStatus::converged_step:
    case PoundersSolveStatus::converged_objective:
    case PoundersSolveStatus::converged_user:
        return true;
    default:
        return false;
    }
}

struct [[nodiscard]] PoundersSolveResult {
    PoundersSolveStatus status;
    Vector parameters;
    Vector residuals;
    double objective;
    int iterations;
    int function_evaluations;

    [[nodiscard]] constexpr bool converged() const noexcept {
        return is_pounders_converged(status);
    }
};

class PoundersOptimizer {
public:
    PoundersSolveResult solve(const BoundedLeastSquaresProblem& problem,
                              const Vector& initial_parameters,
                              const PoundersSolveRequest& request) const;
};
