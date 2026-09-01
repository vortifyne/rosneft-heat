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

bool is_pounders_converged(PoundersSolveStatus status) noexcept;

struct PoundersSolveResult {
    PoundersSolveStatus status;
    Vector parameters;
    Vector residuals;
    double objective;
    int iterations;
    int function_evaluations;

    bool converged() const noexcept;
};

class PoundersOptimizer {
public:
    PoundersSolveResult solve(const BoundedLeastSquaresProblem& problem,
                              const Vector& initial_parameters,
                              const PoundersSolveRequest& request) const;
};
