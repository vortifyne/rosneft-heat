#pragma once

#include "numerics/optimization/bounded_least_squares_problem.hpp"

struct PoundersSolveRequest {
    double gatol;
    double grtol;
    double gttol;
    int maxiter;
    int maxfev;
};

enum class PoundersStatus {
    gatol,
    grtol,
    gttol,
    step_tol,
    min_objective,
    user_converged,
    maxiter,
    maxfev,
    nonfinite,
    ls_failed,
    tr_failed,
    user_diverged,
    unknown,
};

[[nodiscard]] constexpr bool is_pounders_converged(PoundersStatus status) noexcept {
    switch (status) {
    case PoundersStatus::gatol:
    case PoundersStatus::grtol:
    case PoundersStatus::gttol:
    case PoundersStatus::step_tol:
    case PoundersStatus::min_objective:
    case PoundersStatus::user_converged:
        return true;
    default:
        return false;
    }
}

struct [[nodiscard]] PoundersSolveResult {
    PoundersStatus status;
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
    PoundersSolveResult solve(const BoundedLeastSquaresProblem& problem, const Vector& x0,
                              const PoundersSolveRequest& request) const;
};
