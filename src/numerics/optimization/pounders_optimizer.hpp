#pragma once

#include "numerics/optimization/bounded_least_squares_problem.hpp"

class PoundersOptimizer {
public:
    LeastSquaresSolveResult solve(const BoundedLeastSquaresProblem& problem, const Vector& x0,
                                  const LeastSquaresSolveRequest& request,
                                  const LeastSquaresObserver& observe = {}) const;
};
