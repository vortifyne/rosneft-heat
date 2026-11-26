#pragma once

#include "numerics/optimization/bounded_least_squares_problem.hpp"

#include <Eigen/LU>

class GaussNewtonOptimizer {
public:
    LeastSquaresSolveResult solve(const BoundedLeastSquaresProblem& problem, const Vector& initial,
                                  const LeastSquaresSolveRequest& request,
                                  const LeastSquaresObserver& observe = {});

private:
    Eigen::FullPivLU<Eigen::MatrixXd> linear_solver_;
};
