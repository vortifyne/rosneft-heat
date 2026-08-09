#pragma once

#include "linear/sparse_matrix.hpp"
#include "linear/vector.hpp"

struct LinearSolveRequest {
    double relative_tolerance;
    double absolute_tolerance;
    int max_iterations;
};

enum class LinearSolveStatus {
    converged,
    max_iterations,
    failed,
};

struct LinearSolveResult {
    LinearSolveStatus status;
    int iterations;
    double final_residual_norm;
};

class LinearSolver {
public:
    virtual ~LinearSolver() = default;

    virtual LinearSolveResult solve(const SparseMatrix& A, const Vector& b, Vector& x,
                                    const LinearSolveRequest& request) = 0;
};
