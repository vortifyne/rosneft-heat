#pragma once

#include "linear/linear_solver.hpp"

class UmfpackLinearSolver final : public LinearSolver {
public:
    UmfpackLinearSolver() noexcept : LinearSolver(LinearSolverType::direct) {}

    LinearSolveResult solve(const SparseMatrix& A, const Vector& b, Vector& x,
                            const LinearSolveRequest& request) override;
};
