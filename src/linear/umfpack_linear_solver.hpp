#pragma once

#include "linear/linear_solver.hpp"

class UmfpackLinearSolver final : public LinearSolver {
public:
    UmfpackLinearSolver() noexcept : LinearSolver(LinearSolverType::direct) {}

    SparseStorageOrder required_storage_order() const noexcept override {
        return SparseStorageOrder::csc;
    }

    LinearSolveResult solve(const SparseMatrix& A, const Vector& b, Vector& x,
                            const LinearSolveRequest& request) override;
};
