#pragma once

#include "numerics/linear/linear_solver.hpp"

class FGMRES final : public LinearSolver {
public:
    FGMRES() noexcept : LinearSolver(LinearSolverType::iterative) {}

    SparseStorageOrder required_storage_order() const noexcept override {
        return SparseStorageOrder::csr;
    }

    LinearSolveResult solve(const SparseMatrix& A, const Vector& b, Vector& x,
                            const LinearSolveRequest& request) override;

private:
    // apply_preconditioner
    // set_preconditioner
};
