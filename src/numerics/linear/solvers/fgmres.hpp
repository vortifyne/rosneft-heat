#pragma once

#include "numerics/linear/pc/preconditioner.hpp"
#include "numerics/linear/solvers/linear_solver.hpp"

#include <memory>
#include <utility>

class FGMRES final : public LinearSolver {
public:
    explicit FGMRES(std::unique_ptr<Preconditioner> pc = nullptr) noexcept
        : LinearSolver(LinearSolverType::iterative), pc_(std::move(pc)) {}

    SparseStorageOrder required_storage_order() const noexcept override {
        return SparseStorageOrder::csr;
    }

    LinearSolveResult solve(const SparseMatrix& A, const Vector& b, Vector& x,
                            const LinearSolveRequest& request) override;

private:
    std::unique_ptr<Preconditioner> pc_;
};
