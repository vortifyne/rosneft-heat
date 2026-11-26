#pragma once

#include "numerics/linear/linear_solver.hpp"

class UmfpackLinearSolver final : public LinearSolver {
public:
    UmfpackLinearSolver() noexcept : LinearSolver(LinearSolverType::direct) {}

    SparseStorageOrder required_storage_order() const noexcept override {
        return SparseStorageOrder::csc;
    }

    // Factor A once and replace each right-hand side with its solution.
    [[nodiscard]] static Eigen::MatrixXd solve_columns(const SparseMatrix& A, Eigen::MatrixXd rhs);

    LinearSolveResult solve(const SparseMatrix& A, const Vector& b, Vector& x,
                            const LinearSolveRequest& request) override;
};
