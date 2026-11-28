#pragma once

#include "numerics/linear/linear_solver.hpp"

class TdmaLinearSolver final : public LinearSolver {
public:
    TdmaLinearSolver() : LinearSolver(LinearSolverType::direct) {}
    MatrixStorageFormat required_storage_order() const noexcept override {
        return MatrixStorageFormat::tridiagonal;
    }
    LinearSolveResult solve(const SparseMatrix& A, const Vector& b, Vector& x,
                            const LinearSolveRequest&) override;
    static Eigen::MatrixXd solve_columns(const SparseMatrix& A, Eigen::MatrixXd rhs);
};
