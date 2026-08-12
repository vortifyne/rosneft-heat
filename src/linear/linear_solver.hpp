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
    invalid_input,
    factorization_failed,
    solve_failed,
    nonfinite_solution,
};

struct LinearSolveResult {
    LinearSolveStatus status;
    int iterations;
    double final_residual_norm;
};

enum class LinearSolverType {
    direct,
    iterative,
};

enum class LinearSolverKind {
    umfpack_lu,
};

class LinearSolver {
public:
    virtual ~LinearSolver() = default;

    LinearSolverType type() const noexcept {
        return type_;
    }

    virtual SparseStorageOrder required_storage_order() const noexcept = 0;

    virtual LinearSolveResult solve(const SparseMatrix& A, const Vector& b, Vector& x,
                                    const LinearSolveRequest& request) = 0;

protected:
    explicit LinearSolver(LinearSolverType type) noexcept : type_(type) {}

private:
    LinearSolverType type_;
};
