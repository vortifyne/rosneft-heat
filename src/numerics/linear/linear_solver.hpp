#pragma once

#include "numerics/algebra/sparse_matrix.hpp"
#include "numerics/algebra/vector.hpp"

struct LinearSolveRequest {
    double rtol = 1.0e-8;
    double atol = 1.0e-50;
    int maxiter = 1000;
};

enum class LinearSolveStatus {
    converged,
    maxiter,
    invalid_input,
    factorization_failed,
    solve_failed,
    nonfinite_solution,
};

struct [[nodiscard]] LinearSolveResult {
    LinearSolveStatus status;
    int iterations;
    // Original residual for UMFPACK; left-preconditioned residual for PETSc Krylov solvers.
    double r_norm;
};

enum class LinearSolverType {
    direct,
    iterative,
};

enum class LinearMethod {
    lu,
    gmres,
    bicgstab
};
enum class Preconditioner {
    ilu,
    ilu2,
    gamg,
    boomeramg
};

struct LinearSolverOptions {
    LinearMethod method = LinearMethod::lu;
    Preconditioner preconditioner = Preconditioner::ilu;

    bool operator==(const LinearSolverOptions&) const = default;
};

class LinearSolver {
public:
    virtual ~LinearSolver() = default;

    LinearSolverType type() const noexcept {
        return type_;
    }

    virtual SparseStorageOrder required_storage_order() const noexcept = 0;

    // Affects iterative preconditioners; direct solvers ignore this setting.
    virtual void set_preconditioner_reuse(bool) noexcept {}

    // Discard cached numerical data without replacing the solver or its settings.
    virtual void reset() noexcept {}

    virtual LinearSolveResult solve(const SparseMatrix& A, const Vector& b, Vector& x,
                                    const LinearSolveRequest& request) = 0;

protected:
    explicit LinearSolver(LinearSolverType type) noexcept : type_(type) {}

private:
    LinearSolverType type_;
};
