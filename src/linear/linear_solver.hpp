#pragma once

#include "linear/sparse_matrix.hpp"
#include "linear/vector.hpp"

struct LinearSolveRequest {
    double relative_tolerance = 1.0e-8;
    double absolute_tolerance = 1.0e-50;
    int max_iterations = 1000;
};

enum class LinearSolveStatus {
    converged,
    max_iterations,
    invalid_input,
    factorization_failed,
    solve_failed,
    nonfinite_solution,
};

struct [[nodiscard]] LinearSolveResult {
    LinearSolveStatus status;
    int iterations;
    // Original residual for UMFPACK; left-preconditioned residual for PETSc Krylov solvers.
    double final_residual_norm;
};

enum class LinearSolverType {
    direct,
    iterative,
};

enum class LinearSolverKind {
    umfpack_lu,
    petsc_gmres_ilu,
    petsc_gmres_ilu2,
    petsc_gmres_gamg,
    petsc_gmres_boomeramg,
    petsc_bicgstab_ilu,
    petsc_bicgstab_ilu2,
    petsc_bicgstab_gamg,
    petsc_bicgstab_boomeramg,
    petsc_gmres_gamg_reuse,
    petsc_gmres_boomeramg_reuse,
    petsc_bicgstab_gamg_reuse,
    petsc_bicgstab_boomeramg_reuse,
};

class LinearSolver {
public:
    virtual ~LinearSolver() = default;

    LinearSolverType type() const noexcept {
        return type_;
    }

    virtual SparseStorageOrder required_storage_order() const noexcept = 0;

    // Discard cached numerical data without replacing the solver or its settings.
    virtual void reset() noexcept {}

    virtual LinearSolveResult solve(const SparseMatrix& A, const Vector& b, Vector& x,
                                    const LinearSolveRequest& request) = 0;

protected:
    explicit LinearSolver(LinearSolverType type) noexcept : type_(type) {}

private:
    LinearSolverType type_;
};
