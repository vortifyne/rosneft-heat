#pragma once

#include "linear/linear_solver.hpp"

#include <memory>

enum class PetscKrylovMethod {
    gmres,
    bicgstab
};
enum class PetscPreconditioner {
    ilu0,
    ilu2,
    gamg,
    boomeramg
};

// Sequential Krylov solver. Requires an active PETSc session owned by the application.
class PetscKrylovLinearSolver final : public LinearSolver {
public:
    explicit PetscKrylovLinearSolver(PetscKrylovMethod method = PetscKrylovMethod::gmres,
                                     PetscPreconditioner preconditioner = PetscPreconditioner::ilu0,
                                     bool reuse_preconditioner = false);
    ~PetscKrylovLinearSolver() override;

    SparseStorageOrder required_storage_order() const noexcept override {
        return SparseStorageOrder::csr;
    }

    // final_residual_norm is the PETSc preconditioned residual norm.
    LinearSolveResult solve(const SparseMatrix& A, const Vector& b, Vector& x,
                            const LinearSolveRequest& request) override;

private:
    PetscKrylovMethod method_;
    PetscPreconditioner preconditioner_;
    bool reuse_preconditioner_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
