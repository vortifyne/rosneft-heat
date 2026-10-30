#pragma once

#include "numerics/linear/linear_solver.hpp"

#include <memory>

// Sequential Krylov solver. Requires an active PETSc session owned by the application.
class PetscKrylovLinearSolver final : public LinearSolver {
public:
    explicit PetscKrylovLinearSolver(LinearMethod method = LinearMethod::gmres,
                                     Preconditioner preconditioner = Preconditioner::ilu);
    ~PetscKrylovLinearSolver() override;

    void set_preconditioner_reuse(bool reuse) noexcept override;
    void reset() noexcept override;

    SparseStorageOrder required_storage_order() const noexcept override {
        return SparseStorageOrder::csr;
    }

    // final_residual_norm is the PETSc preconditioned residual norm.
    LinearSolveResult solve(const SparseMatrix& A, const Vector& b, Vector& x,
                            const LinearSolveRequest& request) override;

private:
    LinearMethod method_;
    Preconditioner preconditioner_;
    bool reuse_preconditioner_ = false;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
