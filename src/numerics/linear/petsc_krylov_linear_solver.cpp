#include "numerics/linear/petsc_krylov_linear_solver.hpp"

#include <cmath>
#include <limits>
#include <petscksp.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename Handle, PetscErrorCode (*Destroy)(Handle*)> class PetscHandle {
public:
    ~PetscHandle() {
        if (value_ != nullptr)
            Destroy(&value_);
    }
    PetscHandle() = default;
    PetscHandle(const PetscHandle&) = delete;
    PetscHandle& operator=(const PetscHandle&) = delete;
    Handle get() const noexcept {
        return value_;
    }
    Handle* out() noexcept {
        return &value_;
    }

private:
    Handle value_ = nullptr;
};

void check(PetscErrorCode error) {
    if (error != PETSC_SUCCESS) {
        throw std::runtime_error("PETSc linear solver error " + std::to_string(error));
    }
}

LinearSolveResult failure(LinearSolveStatus status, int iterations = 0,
                          double norm = std::numeric_limits<double>::infinity()) {
    return {.status = status, .iterations = iterations, .final_residual_norm = norm};
}

} // namespace

struct PetscKrylovLinearSolver::Impl {
    PetscHandle<Mat, MatDestroy> frozen_matrix;
    PetscHandle<KSP, KSPDestroy> ksp;
    PetscInt size = 0;
};

PetscKrylovLinearSolver::PetscKrylovLinearSolver(LinearMethod method, Preconditioner preconditioner)
    : LinearSolver(LinearSolverType::iterative), method_(method), preconditioner_(preconditioner) {
    if (method != LinearMethod::gmres && method != LinearMethod::bicgstab) {
        throw std::invalid_argument("PETSc requires an iterative method");
    }
}

PetscKrylovLinearSolver::~PetscKrylovLinearSolver() = default;

void PetscKrylovLinearSolver::set_preconditioner_reuse(bool reuse) noexcept {
    if (reuse != reuse_preconditioner_)
        reset();
    reuse_preconditioner_ = reuse;
}

void PetscKrylovLinearSolver::reset() noexcept {
    impl_.reset();
}

LinearSolveResult PetscKrylovLinearSolver::solve(const SparseMatrix& A, const Vector& b, Vector& x,
                                                 const LinearSolveRequest& request) {
    PetscBool initialized = PETSC_FALSE;
    PetscBool finalized = PETSC_FALSE;
    check(PetscInitialized(&initialized));
    check(PetscFinalized(&finalized));
    if (!initialized || finalized) {
        throw std::logic_error("PetscKrylovLinearSolver requires an active PETSc session");
    }
    if (A.storage_order() != SparseStorageOrder::csr || !A.is_compressed() || A.rows() <= 0 ||
        A.rows() != A.cols() || A.rows() != b.size() || !b.all_finite() ||
        A.rows() > std::numeric_limits<PetscInt>::max() ||
        !std::isfinite(request.relative_tolerance) || request.relative_tolerance < 0.0 ||
        request.relative_tolerance >= 1.0 || !std::isfinite(request.absolute_tolerance) ||
        request.absolute_tolerance < 0.0 || request.max_iterations <= 0) {
        return failure(LinearSolveStatus::invalid_input);
    }
    const auto& matrix = A.native_csr();
    for (SparseMatrix::Index i = 0; i < matrix.nonZeros(); ++i) {
        if (!std::isfinite(matrix.valuePtr()[i]))
            return failure(LinearSolveStatus::invalid_input);
    }
    const auto size = static_cast<PetscInt>(A.rows());
    std::vector<PetscInt> offsets(static_cast<std::size_t>(size) + 1);
    std::vector<PetscInt> columns(static_cast<std::size_t>(matrix.nonZeros()));
    std::vector<PetscScalar> values(static_cast<std::size_t>(matrix.nonZeros()));
    for (PetscInt i = 0; i <= size; ++i)
        offsets[static_cast<std::size_t>(i)] = matrix.outerIndexPtr()[i];
    for (SparseMatrix::Index i = 0; i < matrix.nonZeros(); ++i) {
        columns[static_cast<std::size_t>(i)] = matrix.innerIndexPtr()[i];
        values[static_cast<std::size_t>(i)] = matrix.valuePtr()[i];
    }
    PetscHandle<Mat, MatDestroy> petsc_matrix;
    PetscHandle<Vec, VecDestroy> rhs;
    PetscHandle<Vec, VecDestroy> solution;
    PetscHandle<KSP, KSPDestroy> local_ksp;
    check(MatCreateSeqAIJ(PETSC_COMM_SELF, size, size, 0, nullptr, petsc_matrix.out()));
    check(MatSeqAIJSetPreallocationCSR(petsc_matrix.get(), offsets.data(), columns.data(),
                                       values.data()));
    check(MatCreateVecs(petsc_matrix.get(), solution.out(), rhs.out()));
    PetscScalar* rhs_values = nullptr;
    check(VecGetArray(rhs.get(), &rhs_values));
    for (PetscInt i = 0; i < size; ++i)
        rhs_values[i] = b[i];
    check(VecRestoreArray(rhs.get(), &rhs_values));
    check(VecSet(solution.get(), 0.0));
    const bool reused = reuse_preconditioner_ && impl_ && impl_->size == size;
    if (reuse_preconditioner_ && !reused) {
        impl_ = std::make_unique<Impl>();
        impl_->size = size;
        check(MatDuplicate(petsc_matrix.get(), MAT_COPY_VALUES, impl_->frozen_matrix.out()));
        check(KSPCreate(PETSC_COMM_SELF, impl_->ksp.out()));
    } else if (!reuse_preconditioner_) {
        check(KSPCreate(PETSC_COMM_SELF, local_ksp.out()));
    }
    KSP ksp = reuse_preconditioner_ ? impl_->ksp.get() : local_ksp.get();
    // A is current; P defines the preconditioner retained until an explicit reset.
    check(KSPSetOperators(ksp, petsc_matrix.get(),
                          reuse_preconditioner_ ? impl_->frozen_matrix.get() : petsc_matrix.get()));
    check(KSPSetReusePreconditioner(ksp, reused ? PETSC_TRUE : PETSC_FALSE));
    if (!reused) {
        check(KSPSetType(ksp, method_ == LinearMethod::gmres ? KSPGMRES : KSPBCGS));
        if (method_ == LinearMethod::gmres) {
            check(KSPGMRESSetRestart(ksp, 30));
        }
        check(KSPSetPCSide(ksp, PC_LEFT));
        check(KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED));
        PC pc = nullptr;
        check(KSPGetPC(ksp, &pc));
        switch (preconditioner_) {
        case Preconditioner::ilu:
        case Preconditioner::ilu2:
            check(PCSetType(pc, PCILU));
            check(PCFactorSetLevels(pc, preconditioner_ == Preconditioner::ilu ? 0 : 2));
            break;
        case Preconditioner::gamg:
            check(PCSetType(pc, PCGAMG));
            break;
        case Preconditioner::boomeramg:
            check(PCSetType(pc, PCHYPRE));
            check(PCHYPRESetType(pc, "boomeramg"));
            break;
        }
    }
    check(KSPSetTolerances(ksp, request.relative_tolerance, request.absolute_tolerance,
                           PETSC_DEFAULT, request.max_iterations));
    check(KSPSetErrorIfNotConverged(ksp, PETSC_FALSE));
    check(KSPSolve(ksp, rhs.get(), solution.get()));
    KSPConvergedReason reason;
    PetscInt iterations = 0;
    PetscReal norm = 0.0;
    check(KSPGetConvergedReason(ksp, &reason));
    check(KSPGetIterationNumber(ksp, &iterations));
    check(KSPGetResidualNorm(ksp, &norm));
    if (reason <= 0) {
        const auto status = reason == KSP_DIVERGED_ITS ? LinearSolveStatus::max_iterations
                            : reason == KSP_DIVERGED_PC_FAILED
                                ? LinearSolveStatus::factorization_failed
                                : LinearSolveStatus::solve_failed;
        return failure(status, static_cast<int>(iterations), norm);
    }
    Vector candidate(size);
    const PetscScalar* solution_values = nullptr;
    check(VecGetArrayRead(solution.get(), &solution_values));
    for (PetscInt i = 0; i < size; ++i)
        candidate[i] = PetscRealPart(solution_values[i]);
    check(VecRestoreArrayRead(solution.get(), &solution_values));
    if (!candidate.all_finite() || !std::isfinite(norm)) {
        return failure(LinearSolveStatus::nonfinite_solution, static_cast<int>(iterations), norm);
    }
    x = std::move(candidate);
    return {.status = LinearSolveStatus::converged,
            .iterations = static_cast<int>(iterations),
            .final_residual_norm = norm};
}
