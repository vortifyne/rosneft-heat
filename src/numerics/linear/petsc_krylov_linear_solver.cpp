#include "numerics/linear/petsc_krylov_linear_solver.hpp"

#include "numerics/linear/petsc_linear_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <petscksp.h>
#include <stdexcept>

using petsc_linear::check;
using petsc_linear::PetscHandle;

struct PetscKrylovLinearSolver::Impl {
    PetscHandle<Mat, MatDestroy> matrix;
    PetscHandle<Vec, VecDestroy> rhs;
    PetscHandle<Vec, VecDestroy> solution;
    PetscHandle<KSP, KSPDestroy> ksp;
    Vector candidate;
    PetscInt size = 0;
    bool pc_ready = false;
};

PetscKrylovLinearSolver::PetscKrylovLinearSolver(LinearMethod method,
                                                 PreconditionerType preconditioner)
    : LinearSolver(LinearSolverType::iterative), method_(method), preconditioner_(preconditioner) {
    if (method != LinearMethod::gmres && method != LinearMethod::bicgstab)
        throw std::invalid_argument("PETSc requires an iterative method");
}

PetscKrylovLinearSolver::~PetscKrylovLinearSolver() = default;

void PetscKrylovLinearSolver::set_preconditioner_reuse(bool reuse) noexcept {
    if (reuse != reuse_preconditioner_)
        reset();
    reuse_preconditioner_ = reuse;
}

void PetscKrylovLinearSolver::reset() noexcept {
    if (impl_) {
        PC pc = nullptr;
        KSPGetPC(impl_->ksp.get(), &pc);
        PCReset(pc);
        impl_->pc_ready = false;
    }
}

LinearSolveResult PetscKrylovLinearSolver::solve(const SparseMatrix& A, const Vector& b, Vector& x,
                                                 const LinearSolveRequest& request) {
    const auto transfer_start = Clock::now();
    LinearTimings timings;
    if (request.new_pattern &&
        (A.storage_order() != SparseStorageOrder::csr || !A.is_compressed() || A.rows() <= 0 ||
         A.rows() != A.cols() || A.rows() != b.size()))
        return {.status = LinearSolveStatus::invalid_input,
                .iterations = 0,
                .r_norm = std::numeric_limits<double>::infinity()};

    if (!impl_) {
        PetscBool initialized = PETSC_FALSE;
        check(PetscInitialized(&initialized));
        if (!initialized)
            throw std::logic_error("PetscKrylovLinearSolver requires an active PETSc session");
        impl_ = std::make_unique<Impl>();
        check(KSPCreate(PETSC_COMM_SELF, impl_->ksp.out()));
        ++timings.ksp_builds;
        const auto ksp = impl_->ksp.get();
        check(KSPSetType(ksp, method_ == LinearMethod::gmres ? KSPGMRES : KSPBCGS));
        if (method_ == LinearMethod::gmres) {
            check(KSPGMRESSetRestart(ksp, 100));
            check(KSPGMRESSetOrthogonalization(ksp, KSPGMRESModifiedGramSchmidtOrthogonalization));
        }
        check(KSPSetPCSide(ksp, PC_LEFT));
        check(KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED));
        check(KSPSetErrorIfNotConverged(ksp, PETSC_FALSE));
        PC pc = nullptr;
        check(KSPGetPC(ksp, &pc));
        switch (preconditioner_) {
        case PreconditionerType::none:
            check(PCSetType(pc, PCNONE));
            break;
        case PreconditionerType::ilu:
        case PreconditionerType::ilu2:
            check(PCSetType(pc, PCILU));
            check(PCFactorSetLevels(pc, preconditioner_ == PreconditionerType::ilu ? 0 : 2));
            break;
        case PreconditionerType::gamg:
            check(PCSetType(pc, PCGAMG));
            break;
        case PreconditionerType::boomeramg:
            check(PCSetType(pc, PCHYPRE));
            check(PCHYPRESetType(pc, "boomeramg"));
            break;
        }
    }
    auto& work = *impl_;
    const auto ksp = work.ksp.get();
    const auto& matrix = A.native_csr();
    if (request.new_pattern) {
        const auto size = static_cast<PetscInt>(A.rows());
        if (work.size != size) {
            check(KSPReset(ksp));
            work.rhs.reset();
            work.solution.reset();
            check(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, size, nullptr, work.rhs.out()));
            check(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, size, nullptr, work.solution.out()));
            work.size = size;
            work.pc_ready = false;
            timings.vector_builds += 2;
        }
        work.matrix.reset();
        petsc_linear::create_matrix(matrix, work.matrix.out());
        ++timings.matrix_builds;
    } else {
        // The caller guarantees identical CSR ordering throughout this nonlinear solve.
        PetscScalar* values = nullptr;
        check(MatSeqAIJGetArray(work.matrix.get(), &values));
        std::copy_n(matrix.valuePtr(), matrix.nonZeros(), values);
        check(MatSeqAIJRestoreArray(work.matrix.get(), &values));
    }
    check(KSPSetOperators(ksp, work.matrix.get(), work.matrix.get()));
    if (request.new_solve) {
        check(KSPSetTolerances(ksp, request.rtol, request.atol, PETSC_DEFAULT, request.maxiter));
        check(KSPSetFromOptions(ksp));
    }
    const bool reuse_pc = reuse_preconditioner_ && work.pc_ready;
    check(KSPSetReusePreconditioner(ksp, reuse_pc ? PETSC_TRUE : PETSC_FALSE));
    timings.transfer_s = elapsed_seconds(transfer_start);

    auto start = Clock::now();
    check(KSPSetUp(ksp));
    timings.setup_s = elapsed_seconds(start);
    timings.pc_updates = reuse_pc ? 0 : 1;
    work.pc_ready = true;

    work.candidate.resize(work.size);
    // PETSc borrows these arrays only until VecResetArray below.
    check(VecPlaceArray(work.rhs.get(), b.native().data()));
    check(VecPlaceArray(work.solution.get(), work.candidate.native().data()));
    check(VecSet(work.solution.get(), 0.0));
    start = Clock::now();
    const auto error = KSPSolve(ksp, work.rhs.get(), work.solution.get());
    timings.solve_s = elapsed_seconds(start);
    check(VecResetArray(work.solution.get()));
    check(VecResetArray(work.rhs.get()));
    check(error);

    KSPConvergedReason reason;
    PetscInt iterations = 0;
    PetscReal norm = 0.0;
    check(KSPGetConvergedReason(ksp, &reason));
    check(KSPGetIterationNumber(ksp, &iterations));
    check(KSPGetResidualNorm(ksp, &norm));
    auto status = LinearSolveStatus::converged;
    if (reason <= 0) {
        status = reason == KSP_DIVERGED_ITS         ? LinearSolveStatus::maxiter
                 : reason == KSP_DIVERGED_PC_FAILED ? LinearSolveStatus::factorization_failed
                                                    : LinearSolveStatus::solve_failed;
    } else if (!work.candidate.all_finite() || !std::isfinite(norm)) {
        status = LinearSolveStatus::nonfinite_solution;
    } else {
        x.native().swap(work.candidate.native());
    }
    return {.status = status,
            .iterations = static_cast<int>(iterations),
            .r_norm = norm,
            .timings = timings};
}
