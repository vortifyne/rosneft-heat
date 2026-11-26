#include "numerics/linear/petsc_preconditioner.hpp"

#include "numerics/linear/petsc_linear_utils.hpp"

#include <petscpc.h>

using petsc_linear::check;
using petsc_linear::PetscHandle;

struct PetscPreconditioner::Impl {
    PetscHandle<PC, PCDestroy> pc;
    PetscHandle<Vec, VecDestroy> input;
    PetscHandle<Vec, VecDestroy> output;
    PetscInt size = 0;
};

PetscPreconditioner::PetscPreconditioner(int levels) : levels_(levels) {}
PetscPreconditioner::~PetscPreconditioner() = default;

void PetscPreconditioner::setup(const SparseMatrix& A) {
    if (A.storage_order() != SparseStorageOrder::csr || !A.is_compressed() ||
        A.rows() != A.cols() || A.rows() == 0)
        throw std::runtime_error("PETSc preconditioner requires a square compressed CSR matrix");

    PetscBool initialized = PETSC_FALSE;
    check(PetscInitialized(&initialized));
    if (!initialized)
        throw std::logic_error("PetscPreconditioner requires an active PETSc session");

    auto work = std::make_unique<Impl>();
    PetscHandle<Mat, MatDestroy> matrix;
    petsc_linear::create_matrix(A.native_csr(), matrix.out());
    check(PCCreate(PETSC_COMM_SELF, work->pc.out()));
    const auto pc = work->pc.get();
    check(PCSetOperators(pc, matrix.get(), matrix.get()));
    check(PCSetType(pc, PCILU));
    check(PCFactorSetLevels(pc, levels_));
    check(PCSetFromOptions(pc));
    check(PCSetErrorIfFailure(pc, PETSC_TRUE));
    check(PCSetUp(pc));

    work->size = static_cast<PetscInt>(A.rows());
    check(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, work->size, nullptr, work->input.out()));
    check(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, work->size, nullptr, work->output.out()));
    impl_ = std::move(work);
}

void PetscPreconditioner::apply(const SparseMatrix&, Eigen::Ref<const Eigen::VectorXd> v,
                                Eigen::Ref<Eigen::VectorXd> z) {
    if (!impl_ || v.size() != impl_->size || z.size() != impl_->size)
        throw std::runtime_error("PETSc preconditioner and vector sizes must match");

    const auto input = impl_->input.get();
    const auto output = impl_->output.get();
    check(VecPlaceArray(input, v.data()));
    const auto placed = VecPlaceArray(output, z.data());
    if (placed != PETSC_SUCCESS) {
        check(VecResetArray(input));
        check(placed);
    }
    const auto applied = PCApply(impl_->pc.get(), input, output);
    const auto reset_output = VecResetArray(output);
    const auto reset_input = VecResetArray(input);
    check(applied);
    check(reset_output);
    check(reset_input);
}
