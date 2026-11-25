#pragma once

#include "numerics/algebra/sparse_matrix.hpp"

#include <petscmat.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace petsc_linear {

template <typename Handle, PetscErrorCode (*Destroy)(Handle*)> class PetscHandle {
public:
    ~PetscHandle() {
        reset();
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
    void reset() noexcept {
        if (value_ != nullptr)
            Destroy(&value_);
    }

private:
    Handle value_ = nullptr;
};

inline void check(PetscErrorCode error) {
    if (error != PETSC_SUCCESS)
        throw std::runtime_error("PETSc linear solver error " + std::to_string(error));
}

inline void create_matrix(const SparseMatrix::CsrNativeType& csr, Mat* matrix) {
    static_assert(std::is_same_v<PetscInt, SparseMatrix::StorageIndex>,
                  "PETSc and Eigen must use the same CSR index type");
    static_assert(std::is_same_v<PetscScalar, double>, "PETSc must use real double precision");
    const auto size = static_cast<PetscInt>(csr.rows());
    check(MatCreateSeqAIJ(PETSC_COMM_SELF, size, size, 0, nullptr, matrix));
    check(MatSeqAIJSetPreallocationCSR(*matrix, csr.outerIndexPtr(), csr.innerIndexPtr(),
                                       csr.valuePtr()));
}

} // namespace petsc_linear
