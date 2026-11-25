#pragma once

#include "numerics/linear/preconditioner.hpp"

#include <memory>

class PetscPreconditioner final : public Preconditioner {
public:
    explicit PetscPreconditioner(int levels = 0);
    ~PetscPreconditioner() override;

    void setup(const SparseMatrix& A) override;
    void apply(const SparseMatrix& A, Eigen::Ref<const Eigen::VectorXd> v,
               Eigen::Ref<Eigen::VectorXd> z) override;

private:
    int levels_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
