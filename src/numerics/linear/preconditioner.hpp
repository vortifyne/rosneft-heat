#pragma once

#include "numerics/algebra/sparse_matrix.hpp"

class Preconditioner {
public:
    virtual ~Preconditioner() = default;

    virtual void setup(const SparseMatrix& A) = 0;
    virtual void apply(const SparseMatrix& A, Eigen::Ref<const Eigen::VectorXd> v,
                       Eigen::Ref<Eigen::VectorXd> z) = 0;
};
