#pragma once

#include "numerics/algebra/sparse_matrix.hpp"
#include "numerics/algebra/vector.hpp"
#include "numerics/nonlinear/nonlinear_method.hpp"

class SemiDiscreteSystem {
public:
    virtual ~SemiDiscreteSystem() = default;

    virtual void assemble_residual(double t, const Vector& u, const Vector& du_dt,
                                   Vector& r) const = 0;

    // d(du_dt)/du = alpha * I.
    virtual void assemble_matrix(NonlinearMethod method, double t, const Vector& u,
                                 const Vector& du_dt, double alpha, SparseMatrix& A) const = 0;
};
