#pragma once

#include "numerics/algebra/sparse_matrix.hpp"
#include "numerics/algebra/vector.hpp"
#include "numerics/nonlinear/nonlinear_method.hpp"

class SemiDiscreteSystem {
public:
    virtual ~SemiDiscreteSystem() = default;

    virtual void assemble_residual(double time, const Vector& solution,
                                   const Vector& solution_derivative, Vector& residual) const = 0;

    virtual void assemble_matrix(NonlinearMethod method, double time, const Vector& solution,
                                 const Vector& solution_derivative, double derivative_shift,
                                 SparseMatrix& matrix) const = 0;
};
