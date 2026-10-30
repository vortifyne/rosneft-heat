#pragma once

#include "numerics/nonlinear/nonlinear_method.hpp"

class SparseMatrix;
class Vector;

class NonlinearSystem {
public:
    virtual ~NonlinearSystem() = default;

    virtual void assemble_residual(const Vector& x, Vector& r) const = 0;

    void assemble_matrix(NonlinearMethod method, const Vector& x, SparseMatrix& A) const {
        switch (method) {
        case NonlinearMethod::picard:
            assemble_picard_matrix(x, A);
            break;

        case NonlinearMethod::newton:
            assemble_newton_matrix(x, A);
            break;
        }
    }

private:
    virtual void assemble_picard_matrix(const Vector& x, SparseMatrix& A) const = 0;

    virtual void assemble_newton_matrix(const Vector& x, SparseMatrix& A) const = 0;
};
