#pragma once

#include "nonlinear/nonlinear_method.hpp"

class SparseMatrix;
class Vector;

class NonlinearSystem {
public:
    virtual ~NonlinearSystem() = default;

    virtual void assemble_residual(const Vector& x, Vector& F) const = 0;

    void assemble_matrix(NonlinearMethod method, const Vector& x, SparseMatrix& mat) const {
        switch (method) {
        case NonlinearMethod::picard:
            assemble_picard_matrix(x, mat);
            break;

        case NonlinearMethod::newton:
            assemble_newton_matrix(x, mat);
            break;
        }
    }

private:
    virtual void assemble_picard_matrix(const Vector& x, SparseMatrix& mat) const = 0;

    virtual void assemble_newton_matrix(const Vector& x, SparseMatrix& mat) const = 0;
};
