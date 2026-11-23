#pragma once

#include "basin/mesh/col_grid.hpp"
#include "numerics/linear/pc/preconditioner.hpp"

#include <tdma/tdma.hpp>

class TwoGridPC final : public Preconditioner {
public:
    explicit TwoGridPC(const ColGrid& grid) : grid_(grid) {}

    void setup(const SparseMatrix& A) override;
    void apply(const SparseMatrix& A, Eigen::Ref<const Eigen::VectorXd> v,
               Eigen::Ref<Eigen::VectorXd> z) override;

private:
    void solve_columns(const SparseMatrix::CsrNativeType& matrix,
                       Eigen::Ref<Eigen::VectorXd> values);

    const ColGrid& grid_;
    tdma::PivotingSolver<double> column_solver_;
    tdma::PivotingSolver<double> coarse_solver_;
};
