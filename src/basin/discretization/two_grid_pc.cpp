#include "basin/discretization/two_grid_pc.hpp"

#include <stdexcept>
#include <vector>

void TwoGridPC::setup(const SparseMatrix& A) {
    const auto& matrix = A.native_csr();
    const auto n = grid_.columns.size();
    if (n == 0 || matrix.rows() != matrix.cols() ||
        matrix.rows() != static_cast<Eigen::Index>(grid_.cells.size())) {
        throw std::invalid_argument("TwoGridPC matrix does not match the column grid");
    }

    // R sums cell equations and P copies a column value, so RAP sums each column block of A.
    std::vector<double> lower(n - 1), diagonal(n), upper(n - 1);
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        const auto row = grid_.cells[i].column;
        for (SparseMatrix::CsrNativeType::InnerIterator entry(matrix, i); entry; ++entry) {
            const auto col = grid_.cells[entry.col()].column;
            if (col == row)
                diagonal[row] += entry.value();
            else if (col + 1 == row)
                lower[col] += entry.value();
            else if (col == row + 1)
                upper[row] += entry.value();
            else if (entry.value() != 0.0)
                throw std::invalid_argument("TwoGridPC requires adjacent column couplings");
        }
    }
    coarse_solver_.compute({lower, diagonal, upper});
}

void TwoGridPC::apply(const SparseMatrix& A, Eigen::Ref<const Eigen::VectorXd> v,
                      Eigen::Ref<Eigen::VectorXd> z) {
    const auto& matrix = A.native_csr();
    if (v.size() != matrix.rows() || z.size() != v.size() ||
        coarse_solver_.size() != grid_.columns.size()) {
        throw std::invalid_argument("TwoGridPC requires setup and matching vectors");
    }

    std::vector<double> lower, diagonal, upper, rhs;
    for (const auto& cells : grid_.columns) {
        const auto n = cells.size();
        if (n == 0)
            throw std::invalid_argument("TwoGridPC requires nonempty columns");
        diagonal.resize(n);
        lower.resize(n - 1);
        upper.resize(n - 1);
        rhs.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto cell = cells[i];
            diagonal[i] = matrix.coeff(cell, cell);
            rhs[i] = v[cell];
            if (i + 1 < n) {
                const auto next = cells[i + 1];
                upper[i] = matrix.coeff(cell, next);
                lower[i] = matrix.coeff(next, cell);
            }
        }
        column_solver_.compute({lower, diagonal, upper});
        column_solver_.solve(rhs, rhs);
        for (std::size_t i = 0; i < n; ++i)
            z[cells[i]] = rhs[i];
    }

    const Eigen::VectorXd residual = v - matrix * z;
    std::vector<double> coarse(grid_.columns.size());
    for (const auto& cell : grid_.cells)
        coarse[cell.column] += residual[cell.index];
    coarse_solver_.solve(coarse, coarse);
    for (const auto& cell : grid_.cells)
        z[cell.index] += coarse[cell.column];
}
