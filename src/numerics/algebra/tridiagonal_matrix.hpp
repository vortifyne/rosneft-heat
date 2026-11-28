#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

// Fixed band: lower[i] = A(i+1,i), upper[i] = A(i,i+1).
class TridiagonalMatrix {
public:
    using Index = Eigen::Index;

    explicit TridiagonalMatrix(Index n) {
        resize(n, n);
    }
    Index rows() const noexcept {
        return static_cast<Index>(diagonal_.size());
    }
    Index cols() const noexcept {
        return rows();
    }
    Index nonZeros() const noexcept {
        return rows() + 2 * static_cast<Index>(lower_.size());
    }
    bool isCompressed() const noexcept {
        return true;
    }
    void makeCompressed() noexcept {}

    std::span<double> lower() noexcept {
        return lower_;
    }
    std::span<double> diagonal() noexcept {
        return diagonal_;
    }
    std::span<double> upper() noexcept {
        return upper_;
    }
    std::span<const double> lower() const noexcept {
        return lower_;
    }
    std::span<const double> diagonal() const noexcept {
        return diagonal_;
    }
    std::span<const double> upper() const noexcept {
        return upper_;
    }

    void resize(Index n, Index columns) {
        if (n < 0 || n != columns)
            throw std::invalid_argument("Tridiagonal matrix must be square");
        diagonal_.resize(n);
        lower_.resize(n > 0 ? n - 1 : 0);
        upper_.resize(lower_.size());
    }
    void setZero() {
        std::ranges::fill(lower_, 0.);
        std::ranges::fill(diagonal_, 0.);
        std::ranges::fill(upper_, 0.);
    }
    double coeff(Index i, Index j) const {
        if (i == j)
            return diagonal_[i];
        if (i == j + 1)
            return lower_[j];
        if (j == i + 1)
            return upper_[i];
        return 0.;
    }

    template <class Iterator> void setFromTriplets(Iterator begin, Iterator end) {
        TridiagonalMatrix result(rows());
        for (auto it = begin; it != end; ++it) {
            const auto i = it->row(), j = it->col();
            if (i < 0 || j < 0 || i >= rows() || j >= cols())
                throw std::invalid_argument("Triplet outside matrix");
            if (i == j)
                result.diagonal_[i] += it->value();
            else if (i == j + 1)
                result.lower_[j] += it->value();
            else if (j == i + 1)
                result.upper_[i] += it->value();
            else if (it->value() != 0.)
                throw std::invalid_argument("Entry outside tridiagonal band");
        }
        *this = std::move(result);
    }

    Eigen::VectorXd operator*(const Eigen::VectorXd& x) const {
        Eigen::VectorXd result(rows());
        for (Index i = 0; i < rows(); ++i)
            result[i] = diagonal_[i] * x[i];
        for (Index i = 0; i + 1 < rows(); ++i) {
            result[i] += upper_[i] * x[i + 1];
            result[i + 1] += lower_[i] * x[i];
        }
        return result;
    }

private:
    std::vector<double> lower_, diagonal_, upper_;
};
