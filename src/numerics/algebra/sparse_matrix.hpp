#pragma once

#include "numerics/algebra/tridiagonal_matrix.hpp"
#include "numerics/algebra/vector.hpp"

#include <Eigen/SparseCore>
#include <stdexcept>
#include <utility>
#include <variant>

enum class MatrixStorageFormat {
    csr,
    csc,
    tridiagonal,
};

using SparseStorageOrder = MatrixStorageFormat;

class MatrixTriplet {
public:
    using Index = Eigen::Index;

    MatrixTriplet(Index row, Index column, double value)
        : row_(row), column_(column), value_(value) {}

    Index row() const noexcept {
        return row_;
    }

    Index col() const noexcept {
        return column_;
    }

    double value() const noexcept {
        return value_;
    }

private:
    Index row_;
    Index column_;
    double value_;
};

class SparseMatrix {
public:
    using Index = Eigen::Index;
    using StorageIndex = int;
    using CsrNativeType = Eigen::SparseMatrix<double, Eigen::RowMajor, StorageIndex>;
    using CscNativeType = Eigen::SparseMatrix<double, Eigen::ColMajor, StorageIndex>;

    SparseMatrix() = default;

    SparseMatrix(Index rows, Index columns, MatrixStorageFormat storage_order)
        : storage_(make_storage(rows, columns, storage_order)) {}

    Index rows() const noexcept {
        return std::visit([](const auto& matrix) { return matrix.rows(); }, storage_);
    }

    Index cols() const noexcept {
        return std::visit([](const auto& matrix) { return matrix.cols(); }, storage_);
    }

    Index nonzero_count() const noexcept {
        return std::visit([](const auto& matrix) { return matrix.nonZeros(); }, storage_);
    }

    MatrixStorageFormat storage_order() const noexcept {
        if (std::holds_alternative<TridiagonalMatrix>(storage_))
            return MatrixStorageFormat::tridiagonal;
        return std::holds_alternative<CsrNativeType>(storage_) ? MatrixStorageFormat::csr
                                                               : MatrixStorageFormat::csc;
    }

    bool is_compressed() const noexcept {
        return std::visit([](const auto& matrix) { return matrix.isCompressed(); }, storage_);
    }

    void resize(Index rows, Index columns) {
        check_dimensions(rows, columns);
        std::visit([=](auto& matrix) { matrix.resize(rows, columns); }, storage_);
    }

    void set_zero() {
        std::visit([](auto& matrix) { matrix.setZero(); }, storage_);
    }

    void make_compressed() {
        std::visit([](auto& matrix) { matrix.makeCompressed(); }, storage_);
    }

    template <typename InputIterator>
    void set_from_triplets(InputIterator begin, InputIterator end) {
        std::visit([&](auto& matrix) { matrix.setFromTriplets(begin, end); }, storage_);
    }

    template <typename Range> void set_from_triplets(const Range& triplets) {
        set_from_triplets(triplets.begin(), triplets.end());
    }

    double coefficient(Index row, Index column) const {
        return std::visit([=](const auto& matrix) { return matrix.coeff(row, column); }, storage_);
    }

    Vector operator*(const Vector& vector) const {
        if (cols() != vector.size()) {
            throw std::invalid_argument("Sparse matrix and vector sizes do not match");
        }

        return std::visit(
            [&](const auto& matrix) {
                Vector::NativeType result = matrix * vector.native();
                return Vector(std::move(result));
            },
            storage_);
    }

    void convert_to(MatrixStorageFormat storage_order) {
        if (storage_order == this->storage_order()) {
            return;
        }

        if (storage_order == MatrixStorageFormat::tridiagonal ||
            this->storage_order() == MatrixStorageFormat::tridiagonal)
            throw std::invalid_argument("Conversion is supported only between CSR and CSC");
        if (storage_order == MatrixStorageFormat::csr) {
            CsrNativeType converted(std::get<CscNativeType>(storage_));
            converted.makeCompressed();
            storage_ = std::move(converted);
            return;
        }

        CscNativeType converted(std::get<CsrNativeType>(storage_));
        converted.makeCompressed();
        storage_ = std::move(converted);
    }

    TridiagonalMatrix& native_tridiagonal() {
        check_storage_order(MatrixStorageFormat::tridiagonal);
        return std::get<TridiagonalMatrix>(storage_);
    }
    const TridiagonalMatrix& native_tridiagonal() const {
        check_storage_order(MatrixStorageFormat::tridiagonal);
        return std::get<TridiagonalMatrix>(storage_);
    }

    CsrNativeType& native_csr() {
        check_storage_order(MatrixStorageFormat::csr);
        return std::get<CsrNativeType>(storage_);
    }

    const CsrNativeType& native_csr() const {
        check_storage_order(MatrixStorageFormat::csr);
        return std::get<CsrNativeType>(storage_);
    }

    CscNativeType& native_csc() {
        check_storage_order(MatrixStorageFormat::csc);
        return std::get<CscNativeType>(storage_);
    }

    const CscNativeType& native_csc() const {
        check_storage_order(MatrixStorageFormat::csc);
        return std::get<CscNativeType>(storage_);
    }

private:
    using Storage = std::variant<CsrNativeType, CscNativeType, TridiagonalMatrix>;

    static void check_dimensions(Index rows, Index columns) {
        if (rows < 0 || columns < 0) {
            throw std::invalid_argument("Sparse matrix dimensions must be non-negative");
        }
    }

    static Storage make_storage(Index rows, Index columns, MatrixStorageFormat storage_order) {
        check_dimensions(rows, columns);
        if (storage_order == MatrixStorageFormat::tridiagonal) {
            if (rows != columns)
                throw std::invalid_argument("Tridiagonal matrix must be square");
            return TridiagonalMatrix(rows);
        }

        if (storage_order == MatrixStorageFormat::csr) {
            return CsrNativeType(rows, columns);
        }

        return CscNativeType(rows, columns);
    }

    void check_storage_order(MatrixStorageFormat expected) const {
        if (storage_order() != expected) {
            throw std::logic_error("Sparse matrix has a different storage order");
        }
    }

    Storage storage_ = CsrNativeType{};
};
