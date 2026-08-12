#include "linear/sparse_matrix.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {

std::vector<MatrixTriplet> make_test_triplets() {
    return {
        {0, 0, 4.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 2.0},
        {1, 1, 2.0}, {1, 2, -1.0}, {2, 1, -1.0}, {2, 2, 3.0},
    };
}

void expect_test_coefficients(const SparseMatrix& matrix) {
    EXPECT_DOUBLE_EQ(matrix.coefficient(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(matrix.coefficient(0, 1), -1.0);
    EXPECT_DOUBLE_EQ(matrix.coefficient(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(matrix.coefficient(1, 0), -1.0);
    EXPECT_DOUBLE_EQ(matrix.coefficient(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(matrix.coefficient(1, 2), -1.0);
    EXPECT_DOUBLE_EQ(matrix.coefficient(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(matrix.coefficient(2, 1), -1.0);
    EXPECT_DOUBLE_EQ(matrix.coefficient(2, 2), 3.0);
}

TEST(SparseMatrixTest, CreatesRequestedStorageOrder) {
    const SparseMatrix csr_matrix(2, 3, SparseStorageOrder::csr);
    const SparseMatrix csc_matrix(4, 5, SparseStorageOrder::csc);

    EXPECT_EQ(csr_matrix.rows(), 2);
    EXPECT_EQ(csr_matrix.cols(), 3);
    EXPECT_EQ(csr_matrix.storage_order(), SparseStorageOrder::csr);
    EXPECT_EQ(csc_matrix.rows(), 4);
    EXPECT_EQ(csc_matrix.cols(), 5);
    EXPECT_EQ(csc_matrix.storage_order(), SparseStorageOrder::csc);
}

TEST(SparseMatrixTest, BuildsCompressedCsrFromProjectTriplets) {
    SparseMatrix matrix(3, 3, SparseStorageOrder::csr);
    const auto triplets = make_test_triplets();

    matrix.set_from_triplets(triplets);

    EXPECT_TRUE(matrix.is_compressed());
    EXPECT_EQ(matrix.nonzero_count(), 7);
    expect_test_coefficients(matrix);
}

TEST(SparseMatrixTest, BuildsCompressedCscFromProjectTriplets) {
    SparseMatrix matrix(3, 3, SparseStorageOrder::csc);
    const auto triplets = make_test_triplets();

    matrix.set_from_triplets(triplets.begin(), triplets.end());

    EXPECT_TRUE(matrix.is_compressed());
    EXPECT_EQ(matrix.nonzero_count(), 7);
    expect_test_coefficients(matrix);
}

TEST(SparseMatrixTest, ExplicitlyConvertsBetweenCsrAndCsc) {
    SparseMatrix matrix(3, 3, SparseStorageOrder::csr);
    const auto triplets = make_test_triplets();
    matrix.set_from_triplets(triplets);

    matrix.convert_to(SparseStorageOrder::csc);

    EXPECT_EQ(matrix.storage_order(), SparseStorageOrder::csc);
    EXPECT_TRUE(matrix.is_compressed());
    EXPECT_EQ(matrix.nonzero_count(), 7);
    expect_test_coefficients(matrix);

    matrix.convert_to(SparseStorageOrder::csr);

    EXPECT_EQ(matrix.storage_order(), SparseStorageOrder::csr);
    EXPECT_TRUE(matrix.is_compressed());
    EXPECT_EQ(matrix.nonzero_count(), 7);
    expect_test_coefficients(matrix);
}

TEST(SparseMatrixTest, MultipliesVectorInEitherStorageOrder) {
    SparseMatrix matrix(3, 3, SparseStorageOrder::csr);
    const auto triplets = make_test_triplets();
    matrix.set_from_triplets(triplets);
    const Vector vector{1.0, 2.0, 3.0};

    const Vector csr_result = matrix * vector;
    matrix.convert_to(SparseStorageOrder::csc);
    const Vector csc_result = matrix * vector;

    ASSERT_EQ(csr_result.size(), 3);
    ASSERT_EQ(csc_result.size(), 3);
    EXPECT_DOUBLE_EQ(csr_result[0], 2.0);
    EXPECT_DOUBLE_EQ(csr_result[1], 4.0);
    EXPECT_DOUBLE_EQ(csr_result[2], 7.0);
    EXPECT_DOUBLE_EQ(csc_result[0], 2.0);
    EXPECT_DOUBLE_EQ(csc_result[1], 4.0);
    EXPECT_DOUBLE_EQ(csc_result[2], 7.0);
}

TEST(SparseMatrixTest, RejectsMatrixVectorSizeMismatch) {
    const SparseMatrix matrix(2, 3, SparseStorageOrder::csr);
    const Vector vector(2);

    EXPECT_THROW(static_cast<void>(matrix * vector), std::invalid_argument);
}

TEST(SparseMatrixTest, ProvidesNativeAccessOnlyForActiveStorageOrder) {
    SparseMatrix matrix(2, 2, SparseStorageOrder::csr);

    EXPECT_EQ(matrix.native_csr().rows(), 2);
    EXPECT_THROW(static_cast<void>(matrix.native_csc()), std::logic_error);

    matrix.convert_to(SparseStorageOrder::csc);

    EXPECT_EQ(matrix.native_csc().rows(), 2);
    EXPECT_THROW(static_cast<void>(matrix.native_csr()), std::logic_error);
}

TEST(SparseMatrixTest, ClearsValuesWithoutChangingDimensionsOrStorageOrder) {
    SparseMatrix matrix(3, 3, SparseStorageOrder::csc);
    const auto triplets = make_test_triplets();
    matrix.set_from_triplets(triplets);

    matrix.set_zero();

    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_EQ(matrix.nonzero_count(), 0);
    EXPECT_EQ(matrix.storage_order(), SparseStorageOrder::csc);
}

TEST(SparseMatrixTest, RejectsNegativeDimensions) {
    EXPECT_THROW(SparseMatrix(-1, 2, SparseStorageOrder::csr), std::invalid_argument);
    EXPECT_THROW(SparseMatrix(2, -1, SparseStorageOrder::csc), std::invalid_argument);
}

} // namespace
