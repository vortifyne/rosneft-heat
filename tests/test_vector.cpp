#include "linear/vector.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

void expect_vector_equal(const Vector& actual, std::initializer_list<double> expected) {
    ASSERT_EQ(actual.size(), static_cast<Vector::Index>(expected.size()));

    Vector::Index index = 0;
    for (const double value : expected) {
        EXPECT_DOUBLE_EQ(actual[index], value);
        ++index;
    }
}

TEST(VectorTest, ConstructsAndProvidesElementAccess) {
    Vector vector{1.0, 2.0, 3.0};

    ASSERT_EQ(vector.size(), 3);
    EXPECT_FALSE(vector.empty());
    EXPECT_DOUBLE_EQ(vector[0], 1.0);
    EXPECT_DOUBLE_EQ(vector[1], 2.0);
    EXPECT_DOUBLE_EQ(vector[2], 3.0);

    vector[1] = 4.0;
    EXPECT_DOUBLE_EQ(vector[1], 4.0);
}

TEST(VectorTest, SupportsInitializationAndResize) {
    Vector vector(3);
    vector.set_constant(2.5);

    EXPECT_DOUBLE_EQ(vector[0], 2.5);
    EXPECT_DOUBLE_EQ(vector[1], 2.5);
    EXPECT_DOUBLE_EQ(vector[2], 2.5);

    vector.resize(2);
    vector.set_zero();

    EXPECT_EQ(vector.size(), 2);
    EXPECT_DOUBLE_EQ(vector[0], 0.0);
    EXPECT_DOUBLE_EQ(vector[1], 0.0);
}

TEST(VectorTest, SupportsArithmetic) {
    const Vector first{1.0, 2.0, 3.0};
    const Vector second{4.0, 5.0, 6.0};

    const Vector sum = first + second;
    const Vector difference = second - first;
    const Vector scaled = 2.0 * first;
    const Vector divided = second / 2.0;
    const Vector negated = -first;

    expect_vector_equal(sum, {5.0, 7.0, 9.0});
    expect_vector_equal(difference, {3.0, 3.0, 3.0});
    expect_vector_equal(scaled, {2.0, 4.0, 6.0});
    expect_vector_equal(divided, {2.0, 2.5, 3.0});
    expect_vector_equal(negated, {-1.0, -2.0, -3.0});
}

TEST(VectorTest, RejectsArithmeticWithDifferentSizes) {
    const Vector first{1.0, 2.0};
    const Vector second{1.0, 2.0, 3.0};

    EXPECT_THROW(static_cast<void>(first + second), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(first - second), std::invalid_argument);
}

TEST(VectorTest, ComputesNormsAndChecksFiniteValues) {
    Vector vector{3.0, -4.0};

    EXPECT_DOUBLE_EQ(vector.norm(), 5.0);
    EXPECT_DOUBLE_EQ(vector.squared_norm(), 25.0);
    EXPECT_DOUBLE_EQ(vector.infinity_norm(), 4.0);
    EXPECT_TRUE(vector.all_finite());

    vector[0] = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(vector.all_finite());
}

TEST(VectorTest, OwnsAndExposesNativeEigenVector) {
    Eigen::VectorXd native(2);
    native << 1.0, 2.0;

    Vector vector(std::move(native));
    vector.native()[0] = 3.0;

    EXPECT_DOUBLE_EQ(vector[0], 3.0);
    EXPECT_DOUBLE_EQ(vector[1], 2.0);
}

} // namespace
