#include "inverse/model_heat_observation_data.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {

void expect_vector_near(const Vector& actual, const Vector& expected, double tolerance = 1e-14) {
    ASSERT_EQ(actual.size(), expected.size());
    for (Vector::Index index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance);
    }
}

TEST(ModelHeatObservationDataTest, ComputesNormalizedTemperatureResiduals) {
    const ModelHeatObservationData observations(Vector{300.0, 320.0}, Vector{2.0, 5.0});

    const Vector residuals = observations.normalized_residuals(Vector{304.0, 310.0});

    expect_vector_near(residuals, Vector{2.0, -2.0});
    EXPECT_DOUBLE_EQ(residuals.squared_norm(), 8.0);
}

TEST(ModelHeatObservationDataTest, ComputesZeroResidualsForMatchingTemperatures) {
    const ModelHeatObservationData observations(Vector{280.0, 300.0, 340.0}, Vector{1.0, 2.0, 4.0});

    expect_vector_near(observations.normalized_residuals(Vector{280.0, 300.0, 340.0}),
                       Vector{0.0, 0.0, 0.0});
}

TEST(ModelHeatObservationDataTest, ExposesValidatedObservationData) {
    const ModelHeatObservationData observations(Vector{300.0, 320.0}, Vector{2.0, 5.0});

    EXPECT_EQ(observations.size(), 2);
    expect_vector_near(observations.observed_temperatures(), Vector{300.0, 320.0});
    expect_vector_near(observations.temperature_error_bounds(), Vector{2.0, 5.0});
}

TEST(ModelHeatObservationDataTest, RejectsEmptyOrMismatchedObservationData) {
    EXPECT_THROW(ModelHeatObservationData(Vector{}, Vector{}), std::invalid_argument);
    EXPECT_THROW(ModelHeatObservationData(Vector{300.0}, Vector{1.0, 2.0}), std::invalid_argument);
}

TEST(ModelHeatObservationDataTest, RejectsNonfiniteObservedTemperatures) {
    EXPECT_THROW(
        ModelHeatObservationData(Vector{std::numeric_limits<double>::quiet_NaN()}, Vector{1.0}),
        std::invalid_argument);
    EXPECT_THROW(
        ModelHeatObservationData(Vector{std::numeric_limits<double>::infinity()}, Vector{1.0}),
        std::invalid_argument);
}

TEST(ModelHeatObservationDataTest, RejectsInvalidTemperatureErrorBounds) {
    EXPECT_THROW(ModelHeatObservationData(Vector{300.0}, Vector{0.0}), std::invalid_argument);
    EXPECT_THROW(ModelHeatObservationData(Vector{300.0}, Vector{-1.0}), std::invalid_argument);
    EXPECT_THROW(
        ModelHeatObservationData(Vector{300.0}, Vector{std::numeric_limits<double>::quiet_NaN()}),
        std::invalid_argument);
    EXPECT_THROW(
        ModelHeatObservationData(Vector{300.0}, Vector{std::numeric_limits<double>::infinity()}),
        std::invalid_argument);
}

TEST(ModelHeatObservationDataTest, RejectsInvalidCalculatedTemperatures) {
    const ModelHeatObservationData observations(Vector{300.0, 320.0}, Vector{2.0, 5.0});

    EXPECT_THROW(static_cast<void>(observations.normalized_residuals(Vector{300.0})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(observations.normalized_residuals(
                     Vector{300.0, std::numeric_limits<double>::quiet_NaN()})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(observations.normalized_residuals(
                     Vector{300.0, std::numeric_limits<double>::infinity()})),
                 std::invalid_argument);
}

TEST(ModelHeatObservationDataTest, RejectsOverflowingNormalizedResidual) {
    const ModelHeatObservationData observations(Vector{-std::numeric_limits<double>::max()},
                                                Vector{1.0});

    EXPECT_THROW(static_cast<void>(
                     observations.normalized_residuals(Vector{std::numeric_limits<double>::max()})),
                 std::overflow_error);
}

} // namespace
