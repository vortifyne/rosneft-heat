#include "inverse/model_heat_parameter_space.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {

ModelHeatParameterBounds make_bounds() {
    return {
        .lower = {.surface_temperature = 250.0, .basal_heat_flux = 0.03},
        .upper = {.surface_temperature = 350.0, .basal_heat_flux = 0.12},
    };
}

void expect_vector_near(const Vector& actual, const Vector& expected, double tolerance = 1e-14) {
    ASSERT_EQ(actual.size(), expected.size());
    for (Vector::Index index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance);
    }
}

TEST(ModelHeatParameterSpaceTest, MapsPhysicalBoundsToNormalizedEndpoints) {
    const ModelHeatParameterSpace space(make_bounds());

    expect_vector_near(space.to_normalized(space.physical_bounds().lower), Vector{0.0, 0.0});
    expect_vector_near(space.to_normalized(space.physical_bounds().upper), Vector{1.0, 1.0});
}

TEST(ModelHeatParameterSpaceTest, MapsPhysicalMidpointToNormalizedMidpoint) {
    const ModelHeatParameterSpace space(make_bounds());

    expect_vector_near(
        space.to_normalized({.surface_temperature = 300.0, .basal_heat_flux = 0.075}),
        Vector{0.5, 0.5});
}

TEST(ModelHeatParameterSpaceTest, MapsNormalizedCoordinatesToNamedPhysicalParameters) {
    const ModelHeatParameterSpace space(make_bounds());

    const ModelHeatParameters parameters = space.to_physical(Vector{0.25, 0.75});

    EXPECT_DOUBLE_EQ(parameters.surface_temperature, 275.0);
    EXPECT_NEAR(parameters.basal_heat_flux, 0.0975, 1e-15);
}

TEST(ModelHeatParameterSpaceTest, RoundTripsPhysicalAndNormalizedParameters) {
    const ModelHeatParameterSpace space(make_bounds());
    const ModelHeatParameters physical{
        .surface_temperature = 287.5,
        .basal_heat_flux = 0.084,
    };
    const Vector normalized{0.625, 0.4};

    const ModelHeatParameters physical_round_trip =
        space.to_physical(space.to_normalized(physical));
    EXPECT_NEAR(physical_round_trip.surface_temperature, physical.surface_temperature, 1e-13);
    EXPECT_NEAR(physical_round_trip.basal_heat_flux, physical.basal_heat_flux, 1e-15);

    expect_vector_near(space.to_normalized(space.to_physical(normalized)), normalized);
}

TEST(ModelHeatParameterSpaceTest, ExposesNormalizedUnitBox) {
    const ModelHeatParameterSpace space(make_bounds());

    EXPECT_EQ(ModelHeatParameterSpace::kParameterCount, 2);
    expect_vector_near(space.normalized_lower_bounds(), Vector{0.0, 0.0});
    expect_vector_near(space.normalized_upper_bounds(), Vector{1.0, 1.0});
}

TEST(ModelHeatParameterSpaceTest, RejectsInvalidPhysicalBounds) {
    ModelHeatParameterBounds bounds = make_bounds();
    bounds.upper.surface_temperature = bounds.lower.surface_temperature;
    EXPECT_THROW(static_cast<void>(ModelHeatParameterSpace(bounds)), std::invalid_argument);

    bounds = make_bounds();
    bounds.upper.basal_heat_flux = bounds.lower.basal_heat_flux - 0.01;
    EXPECT_THROW(static_cast<void>(ModelHeatParameterSpace(bounds)), std::invalid_argument);

    bounds = make_bounds();
    bounds.lower.surface_temperature = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(static_cast<void>(ModelHeatParameterSpace(bounds)), std::invalid_argument);

    bounds = make_bounds();
    bounds.upper.basal_heat_flux = std::numeric_limits<double>::infinity();
    EXPECT_THROW(static_cast<void>(ModelHeatParameterSpace(bounds)), std::invalid_argument);

    bounds = {
        .lower = {.surface_temperature = -std::numeric_limits<double>::max(),
                  .basal_heat_flux = 0.03},
        .upper = {.surface_temperature = std::numeric_limits<double>::max(),
                  .basal_heat_flux = 0.12},
    };
    EXPECT_THROW(static_cast<void>(ModelHeatParameterSpace(bounds)), std::invalid_argument);
}

TEST(ModelHeatParameterSpaceTest, RejectsInvalidPhysicalParameters) {
    const ModelHeatParameterSpace space(make_bounds());

    EXPECT_THROW(space.to_normalized({.surface_temperature = 249.0, .basal_heat_flux = 0.075}),
                 std::invalid_argument);
    EXPECT_THROW(space.to_normalized({.surface_temperature = 300.0, .basal_heat_flux = 0.121}),
                 std::invalid_argument);
    EXPECT_THROW(
        space.to_normalized({.surface_temperature = std::numeric_limits<double>::infinity(),
                             .basal_heat_flux = 0.075}),
        std::invalid_argument);
}

TEST(ModelHeatParameterSpaceTest, RejectsInvalidNormalizedCoordinates) {
    const ModelHeatParameterSpace space(make_bounds());

    EXPECT_THROW(space.to_physical(Vector{}), std::invalid_argument);
    EXPECT_THROW(space.to_physical(Vector{0.5}), std::invalid_argument);
    EXPECT_THROW(space.to_physical(Vector{0.5, 0.5, 0.5}), std::invalid_argument);
    EXPECT_THROW(space.to_physical(Vector{-0.01, 0.5}), std::invalid_argument);
    EXPECT_THROW(space.to_physical(Vector{0.5, 1.01}), std::invalid_argument);
    EXPECT_THROW(space.to_physical(Vector{std::numeric_limits<double>::quiet_NaN(), 0.5}),
                 std::invalid_argument);
    EXPECT_THROW(space.to_physical(Vector{0.5, std::numeric_limits<double>::infinity()}),
                 std::invalid_argument);
}

} // namespace
