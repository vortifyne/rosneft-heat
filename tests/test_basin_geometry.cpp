#include "basin/basin_geometry.hpp"
#include "basin/layered_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

BasinLayerProfile profile(const int id, const std::vector<double>& x,
                          const std::vector<double>& top, const std::vector<double>& bottom) {
    return {.id = id,
            .name = {},
            .x = x,
            .top = top,
            .bottom = bottom,
            .lithotype = {},
            .porosity = {}};
}

BasinConfiguration configuration(std::vector<BasinLayerProfile> layers, double age = 0.) {
    return {.age_ma = age,
            .layers = std::move(layers),
            .surface_temperature = {},
            .basal_heat_flux = {}};
}

TEST(BasinGeometry, ExactModePreservesOriginalPolylines) {
    const auto input =
        configuration({profile(7, {0., .3, 1., 2.}, {0., .2, .8, 1.}, {2., 3., 3., 4.})}, 12.);
    const auto geometry = make_basin_geometry(input);
    EXPECT_DOUBLE_EQ(geometry.age_ma, input.age_ma);
    ASSERT_EQ(geometry.layers.size(), 1U);
    EXPECT_EQ(geometry.layer(7).x, input.layer(7).x);
    EXPECT_EQ(geometry.layer(7).top, input.layer(7).top);
    EXPECT_EQ(geometry.layer(7).bottom, input.layer(7).bottom);
    EXPECT_DOUBLE_EQ(geometry.layer(7).top_at(.65), input.layer(7).top_at(.65));
}

TEST(BasinGeometry, SharesContactsAcrossDifferentSourceSamplings) {
    const auto input = configuration(
        {profile(1, {0., .5, 1., 1.5, 2.}, {0., 0., 0., 0., 0.}, {1., 1.25, 1.5, 1.25, 1.}),
         profile(2, {0., .25, 1., 1.75, 2.}, {1., 1.125, 1.5, 1.125, 1.}, {3., 3., 3., 3., 3.})});
    const auto geometry = make_basin_geometry(input);
    ASSERT_EQ(geometry.boundaries.size(), 3U);
    EXPECT_EQ(geometry.layers[0].bottom_boundary, geometry.layers[1].top_boundary);
    EXPECT_EQ(geometry.layer(1).x, geometry.layer(2).x);
    for (const double x : geometry.layer(1).x) {
        EXPECT_DOUBLE_EQ(geometry.layer(1).bottom_at(x), geometry.layer(2).top_at(x));
    }
    const auto layout = make_layered_mesh_layout({&geometry, 1}, .5);
    const auto mesh = make_layered_mesh(geometry, layout);
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, geometry));
}

TEST(BasinGeometry, PreservesLayerSupportAndZeroThicknessIntervals) {
    const auto input = configuration({profile(1, {0., 1., 2., 3., 4., 5.}, {0., 0., 0., 0., 0., 0.},
                                              {0., 0., .001, 0., 0., 0.})});
    const auto geometry = make_basin_grid_geometry(input, 1.);
    const auto& layer = geometry.layer(1);
    EXPECT_EQ(layer.x.front(), 0.);
    EXPECT_EQ(layer.x.back(), 5.);
    EXPECT_DOUBLE_EQ(layer.bottom_at(2.), .001);
    EXPECT_DOUBLE_EQ(layer.bottom_at(.5), 0.);
    EXPECT_DOUBLE_EQ(layer.bottom_at(3.5), 0.);
}

TEST(BasinGeometry, RejectsInvalidGeometryBeforeMeshing) {
    auto input = configuration({profile(1, {0., 1.}, {0., 0.}, {1., 1.})});
    input.layers[0].bottom[1] = -1.;
    EXPECT_THROW(make_basin_geometry(input), std::invalid_argument);
    input.layers[0].bottom[1] = 1.;
    input.layers[0].x[1] = 0.;
    EXPECT_THROW(make_basin_geometry(input), std::invalid_argument);
}

TEST(BasinGeometry, GridContourSimplifiesInteriorButPreservesThermalBoundaries) {
    std::vector<double> x, roof, contact, floor;
    for (int i = 0; i <= 1000; ++i) {
        const double p = i * .01;
        x.push_back(p);
        roof.push_back(.001 * std::sin(p));
        contact.push_back(2. + .2 * p + .001 * std::sin(p));
        floor.push_back(6. + .001 * std::cos(p));
    }
    const auto input = configuration({profile(1, x, roof, contact), profile(2, x, contact, floor)});
    const auto geometry = make_basin_grid_geometry(input, 1.);
    const auto& prepared = geometry.boundaries[1].depth;
    ASSERT_EQ(prepared.x.size(), 11U);
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(geometry.boundaries.front().depth.at(x[i]), roof[i]);
        EXPECT_DOUBLE_EQ(geometry.boundaries.back().depth.at(x[i]), floor[i]);
        EXPECT_LE(std::abs(prepared.at(x[i]) - contact[i]), .01);
    }
    EXPECT_EQ(input.layer(1).x, x);
}

TEST(BasinGeometry, GridContourRetainsThinLayerAndPinchWithoutOpeningAbsentMaterial) {
    const auto input = configuration({profile(
        1, {0., 1., 1.1, 1.2, 2., 3.}, {0., 0., 0., 0., 0., 0.}, {0., 0., .001, 0., 0., 0.})});
    const auto geometry = make_basin_grid_geometry(input, 1.);
    const auto prepared = geometry.layer(1);
    EXPECT_DOUBLE_EQ(prepared.bottom_at(1.1), .001);
    EXPECT_DOUBLE_EQ(prepared.bottom_at(.5), 0.);
    EXPECT_DOUBLE_EQ(prepared.bottom_at(1.5), 0.);
    EXPECT_THROW(make_basin_grid_geometry(input, 0.), std::invalid_argument);
}

TEST(BasinGeometry, GridContourKeepsContactSharedOutsideIntermediateLayerSupport) {
    const auto input = configuration(
        {profile(1, {0., 1., 2., 3., 4.}, {0., 1.1, 0., 0., 0.}, {1., 1.1, 1.4, 1.9, 2.6}),
         profile(2, {2., 3., 4.}, {1.4, 1.9, 2.6}, {1.4, 2.2, 3.}),
         profile(3, {0., 1., 2., 3., 4.}, {1., 1.1, 1.4, 2.2, 3.}, {4., 4., 4., 4., 4.})});
    const auto geometry = make_basin_grid_geometry(input, 1.5);
    const auto upper = geometry.layer(1);
    const auto lower = geometry.layer(3);
    for (int i = 0; i <= 40; ++i) {
        const double x = .05 * i;
        EXPECT_DOUBLE_EQ(upper.bottom_at(x), lower.top_at(x));
    }
}

TEST(BasinGeometry, GridContourDoesNotPropagateCurvedContactNodesToFlatHorizons) {
    std::vector<double> x, zero, contact, bottom;
    for (int i = 0; i <= 100; ++i) {
        const double p = .1 * i;
        x.push_back(p);
        zero.push_back(0.);
        contact.push_back(20. + 10. * std::sin(p));
        bottom.push_back(40.);
    }
    const auto input =
        configuration({profile(1, x, zero, contact), profile(2, x, contact, bottom)});
    const auto geometry = make_basin_grid_geometry(input, 1.);
    ASSERT_EQ(geometry.boundaries.size(), 3U);
    EXPECT_EQ(geometry.boundaries.front().depth.x.size(), x.size());
    EXPECT_EQ(geometry.boundaries.back().depth.x.size(), x.size());
    EXPECT_GT(geometry.boundaries[1].depth.x.size(), 11U);
}

TEST(BasinGeometry, GridContourEquidistributesArcLengthOnSteepStraightBoundary) {
    const auto input =
        configuration({profile(1, {0., .1, 1., 4.}, {0., 1., 10., 40.}, {2., 3., 12., 42.})});
    const auto geometry = make_basin_grid_geometry(input, 1.);
    const auto& boundary = geometry.boundaries.front().depth;
    EXPECT_DOUBLE_EQ(boundary.x.front(), 0.);
    EXPECT_DOUBLE_EQ(boundary.x.back(), 4.);
    EXPECT_GT(boundary.x.size(), 20U);
    for (std::size_t i = 1; i < boundary.x.size(); ++i) {
        const double dx = boundary.x[i] - boundary.x[i - 1];
        const double dz = boundary.values[i] - boundary.values[i - 1];
        EXPECT_GT(dx, 0.);
        EXPECT_LE(std::hypot(dx, dz), 1.5);
        EXPECT_NEAR(boundary.values[i], 10. * boundary.x[i], 1e-12);
    }
}

TEST(BasinGeometry, GridContourRetainsNarrowTurningPoint) {
    const auto input = configuration(
        {profile(1, {0., 4.6, 4.7, 4.8, 10.}, {0., 0., 0., 0., 0.}, {2., 2.1, 4., 2.1, 2.})});
    const auto geometry = make_basin_grid_geometry(input, 5.);
    EXPECT_DOUBLE_EQ(geometry.layer(1).bottom_at(4.7), 4.);
}

} // namespace
