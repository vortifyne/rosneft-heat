#include "basin/basin_geometry.hpp"

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
}

TEST(BasinGeometry, PreservesLayerSupportAndZeroThicknessIntervals) {
    const auto input = configuration({profile(1, {0., 1., 2., 3., 4., 5.}, {0., 0., 0., 0., 0., 0.},
                                              {0., 0., .001, 0., 0., 0.})});
    const auto geometry = make_basin_geometry(input);
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

} // namespace
