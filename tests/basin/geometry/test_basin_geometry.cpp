#include "basin/basin_problem.hpp"
#include "basin/geometry/basin_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

BasinLayerProfile profile(const int id, const std::vector<double>& x,
                          const std::vector<double>& top, const std::vector<double>& bottom) {
    return {.id = id, .name = {}, .x = x, .top = top, .bottom = bottom, .lithotype = {}, .phi = {}};
}

BasinConfig cfg(std::vector<BasinLayerProfile> layers, double age = 0.) {
    return {.age_ma = age, .layers = std::move(layers), .T_surf = {}, .q_basal = {}};
}

TEST(BasinGeometry, PreservesOriginalPolylines) {
    const auto input = cfg({profile(7, {0., .3, 1., 2.}, {0., .2, .8, 1.}, {2., 3., 3., 4.})}, 12.);
    const auto geometry = make_basin_geometry(input);
    ASSERT_EQ(geometry.layers.size(), 1U);
    EXPECT_EQ(geometry.x, input.layer(7).x);
    EXPECT_EQ(geometry.z[0], input.layer(7).top);
    EXPECT_EQ(geometry.z[1], input.layer(7).bottom);
    EXPECT_DOUBLE_EQ(geometry.boundary_at(0, .65), input.layer(7).top_at(.65));
}

TEST(BasinGeometry, SharesContactsAcrossDifferentSourceSamplings) {
    const auto input = cfg(
        {profile(1, {0., .5, 1., 1.5, 2.}, {0., 0., 0., 0., 0.}, {1., 1.25, 1.5, 1.25, 1.}),
         profile(2, {0., .25, 1., 1.75, 2.}, {1., 1.125, 1.5, 1.125, 1.}, {3., 3., 3., 3., 3.})});
    const auto geometry = make_basin_geometry(input);
    ASSERT_EQ(geometry.z.size(), 3U);
    EXPECT_EQ(geometry.x, (std::vector<double>{0., .25, .5, 1., 1.5, 1.75, 2.}));
    EXPECT_DOUBLE_EQ(geometry.boundary_at(1, 1.), 1.5);
    EXPECT_DOUBLE_EQ(geometry.boundary_at(2, 1.), 3.);
}

TEST(BasinGeometry, PreservesLayerSupportAndZeroThicknessIntervals) {
    const auto input = cfg({profile(1, {0., 1., 2., 3., 4., 5.}, {0., 0., 0., 0., 0., 0.},
                                    {0., 0., .001, 0., 0., 0.})});
    const auto geometry = make_basin_geometry(input);
    EXPECT_EQ(geometry.layers.front().x_min, 0.);
    EXPECT_EQ(geometry.layers.front().x_max, 5.);
    EXPECT_DOUBLE_EQ(geometry.boundary_at(1, 2.), .001);
    EXPECT_DOUBLE_EQ(geometry.boundary_at(1, .5), 0.);
    EXPECT_DOUBLE_EQ(geometry.boundary_at(1, 3.5), 0.);
}

TEST(BasinGeometry, RejectsInvalidGeometryBeforeMeshing) {
    auto input = cfg({profile(1, {0., 1.}, {0., 0.}, {1., 1.})});
    input.layers[0].bottom[1] = -1.;
    EXPECT_THROW(make_basin_geometry(input), std::invalid_argument);
    input.layers[0].bottom[1] = 1.;
    input.layers[0].x[1] = 0.;
    EXPECT_THROW(make_basin_geometry(input), std::invalid_argument);
}

} // namespace
