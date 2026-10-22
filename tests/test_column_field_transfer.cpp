#include "basin/column_field_transfer.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(ColumnFieldTransfer, BilinearFieldAndNewLayerHistory) {
    const std::vector<double> x{0.0, 1.0, 2.0};
    const BasinGeometry old_geometry{
        .age_ma = 1.0,
        .boundaries = {{{x, {0.0, 0.0, 0.0}}}, {{x, {2.0, 2.0, 2.0}}}},
        .layers = {{2, 0, 1, 0.0, 2.0}}};
    const NonmatchingColumnGrid old_grid = make_nonmatching_column_grid(
        old_geometry, {.x = x, .cell_size = 1.0});
    const BasinGeometry new_geometry{
        .age_ma = 0.0,
        .boundaries = {{{x, {0.0, 0.0, 0.0}}},
                       {{x, {0.5, 0.5, 0.5}}},
                       {{x, {2.0, 2.0, 2.0}}}},
        .layers = {{1, 0, 1, 0.0, 2.0}, {2, 1, 2, 0.0, 2.0}}};
    const NonmatchingColumnGrid new_grid = make_nonmatching_column_grid(
        new_geometry, {.x = {0.5, 1.5}, .cell_size = 2.0});
    const ColumnFieldTransfer map = make_column_field_transfer(old_grid, new_grid);
    std::vector<double> old_values(old_grid.cells.size());
    for (const auto& cell : old_grid.cells) {
        old_values[cell.index] = 300.0 + 2.0 * cell.center().x + 3.0 * cell.center().z;
    }
    const auto temperature = map.interpolate_temperature(old_values);
    const auto history = map.interpolate_same_layer(old_values);
    ASSERT_EQ(temperature.size(), 2U);
    ASSERT_EQ(history.size(), 2U);
    EXPECT_NEAR(temperature[0], 302.75, 1.0e-12); // z=0.25: linear extrapolation.
    EXPECT_NEAR(temperature[1], 305.75, 1.0e-12); // z=1.25: bilinear.
    EXPECT_TRUE(map.same_layer[0].empty());        // Newly born layer.
    EXPECT_NEAR(history[1], temperature[1], 1.0e-12);

    const NonmatchingColumnGrid edge_grid = make_nonmatching_column_grid(
        new_geometry, {.x = {0.0, 0.5}, .cell_size = 2.0});
    const auto edge_map = make_column_field_transfer(old_grid, edge_grid);
    const auto edge_temperature = edge_map.interpolate_temperature(old_values);
    EXPECT_NEAR(edge_temperature.front(), 301.25, 1.0e-12);
}
