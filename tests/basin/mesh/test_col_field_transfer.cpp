#include "basin/geometry/basin_geometry.hpp"
#include "basin/mesh/col_field_transfer.hpp"
#include "basin/mesh/column_test_utils.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <vector>

TEST(ColFieldTransfer, BilinearFieldAndNewLayerHistory) {
    const std::vector<double> x{0.0, 1.0, 2.0};
    const BasinGeometry old_geometry{
        .x = x, .z = {{0.0, 0.0, 0.0}, {2.0, 2.0, 2.0}}, .layers = {{2, 0.0, 2.0}}};
    const ColGrid old_grid = make_col_grid(old_geometry, x, 1.0);
    const BasinGeometry new_geometry{.x = x,
                                     .z = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {2.0, 2.0, 2.0}},
                                     .layers = {{1, 0.0, 2.0}, {2, 0.0, 2.0}}};
    const ColGrid new_grid = make_col_grid(new_geometry, {0.5, 1.5}, 2.0);
    const ColFieldTransfer map = make_col_field_transfer(old_grid, new_grid);
    std::vector<double> old_values(old_grid.cells.size());
    for (const auto& cell : old_grid.cells) {
        old_values[cell.index] = 300.0 + 2.0 * cell.center().x + 3.0 * cell.center().z;
    }
    const auto T = map.interpolate_temperature(old_values);
    const auto history = map.interpolate_same_layer(old_values);
    ASSERT_EQ(T.size(), 2U);
    ASSERT_EQ(history.size(), 2U);
    EXPECT_NEAR(T[0], 302.75, 1.0e-12);         // z=0.25: linear extrapolation.
    EXPECT_NEAR(T[1], 305.75, 1.0e-12);         // z=1.25: bilinear.
    EXPECT_TRUE(map.layer_stencils[0].empty()); // Newly born layer.
    EXPECT_NEAR(history[1], T[1], 1.0e-12);

    const ColGrid edge_grid = make_col_grid(new_geometry, {0.0, 0.5}, 2.0);
    const auto edge_map = make_col_field_transfer(old_grid, edge_grid);
    const auto edge_temperature = edge_map.interpolate_temperature(old_values);
    EXPECT_NEAR(edge_temperature.front(), 301.25, 1.0e-12);
}

TEST(ColFieldTransfer, PreservesConstantAcrossDifferentColumnPartitions) {
    const auto source = make_test_column_grid(3, 3, 4.0, 2.0);
    const auto target = make_test_column_grid(5, 4, 4.0, 2.0);
    const auto transfer = make_col_field_transfer(source, target);
    const std::vector<double> values(source.cells.size(), 315.0);
    for (double value : transfer.interpolate_temperature(values))
        EXPECT_NEAR(value, 315.0, 1e-12);
    for (double value : transfer.interpolate_same_layer(values))
        EXPECT_NEAR(value, 315.0, 1e-12);
}

namespace {

double transfer_error(std::size_t nx, std::size_t nz) {
    constexpr double kx = std::numbers::pi / 4.0;
    constexpr double kz = std::numbers::pi / 2.0;
    const auto source = make_test_column_grid(nx, nz, 4.0, 2.0);
    const auto target = make_test_column_grid(nx * 2, nz * 2, 4.0, 2.0);
    const auto exact = [](double x, double z) {
        return 300.0 + 15.0 * std::cos(kx * x) * std::sin(kz * z);
    };
    std::vector<double> values(source.cells.size());
    for (const auto& cell : source.cells)
        values[cell.index] = exact(cell.center().x, cell.center().z);
    const auto transferred =
        make_col_field_transfer(source, target).interpolate_temperature(values);
    return area_weighted_error(target, transferred, exact);
}

} // namespace

TEST(ColFieldTransfer, SmoothFieldConvergesUnderRefinement) {
    const double coarse = transfer_error(8, 4);
    const double fine = transfer_error(16, 8);
    EXPECT_GT(coarse / fine, 1.7) << coarse << ", " << fine;
}
