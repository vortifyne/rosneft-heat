#include "basin/basin_geometry_rdp.hpp"
#include "basin/column_epoch_grid.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace {

BasinLayerProfile layer(int id, std::vector<double> x, std::vector<double> top,
                        std::vector<double> bottom) {
    return {.id = id,
            .name = "layer",
            .x = std::move(x),
            .top = std::move(top),
            .bottom = std::move(bottom),
            .lithotype = {},
            .porosity = {}};
}

BasinConfiguration old_configuration() {
    return {.age_ma = 10.0,
            .layers = {layer(2, {1.0, 2.0}, {0.0, 0.0}, {0.0, 1.0}),
                       layer(3, {0.0, 0.5, 1.0, 2.0}, {0.0, 0.0, 0.0, 1.0}, {3.0, 3.0, 3.0, 3.0})},
            .surface_temperature = {},
            .basal_heat_flux = {}};
}

BasinConfiguration new_configuration() {
    return {.age_ma = 5.0,
            .layers = {layer(1, {0.5, 1.0, 2.0}, {0.0, 0.0, 0.0}, {0.0, 0.4, 0.4}),
                       layer(2, {1.0, 2.0}, {0.4, 0.4}, {0.4, 1.2}),
                       layer(3, {0.0, 0.5, 1.0, 2.0}, {0.0, 0.0, 0.4, 1.2}, {3.0, 3.0, 3.0, 3.0})},
            .surface_temperature = {},
            .basal_heat_flux = {}};
}

} // namespace

TEST(ColumnEpochGrid, PreparesBirthAtTheLocalUppermostOldLayer) {
    const auto epoch =
        make_column_epoch_grid(make_basin_geometry(old_configuration()),
                               make_basin_geometry(new_configuration()), 10.0, 0.4, 1e-4);
    ASSERT_EQ(epoch.born_layer_id, 1);
    EXPECT_EQ(epoch.x, (std::vector<double>{0.0, 0.5, 1.0, 2.0}));
    EXPECT_EQ(epoch.layer_ids, (std::vector<int>{2, 3}));
    // The future new-layer volume belongs to layer 3 left of x=1, and layer 2 right of it.
    EXPECT_NEAR(epoch.end_boundaries[1][1], 0.0, 1e-12);
    EXPECT_NEAR(epoch.end_boundaries[1][2], 3.0, 1e-12);
    EXPECT_NEAR(epoch.end_boundaries[2][0], 0.0, 1e-12);
    EXPECT_NEAR(epoch.end_boundaries[2][1], 0.8, 1e-12);
}

TEST(ColumnEpochGrid, InterpolatesOnlyOldLayersDuringEpoch) {
    const auto epoch =
        make_column_epoch_grid(make_basin_geometry(old_configuration()),
                               make_basin_geometry(new_configuration()), 10.0, 0.4, 1e-4);
    const auto grid = epoch.at(0.5);
    const auto& column = grid.columns.back();
    EXPECT_NEAR(grid.cells[column.front()].z_top, 0.0, 1e-12);
    EXPECT_NEAR(grid.cells[column[1]].z_bottom, 0.65, 1e-12);
    EXPECT_NEAR(grid.cells[column.back()].z_bottom, 3.0, 1e-12);
    for (const auto& cell : grid.cells)
        EXPECT_NE(cell.layer_id, 1);
}

TEST(ColumnEpochGrid, MovingGridPreservesCellIdentityAndOuterArea) {
    constexpr double minimum_height = 1e-4;
    const auto epoch =
        make_column_epoch_grid(make_basin_geometry(old_configuration()),
                               make_basin_geometry(new_configuration()), 0.25, 0.4, minimum_height);
    const auto start = epoch.at(0.0);
    for (const double fraction : {0.0, 0.5, 1.0}) {
        const auto grid = epoch.at(fraction);
        ASSERT_EQ(grid.cells.size(), start.cells.size());
        EXPECT_NEAR(grid.area(), 6.0, 1e-12);
        for (std::size_t i = 0; i < grid.cells.size(); ++i) {
            EXPECT_EQ(grid.cells[i].layer_id, start.cells[i].layer_id);
            EXPECT_EQ(grid.cells[i].column, start.cells[i].column);
            EXPECT_GE(grid.cells[i].height(), minimum_height - 1e-12);
        }
        for (const auto& face : grid.vertical_subfaces)
            EXPECT_GT(face.length(), 0.0);
    }
}

TEST(ColumnEpochGrid, RdpRetainsPinchoutAndLocalOwner) {
    const auto first = make_basin_geometry(old_configuration());
    const auto second = make_basin_geometry(new_configuration());
    const auto reduced =
        make_column_epoch_grid(simplify_basin_geometry_rdp(first, 0.5),
                               simplify_basin_geometry_rdp(second, 0.5), 10.0, 0.4, 1e-4);
    EXPECT_NE(std::find(reduced.x.begin(), reduced.x.end(), 1.0), reduced.x.end());
    EXPECT_NEAR(reduced.end_boundaries.back()[1], 0.8, 1e-12);
    const auto original_on_reduced_axis =
        make_column_epoch_grid(first, second, 10.0, 0.4, 1e-4, 0.5);
    EXPECT_NE(std::find(original_on_reduced_axis.x.begin(), original_on_reduced_axis.x.end(), 1.0),
              original_on_reduced_axis.x.end());
}

TEST(ColumnEpochGrid, ReducedAxisIntegratesOriginalContactBetweenOmittedKnots) {
    const BasinConfiguration first{
        .age_ma = 1.0,
        .layers = {layer(1, {0.0, 1.0, 2.0}, {0.0, 0.1, 0.0}, {2.0, 2.1, 2.0})},
        .surface_temperature = {},
        .basal_heat_flux = {}};
    const auto geometry = make_basin_geometry(first);
    const auto fine = make_column_epoch_grid(geometry, geometry, 10.0, 3.0, 1e-4);
    const auto reduced = make_column_epoch_grid(geometry, geometry, 10.0, 3.0, 1e-4, 1.0);
    EXPECT_EQ(fine.x, (std::vector<double>{0.0, 1.0, 2.0}));
    EXPECT_EQ(reduced.x, (std::vector<double>{0.0, 2.0}));
    const auto grid = reduced.at(0.0);
    ASSERT_EQ(grid.cells.size(), 1U);
    EXPECT_NEAR(grid.cells.front().z_top, 0.05, 1e-12);
    EXPECT_NEAR(grid.cells.front().z_bottom, 2.05, 1e-12);
}
