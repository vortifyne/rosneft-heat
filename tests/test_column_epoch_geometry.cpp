#include "basin/column_epoch_geometry.hpp"
#include "basin/column_epoch_grid.hpp"

#include <gtest/gtest.h>
#include <algorithm>
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
                       layer(3, {0.0, 0.5, 1.0, 2.0}, {0.0, 0.0, 0.0, 1.0},
                             {3.0, 3.0, 3.0, 3.0})},
            .surface_temperature = {}, .basal_heat_flux = {}};
}

BasinConfiguration new_configuration() {
    return {.age_ma = 5.0,
            .layers = {layer(1, {0.5, 1.0, 2.0}, {0.0, 0.0, 0.0}, {0.0, 0.4, 0.4}),
                       layer(2, {1.0, 2.0}, {0.4, 0.4}, {0.4, 1.2}),
                       layer(3, {0.0, 0.5, 1.0, 2.0}, {0.0, 0.0, 0.4, 1.2},
                             {3.0, 3.0, 3.0, 3.0})},
            .surface_temperature = {}, .basal_heat_flux = {}};
}

} // namespace

TEST(ColumnEpochGeometry, NewPartialLayerIsMergedIntoTheLocalUppermostOldLayer) {
    const ColumnEpochGeometry epoch =
        make_column_epoch_geometry(old_configuration(), new_configuration());
    ASSERT_EQ(epoch.born_layer_id, 1);
    const auto axis = make_column_epoch_axis(epoch, 10.0);
    EXPECT_NE(std::find(axis.begin(), axis.end(), 1.0), axis.end());
    const auto left = sample_column_epoch(epoch, 0.5, 1.0, 1.0);
    const auto right = sample_column_epoch(epoch, 1.0, 2.0, 1.0);
    ASSERT_EQ(right.layer_ids, (std::vector<int>{2, 3}));

    // Left of the old layer-2 pinchout, the new area belonged to layer 3.
    EXPECT_NEAR(left.boundaries[1], 0.0, 1.0e-12);
    EXPECT_NEAR(left.boundaries[2], 3.0, 1.0e-12);
    // Right of the pinchout, it belonged to layer 2.
    EXPECT_NEAR(right.boundaries[0], 0.0, 1.0e-12);
    EXPECT_NEAR(right.boundaries[1], 0.8, 1.0e-12);
    // The volume born at the event was assigned to the old material above
    // each local contact: to layer 3 on the left and layer 2 on the right.
    EXPECT_NEAR((left.boundaries[2] - left.boundaries[1]) * 0.5,
                (3.0 - 0.2) * 0.5 + 0.2 * 0.5, 1.0e-12);
    EXPECT_NEAR((right.boundaries[1] - right.boundaries[0]) * 1.0,
                0.4 * 1.0 + 0.4 * 1.0, 1.0e-12);

    for (std::size_t i = 1; i < axis.size(); ++i) {
        const auto section = sample_column_epoch(epoch, axis[i - 1], axis[i], 1.0);
        const double width = axis[i] - axis[i - 1];
        const auto top = epoch.after_event.boundaries.front().depth;
        const auto bottom = epoch.after_event.boundaries.back().depth;
        EXPECT_NEAR(section.boundaries.front(),
                    0.5 * (top.at(axis[i - 1]) + top.at(axis[i])), 1.0e-12);
        EXPECT_NEAR(section.boundaries.back(),
                    0.5 * (bottom.at(axis[i - 1]) + bottom.at(axis[i])), 1.0e-12);
        EXPECT_GT(width, 0.0);
    }
}

TEST(ColumnEpochGeometry, InterpolatesOnlyOldLayerBoundariesDuringEpoch) {
    const ColumnEpochGeometry epoch =
        make_column_epoch_geometry(old_configuration(), new_configuration());
    const auto midpoint = sample_column_epoch(epoch, 1.0, 2.0, 0.5);
    EXPECT_EQ(midpoint.layer_ids.size(), 2U);
    EXPECT_NEAR(midpoint.boundaries[0], 0.0, 1.0e-12);
    EXPECT_NEAR(midpoint.boundaries[1], 0.65, 1.0e-12);
    EXPECT_NEAR(midpoint.boundaries[2], 3.0, 1.0e-12);
}

TEST(ColumnEpochGeometry, MovingGridPreservesCellIdentityAndOuterArea) {
    const auto geometry = make_column_epoch_geometry(old_configuration(), new_configuration());
    const auto epoch = make_column_epoch_grid(geometry, 0.25, 0.4, 1.0e-4);
    const auto start = epoch.at(0.0);
    const auto middle = epoch.at(0.5);
    const auto end = epoch.at(1.0);
    ASSERT_EQ(start.cells.size(), end.cells.size());
    ASSERT_EQ(start.cells.size(), middle.cells.size());
    for (std::size_t cell = 0; cell < start.cells.size(); ++cell) {
        EXPECT_EQ(start.cells[cell].layer_id, middle.cells[cell].layer_id);
        EXPECT_EQ(start.cells[cell].layer_id, end.cells[cell].layer_id);
        EXPECT_EQ(start.cells[cell].column, end.cells[cell].column);
        EXPECT_GT(start.cells[cell].height(), 0.0);
        EXPECT_GT(middle.cells[cell].height(), 0.0);
        EXPECT_GT(end.cells[cell].height(), 0.0);
    }
    for (std::size_t column = 0; column < epoch.rows.size(); ++column) {
        const auto section = sample_column_epoch(geometry, epoch.x[column],
                                                  epoch.x[column + 1], 1.0);
        const auto& cells = end.columns[column];
        EXPECT_NEAR(end.cells[cells.front()].z_top, section.boundaries.front(), 1.0e-9);
        EXPECT_NEAR(end.cells[cells.back()].z_bottom, section.boundaries.back(), 1.0e-9);
    }
    for (const auto& face : end.vertical_subfaces) {
        EXPECT_GT(face.length(), 0.0);
    }
}

TEST(ColumnEpochGeometry, RejectsColumnCrossingOwnerChange) {
    const auto epoch = make_column_epoch_geometry(old_configuration(), new_configuration());
    EXPECT_THROW(sample_column_epoch(epoch, 0.75, 1.25, 1.0), std::invalid_argument);
}

TEST(ColumnEpochGeometry, RdpRetainsPinchoutAndLocalOwner) {
    const auto original = make_column_epoch_geometry(old_configuration(), new_configuration());
    const auto reduced = make_column_epoch_geometry(old_configuration(), new_configuration(), 0.5);
    EXPECT_LE(reduced.start.boundaries.front().depth.x.size(),
              original.start.boundaries.front().depth.x.size());
    const auto axis = make_column_epoch_axis(reduced, 10.0);
    EXPECT_NE(std::find(axis.begin(), axis.end(), 1.0), axis.end());
    const auto right = sample_column_epoch(reduced, 1.0, 2.0, 1.0);
    EXPECT_NEAR(right.boundaries[1], 0.8, 1.0e-12);
    const auto original_on_reduced_axis =
        make_column_epoch_grid(original, 10.0, 0.4, 1.0e-4, 0.5);
    EXPECT_NE(std::find(original_on_reduced_axis.x.begin(),
                        original_on_reduced_axis.x.end(), 1.0),
              original_on_reduced_axis.x.end());
}

TEST(ColumnEpochGeometry, ReducedAxisIntegratesOriginalContactBetweenOmittedKnots) {
    const BasinConfiguration first{
        .age_ma = 1.0,
        .layers = {layer(1, {0.0, 1.0, 2.0}, {0.0, 0.1, 0.0},
                         {2.0, 2.1, 2.0})},
        .surface_temperature = {}, .basal_heat_flux = {}};
    BasinConfiguration second = first;
    second.age_ma = 0.0;
    const auto geometry = make_column_epoch_geometry(first, second);
    const auto fine_axis = make_column_epoch_grid(geometry, 10.0, 3.0, 0.0);
    const auto reduced_axis = make_column_epoch_grid(geometry, 10.0, 3.0, 0.0, 1.0);
    EXPECT_EQ(fine_axis.x, (std::vector<double>{0.0, 1.0, 2.0}));
    EXPECT_EQ(reduced_axis.x, (std::vector<double>{0.0, 2.0}));
    const auto grid = reduced_axis.at(0.0);
    ASSERT_EQ(grid.cells.size(), 1U);
    EXPECT_NEAR(grid.cells.front().z_top, 0.05, 1.0e-12);
    EXPECT_NEAR(grid.cells.front().z_bottom, 2.05, 1.0e-12);
}

TEST(ColumnEpochGeometry, ExactZeroRemovesCollapsedRowsAndRejoinsNeighbors) {
    const BasinConfiguration first{
        .age_ma = 1.0,
        .layers = {layer(1, {0.0, 2.0}, {0.0, 0.0}, {0.5, 0.5}),
                   layer(2, {0.0, 2.0}, {0.5, 0.5}, {2.0, 2.0})},
        .surface_temperature = {}, .basal_heat_flux = {}};
    const BasinConfiguration second{
        .age_ma = 0.0,
        .layers = {layer(1, {0.0, 2.0}, {0.0, 0.0}, {0.0, 0.0}),
                   layer(2, {0.0, 2.0}, {0.0, 0.0}, {2.0, 2.0})},
        .surface_temperature = {}, .basal_heat_flux = {}};
    const auto epoch = make_column_epoch_grid(make_column_epoch_geometry(first, second),
                                              1.0, 0.5, 0.0);
    const auto start = epoch.at_active(0.0);
    const auto end = epoch.at_active(1.0);
    ASSERT_GT(start.grid.cells.size(), end.grid.cells.size());
    ASSERT_EQ(end.grid.cells.size(), end.full_indices.size());
    for (const auto& cell : end.grid.cells) {
        EXPECT_EQ(cell.layer_id, 2);
        EXPECT_GT(cell.height(), 0.0);
    }
    for (const auto index : end.grid.top_cells) {
        EXPECT_EQ(end.grid.cells[index].layer_id, 2);
        EXPECT_NEAR(end.grid.cells[index].z_top, 0.0, 1.0e-12);
    }
}
