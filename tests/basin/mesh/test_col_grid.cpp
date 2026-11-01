#include "basin/mesh/column_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <map>

namespace {

std::vector<double> shifted_axis(std::span<const double> knots, double dx, double phase) {
    auto result = make_column_axis(knots, dx);
    for (std::size_t interval = 1; interval < knots.size(); ++interval) {
        const double left = knots[interval - 1], right = knots[interval];
        const auto first = std::lower_bound(result.begin(), result.end(), left);
        const auto count = std::lower_bound(first, result.end(), right) - first;
        for (std::ptrdiff_t j = 1; j < count; ++j) {
            first[j] = std::lerp(left, right,
                                 (static_cast<double>(j) + phase) / static_cast<double>(count));
        }
    }
    return result;
}

} // namespace

TEST(ColGrid, MergeCreatesSubfacesWithoutCreatingCells) {
    const BasinGeometry source = make_test_layered_geometry();
    const std::vector<double> layout = {0.0, 1.0, 2.0};
    const double dz = 0.55;
    const ColGrid grid = make_col_grid(source, layout, dz);

    ASSERT_EQ(grid.columns.size(), 2U);
    const std::size_t cells_before_merge = grid.columns[0].size() + grid.columns[1].size();
    EXPECT_EQ(grid.cells.size(), cells_before_merge);
    EXPECT_GT(grid.vertical_subfaces.size(), 1U);
    double interface_length = 0.0;
    for (const VerticalSubface& face : grid.vertical_subfaces) {
        EXPECT_GT(face.length(), 0.0);
        interface_length += face.length();
    }
    const double common_top =
        std::max(grid.cells[grid.top_cells[0]].z_top, grid.cells[grid.top_cells[1]].z_top);
    const double common_bottom = std::min(grid.cells[grid.bottom_cells[0]].z_bottom,
                                          grid.cells[grid.bottom_cells[1]].z_bottom);
    EXPECT_NEAR(interface_length, common_bottom - common_top, 1.0e-14);
}

TEST(ColGrid, AxisRetainsContourSamples) {
    const BasinGeometry source = make_test_layered_geometry();
    const auto layout = make_column_axis(source.x, 0.75);
    EXPECT_EQ(layout, (std::vector<double>{0.0, 0.5, 1.0, 1.5, 2.0}));
}

TEST(ColGrid, PreservesLayerAreasAndCoversEveryColumnInterfaceAcrossLayouts) {
    const BasinGeometry source = make_test_layered_geometry();
    for (const double dx : {1.0, 0.75, 0.35}) {
        for (const double dz : {0.55, 0.23}) {
            for (const double phase : {-0.25, 0.0, 0.45}) {
                SCOPED_TRACE(testing::Message()
                             << "dx=" << dx << " dz=" << dz << " phase=" << phase);
                const auto grid = make_col_grid(source, shifted_axis(source.x, dx, phase), dz);
                std::map<int, double> area;
                std::vector<double> coverage(grid.columns.size() - 1, 0.0);
                for (const auto& cell : grid.cells) {
                    ASSERT_GT(cell.area(), 0.0);
                    area[cell.layer_id] += cell.area();
                }
                EXPECT_NEAR(area[1], 1.7, 1.0e-14);
                EXPECT_NEAR(area[2], 2.4, 1.0e-14);
                EXPECT_NEAR(grid.area(), 4.1, 1.0e-14);
                for (const auto& face : grid.vertical_subfaces) {
                    const auto left = grid.cells[face.left_cell].column;
                    const auto right = grid.cells[face.right_cell].column;
                    ASSERT_EQ(right, left + 1);
                    EXPECT_DOUBLE_EQ(face.x, grid.x[right]);
                    ASSERT_GT(face.length(), 0.0);
                    coverage[left] += face.length();
                }
                for (std::size_t i = 0; i < coverage.size(); ++i) {
                    const double top = std::max(grid.cells[grid.columns[i].front()].z_top,
                                                grid.cells[grid.columns[i + 1].front()].z_top);
                    const double bottom = std::min(grid.cells[grid.columns[i].back()].z_bottom,
                                                   grid.cells[grid.columns[i + 1].back()].z_bottom);
                    EXPECT_NEAR(coverage[i], bottom - top, 1.0e-12);
                }
            }
        }
    }
}

TEST(ColGrid, RepresentsVerticalOuterBoundarySteps) {
    const std::vector<double> x{0.0, 1.0, 2.0};
    const BasinGeometry source{.x = x,
                               .z = {{0.0, 0.4, 0.8}, {1.0, 1.4, 1.8}, {2.0, 2.4, 2.8}},
                               .layers = {{1, 0.0, 2.0}, {2, 0.0, 2.0}}};
    const std::vector<double> layout = {0.0, 1.0, 2.0};
    const double dz = 0.55;
    const ColGrid grid = make_col_grid(source, layout, dz);

    ASSERT_EQ(grid.surface_vertical_faces.size(), 1U);
    EXPECT_NEAR(grid.surface_vertical_faces.front().length(), 0.4, 1.0e-14);
    EXPECT_DOUBLE_EQ(grid.surface_vertical_faces.front().x, 1.0);
    EXPECT_DOUBLE_EQ(grid.surface_vertical_faces.front().outward_x, 1.0);
    ASSERT_EQ(grid.basal_vertical_faces.size(), 1U);
    EXPECT_NEAR(grid.basal_vertical_faces.front().length(), 0.4, 1.0e-14);
    EXPECT_DOUBLE_EQ(grid.basal_vertical_faces.front().outward_x, -1.0);
}
