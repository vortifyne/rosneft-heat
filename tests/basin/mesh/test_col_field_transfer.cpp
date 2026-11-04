#include "basin/geometry/basin_geometry.hpp"
#include "basin/mesh/col_field_transfer.hpp"
#include "basin/mesh/column_test_utils.hpp"

#include <algorithm>
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
    Vector old_T(static_cast<Vector::Index>(old_grid.cells.size()));
    constexpr std::size_t reactions = 3;
    std::vector<double> old_I((old_grid.cells.size() + old_grid.columns.size()) * reactions);
    for (const auto& cell : old_grid.cells) {
        const auto [x, z] = cell.center();
        const double value = 300.0 + 2.0 * x + 3.0 * z + 4.0 * x * z;
        old_T[static_cast<Vector::Index>(cell.index)] = value;
    }
    const auto depths = horizontal_node_depths(old_grid);
    for (const auto& column : old_grid.columns) {
        const auto c = old_grid.cells[column.front()].column;
        const double x = 0.5 * (old_grid.x[c] + old_grid.x[c + 1]);
        for (std::size_t node = column.front() + c; node <= column.back() + c + 1; ++node)
            for (std::size_t reaction = 0; reaction < reactions; ++reaction)
                old_I[node * reactions + reaction] =
                    (300.0 + 2.0 * x + 3.0 * depths[node] + 4.0 * x * depths[node]) *
                    static_cast<double>(reaction + 1);
    }
    const auto T = interpolate_temperature(old_grid, new_grid, old_T);
    const auto I = interpolate_reaction_integrals(old_grid, new_grid, old_I, reactions, 1);
    ASSERT_EQ(T.size(), 2);
    ASSERT_EQ(I.size(), 3 * reactions);
    EXPECT_NEAR(T[0], 303.75, 1.0e-12); // z=0.25: linear extrapolation.
    EXPECT_NEAR(T[1], 310.75, 1.0e-12); // z=1.25: bilinear.
    for (std::size_t reaction = 0; reaction < reactions; ++reaction) {
        EXPECT_DOUBLE_EQ(I[reaction], 0.0);
        EXPECT_DOUBLE_EQ(I[reactions + reaction], 0.0);
        EXPECT_NEAR(I[2 * reactions + reaction], 316.0 * static_cast<double>(reaction + 1),
                    1.0e-12);
    }

    const ColGrid edge_grid = make_col_grid(new_geometry, {0.0, 0.5}, 2.0);
    const auto edge_temperature = interpolate_temperature(old_grid, edge_grid, old_T);
    EXPECT_NEAR(edge_temperature[0], 301.5, 1.0e-12);
}

TEST(ColFieldTransfer, PreservesConstantAcrossDifferentColumnPartitions) {
    const auto source = make_test_column_grid(3, 3, 4.0, 2.0);
    const auto target = make_test_column_grid(5, 4, 4.0, 2.0);
    Vector T(static_cast<Vector::Index>(source.cells.size()));
    T.set_constant(315.0);
    const auto next_T = interpolate_temperature(source, target, T);
    for (Vector::Index cell = 0; cell < next_T.size(); ++cell)
        EXPECT_NEAR(next_T[cell], 315.0, 1e-12);
    const std::vector<double> I(source.cells.size() + source.columns.size(), 315.0);
    for (double value : interpolate_reaction_integrals(source, target, I, 1, -1))
        EXPECT_NEAR(value, 315.0, 1e-12);
}

TEST(ColFieldTransfer, BilinearFieldOnNonmatchingColumns) {
    const auto source = make_test_column_grid(3, 3, 4.0, 2.0);
    const auto target = make_test_column_grid(5, 4, 4.0, 2.0);
    const auto exact = [](Point2D p) { return 300.0 + 2.0 * p.x + 3.0 * p.z + 4.0 * p.x * p.z; };
    Vector T(static_cast<Vector::Index>(source.cells.size()));
    for (const auto& cell : source.cells)
        T[static_cast<Vector::Index>(cell.index)] = exact(cell.center());
    const auto next_T = interpolate_temperature(source, target, T);
    for (const auto& cell : target.cells)
        EXPECT_NEAR(next_T[static_cast<Vector::Index>(cell.index)], exact(cell.center()), 1e-12);
}

TEST(ColFieldTransfer, NodalHistoriesStayWithinTheirLayer) {
    const BasinGeometry first{.x = {0.0, 2.0},
                              .z = {{0.0, 0.0}, {1.0, 1.0}, {3.0, 3.0}},
                              .layers = {{7, 0.0, 2.0}, {2, 0.0, 2.0}}};
    auto second = first;
    second.z[1] = {1.5, 1.5};
    const auto source = make_col_grid(first, {0.0, 1.0, 2.0}, 0.5);
    const auto target = make_col_grid(second, {0.5, 1.5}, 0.25);
    const auto old_z = horizontal_node_depths(source);
    std::vector<double> I(old_z.size());
    for (std::size_t i = 0; i < I.size(); ++i)
        I[i] = old_z[i] <= 1.0 ? 10.0 + 20.0 * old_z[i] : 30.0 + 85.0 * (old_z[i] - 1.0);
    const auto next = interpolate_reaction_integrals(source, target, I, 1, -1);
    const auto z = horizontal_node_depths(target);
    for (std::size_t i = 0; i < next.size(); ++i) {
        const double expected =
            z[i] < 1.5 ? 10.0 + 20.0 * std::min(z[i], 1.0) : 30.0 + 85.0 * (z[i] - 1.0);
        EXPECT_NEAR(next[i], expected, 1e-12);
    }
}

TEST(ColFieldTransfer, ReconstructsLinearTemperatureAtHorizontalNodes) {
    const auto grid = make_test_column_grid(2, 3, 4.0, 2.0);
    Vector T(static_cast<Vector::Index>(grid.cells.size()));
    for (const auto& cell : grid.cells)
        T[static_cast<Vector::Index>(cell.index)] = 300.0 + 5.0 * cell.center().z;
    const auto z = horizontal_node_depths(grid);
    std::vector<double> nodes(z.size());
    interpolate_horizontal_temperature(grid, T, nodes);
    for (std::size_t i = 0; i < nodes.size(); ++i)
        EXPECT_NEAR(nodes[i], 300.0 + 5.0 * z[i], 1e-12);
}

TEST(ColFieldTransfer, SingleSourceCell) {
    const auto source = make_test_column_grid(1, 1, 4.0, 2.0);
    const auto target = make_test_column_grid(4, 4, 4.0, 2.0);
    const auto T = interpolate_temperature(source, target, Vector{315.0});
    const std::vector<double> I{2.0, 5.0, 2.0, 5.0};
    const auto next_I = interpolate_reaction_integrals(source, target, I, 2, -1);
    for (const auto& cell : target.cells) {
        EXPECT_DOUBLE_EQ(T[static_cast<Vector::Index>(cell.index)], 315.0);
    }
    for (std::size_t node = 0; node < next_I.size() / 2; ++node) {
        EXPECT_DOUBLE_EQ(next_I[node * 2], 2.0);
        EXPECT_DOUBLE_EQ(next_I[node * 2 + 1], 5.0);
    }
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
    Vector values(static_cast<Vector::Index>(source.cells.size()));
    for (const auto& cell : source.cells)
        values[static_cast<Vector::Index>(cell.index)] = exact(cell.center().x, cell.center().z);
    const auto transferred = interpolate_temperature(source, target, values);
    return area_weighted_error(target, {transferred.native().data(), target.cells.size()}, exact);
}

} // namespace

TEST(ColFieldTransfer, SmoothFieldConvergesUnderRefinement) {
    const double coarse = transfer_error(8, 4);
    const double fine = transfer_error(16, 8);
    EXPECT_GT(coarse / fine, 1.7) << coarse << ", " << fine;
}
