#include "basin/geometry/basin_geometry.hpp"
#include "basin/mesh/col_field_transfer.hpp"
#include "basin/mesh/column_test_utils.hpp"
#include "basin/physics/easy_ro.hpp"

#include <algorithm>
#include <array>
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
    std::vector<double> old_I(old_grid.maturity_node_count * reactions);
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
    ASSERT_EQ(I.size(), 4 * reactions);
    EXPECT_NEAR(T[0], 303.75, 1.0e-12); // z=0.25: linear extrapolation.
    EXPECT_NEAR(T[1], 310.75, 1.0e-12); // z=1.25: bilinear.
    for (std::size_t reaction = 0; reaction < reactions; ++reaction) {
        EXPECT_DOUBLE_EQ(I[reaction], 0.0);
        EXPECT_DOUBLE_EQ(I[reactions + reaction], 0.0);
        EXPECT_NEAR(I[2 * reactions + reaction], 302.0 * static_cast<double>(reaction + 1),
                    1.0e-12);
        EXPECT_NEAR(I[3 * reactions + reaction], 316.0 * static_cast<double>(reaction + 1),
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
    const std::vector<double> I(source.maturity_node_count, 315.0);
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
    std::vector<double> I(source.maturity_node_count);
    for (const auto& cell : source.cells) {
        const auto history = [&](double z) {
            return cell.layer_id == 7 ? 10.0 + 20.0 * z : 40.0 + 85.0 * (z - 1.0);
        };
        I[cell.maturity_top] = history(cell.z_top);
        I[cell.maturity_top + 1] = history(cell.z_bottom);
    }
    const auto next = interpolate_reaction_integrals(source, target, I, 1, -1);
    for (const auto& cell : target.cells) {
        const auto history = [&](double z) {
            if (cell.layer_id == 7)
                return 10.0 + 20.0 * std::min(z, 1.0);
            return z == 1.5 ? 40.0 : 40.0 + 85.0 * (z - 1.0);
        };
        EXPECT_NEAR(next[cell.maturity_top], history(cell.z_top), 1e-12);
        EXPECT_NEAR(next[cell.maturity_top + 1], history(cell.z_bottom), 1e-12);
    }
}

TEST(ColFieldTransfer, SurvivingSidesKeepIndependentHistoriesAfterLayerDisappears) {
    const BasinGeometry geometry{.x = {0.0, 1.0},
                                 .z = {{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}},
                                 .layers = {{7, 0.0, 2.0}, {5, 0.0, 2.0}, {2, 0.0, 2.0}}};
    const auto source = make_col_grid(geometry, geometry.x, 2.0);
    auto collapsed = geometry;
    collapsed.z[1] = collapsed.z[2] = {1.5, 1.5};
    const auto target = make_col_grid(collapsed, collapsed.x, 2.0);
    ASSERT_EQ(source.cells.size(), 3);
    ASSERT_EQ(target.cells.size(), 2);
    const std::array histories{1.0, 2.0, 10.0, 20.0, 3.0, 4.0};
    ASSERT_EQ(source.maturity_node_count, histories.size());
    auto next = interpolate_reaction_integrals(source, target, histories, 1, -1);
    ASSERT_EQ(next, (std::vector{1.0, 2.0, 3.0, 4.0}));

    const EasyRoModel model({.A = 1.0e13, .Rg = 8.314, .E = {142351.2}, .f = {0.03}});
    const std::array T{450.0, 450.0, 450.0, 450.0};
    model.advance(T, T, 0.0, 1.0e6, std::array{0.0, 0.0, 0.0, 0.0}, next);
    EXPECT_GT(next[1], 2.0);
    EXPECT_GT(next[2], 3.0);
    EXPECT_NEAR(next[2] - next[1], 1.0, 1e-12);
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
