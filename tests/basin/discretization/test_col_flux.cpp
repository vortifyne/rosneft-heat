#include "basin/discretization/col_flux.hpp"
#include "basin/mesh/col_field_transfer.hpp"
#include "basin/mesh/column_test_utils.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(ColFlux, ReproducesAffineFluxOnMergedSubfaces) {
    const BasinGeometry source = make_test_layered_geometry();
    const std::vector<double> layout = {0.0, 1.0, 2.0};
    const double dz = 0.55;
    const ColGrid grid = make_col_grid(source, layout, dz);
    const std::vector<double> lambda(grid.cells.size(), 2.5);
    std::vector<double> T(grid.cells.size());
    constexpr double x_gradient = 3.0;
    constexpr double z_gradient = -1.25;
    for (const ColCell& cell : grid.cells) {
        const Point2D center = cell.center();
        T[cell.index] = 300.0 + x_gradient * center.x + z_gradient * center.z;
    }
    const HeatBoundaryData boundary{
        .T_surf =
            [](const Point2D point) { return 300.0 + x_gradient * point.x + z_gradient * point.z; },
        .q_basal = [](Point2D) { return 2.5 * z_gradient; }};
    FluxStencil stencil;
    for (const MultipointFluxScheme scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        for (const auto& face : grid.vertical_subfaces) {
            compute_internal_flux(grid, face, scheme, lambda, T, boundary, stencil);
            ASSERT_LE(stencil.size, 4U);
            const double expected = -2.5 * x_gradient * face.length();
            EXPECT_NEAR(evaluate_flux(stencil, T), expected, 2.0e-12);
        }
    }
}

TEST(ColFlux, UsesHarmonicNormalFluxAcrossMaterialJump) {
    ColGrid grid{.x = {0.0, 1.0, 2.0},
                 .cells = {{.index = 0,
                            .column = 0,
                            .layer_id = 1,
                            .x_left = 0.0,
                            .x_right = 1.0,
                            .z_top = 0.0,
                            .z_bottom = 1.0,
                            .upward = std::nullopt,
                            .downward = std::nullopt},
                           {.index = 1,
                            .column = 1,
                            .layer_id = 2,
                            .x_left = 1.0,
                            .x_right = 2.0,
                            .z_top = 0.0,
                            .z_bottom = 1.0,
                            .upward = std::nullopt,
                            .downward = std::nullopt}},
                 .columns = {{0}, {1}},
                 .horizontal_faces = {},
                 .vertical_subfaces = {{0, 1, 1.0, 0.0, 1.0}},
                 .surface_vertical_faces = {},
                 .basal_vertical_faces = {},
                 .top_cells = {0, 1},
                 .bottom_cells = {0, 1}};
    const std::vector<double> lambda{1.0, 1000.0};
    constexpr double flux_scale = 5.0;
    const std::vector<double> T{300.0 + 0.5 * flux_scale,
                                300.0 + flux_scale + 0.5 * flux_scale / 1000.0};
    const HeatBoundaryData boundary{.T_surf = [](Point2D) { return 300.0; },
                                    .q_basal = [](Point2D) { return 0.0; }};
    FluxStencil stencil;
    for (const MultipointFluxScheme scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        ASSERT_EQ(grid.vertical_subfaces.size(), 1U);
        compute_internal_flux(grid, grid.vertical_subfaces.front(), scheme, lambda, T, boundary,
                              stencil);
        EXPECT_NEAR(evaluate_flux(stencil, T), -flux_scale, 2.0e-11);
    }
}

TEST(ColFlux, ReconstructsLinearTemperatureAtHorizontalNodes) {
    const auto grid = make_test_column_grid(2, 3, 4.0, 2.0);
    Vector T(static_cast<Vector::Index>(grid.cells.size()));
    for (const auto& cell : grid.cells)
        T[static_cast<Vector::Index>(cell.index)] = 300.0 + 5.0 * cell.center().z;
    const auto z = horizontal_node_depths(grid);
    std::vector<double> nodes(z.size());
    const std::vector<double> lambda(grid.cells.size(), 2.0);
    interpolate_horizontal_temperature(
        grid, {T.native().data(), grid.cells.size()}, lambda,
        {.T_surf = [](Point2D) { return 300.0; }, .q_basal = [](Point2D) { return 10.0; }}, nodes);
    for (std::size_t i = 0; i < nodes.size(); ++i)
        EXPECT_NEAR(nodes[i], 300.0 + 5.0 * z[i], 1e-12);
}

TEST(ColFlux, ReconstructsTemperatureAcrossAConductivityJumpAndAtBoundaries) {
    const BasinGeometry geometry{.x = {0.0, 1.0},
                                 .z = {{0.0, 0.0}, {0.4, 0.4}, {2.0, 2.0}},
                                 .layers = {{1, 0.0, 1.0}, {2, 0.0, 1.0}}};
    const auto grid = make_col_grid(geometry, geometry.x, 2.0);
    const std::vector<double> T{300.4, 301.2};
    const std::vector<double> lambda{1.0, 4.0};
    std::vector<double> nodes(3);
    interpolate_horizontal_temperature(
        grid, T, lambda,
        {.T_surf = [](Point2D) { return 300.0; }, .q_basal = [](Point2D) { return 2.0; }}, nodes);
    EXPECT_DOUBLE_EQ(nodes.front(), 300.0);
    EXPECT_NEAR(nodes[1], 300.8, 1e-12);
    EXPECT_NEAR(nodes.back(), 301.6, 1e-12);
    EXPECT_NEAR(lambda[0] * (nodes[1] - T[0]) / 0.2, lambda[1] * (T[1] - nodes[1]) / 0.8, 1e-12);
}
