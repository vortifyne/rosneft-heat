#include "basin/discretization/nonmatching_column_flux.hpp"
#include "basin/mesh/column_test_utils.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(NonmatchingColumnFlux, ReproducesAffineFluxOnMergedSubfaces) {
    const BasinGeometry source = make_test_layered_geometry();
    const std::vector<double> layout = {0.0, 1.0, 2.0};
    const double vertical_step = 0.55;
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout, vertical_step);
    const std::vector<double> conductivity(grid.cells.size(), 2.5);
    std::vector<double> temperature(grid.cells.size());
    constexpr double x_gradient = 3.0;
    constexpr double z_gradient = -1.25;
    for (const NonmatchingColumnCell& cell : grid.cells) {
        const Point2D center = cell.center();
        temperature[cell.index] = 300.0 + x_gradient * center.x + z_gradient * center.z;
    }
    const NonmatchingColumnBoundaryData boundary{
        .surface_temperature =
            [](const Point2D point) { return 300.0 + x_gradient * point.x + z_gradient * point.z; },
        .basal_inward_flux = [](Point2D) { return 2.5 * z_gradient; }};
    NonmatchingColumnFluxStencil stencil;
    for (const MultipointFluxScheme scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        for (const auto& face : grid.vertical_subfaces) {
            compute_internal_flux(grid, face, scheme, conductivity, temperature, boundary, stencil);
            ASSERT_LE(stencil.size, 4U);
            const double expected = -2.5 * x_gradient * face.length();
            EXPECT_NEAR(evaluate_flux(stencil, temperature), expected, 2.0e-12);
        }
    }
}

TEST(NonmatchingColumnFlux, UsesHarmonicNormalFluxAcrossMaterialJump) {
    NonmatchingColumnGrid grid{.x = {0.0, 1.0, 2.0},
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
    const std::vector<double> conductivity{1.0, 1000.0};
    constexpr double flux_scale = 5.0;
    const std::vector<double> temperature{300.0 + 0.5 * flux_scale,
                                          300.0 + flux_scale + 0.5 * flux_scale / 1000.0};
    const NonmatchingColumnBoundaryData boundary{
        .surface_temperature = [](Point2D) { return 300.0; },
        .basal_inward_flux = [](Point2D) { return 0.0; }};
    NonmatchingColumnFluxStencil stencil;
    for (const MultipointFluxScheme scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        ASSERT_EQ(grid.vertical_subfaces.size(), 1U);
        compute_internal_flux(grid, grid.vertical_subfaces.front(), scheme, conductivity,
                              temperature, boundary, stencil);
        EXPECT_NEAR(evaluate_flux(stencil, temperature), -flux_scale, 2.0e-11);
    }
}
