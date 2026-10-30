#include "basin/nonmatching_column_grid.hpp"
#include "discretization/nonmatching_column_flux.hpp"
#include "discretization/nonmatching_column_heat_system.hpp"

#include <gtest/gtest.h>
#include <map>

namespace {

BasinGeometry geometry() {
    const std::vector<double> x{0.0, 1.0, 2.0};
    return {.age_ma = 0.0,
            .x = x,
            .boundaries = {{0.0, 0.4, 0.0}, {0.8, 1.2, 1.0}, {2.0, 2.4, 2.2}},
            .layers = {{1, 0.0, 2.0}, {2, 0.0, 2.0}}};
}

} // namespace

TEST(NonmatchingColumnGrid, PreservesEveryLayerAreaWithMeanHorizontalBoundaries) {
    const BasinGeometry source = geometry();
    const std::vector<double> layout = {0.0, 1.0, 2.0};
    const double vertical_step = 0.55;
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout, vertical_step);

    std::map<int, double> area;
    for (const NonmatchingColumnCell& cell : grid.cells) {
        area[cell.layer_id] += cell.area();
        EXPECT_GT(cell.area(), 0.0);
    }
    EXPECT_NEAR(area[1], 1.7, 1.0e-14);
    EXPECT_NEAR(area[2], 2.4, 1.0e-14);
    EXPECT_NEAR(grid.area(), 4.1, 1.0e-14);
}

TEST(NonmatchingColumnGrid, MergeCreatesSubfacesWithoutCreatingCells) {
    const BasinGeometry source = geometry();
    const std::vector<double> layout = {0.0, 1.0, 2.0};
    const double vertical_step = 0.55;
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout, vertical_step);

    ASSERT_EQ(grid.columns.size(), 2U);
    const std::size_t cells_before_merge = grid.columns[0].size() + grid.columns[1].size();
    EXPECT_EQ(grid.cells.size(), cells_before_merge);
    EXPECT_GT(grid.vertical_subfaces.size(), 1U);
    double interface_length = 0.0;
    for (const NonmatchingVerticalSubface& face : grid.vertical_subfaces) {
        EXPECT_GT(face.length(), 0.0);
        interface_length += face.length();
    }
    const double common_top =
        std::max(grid.cells[grid.top_cells[0]].z_top, grid.cells[grid.top_cells[1]].z_top);
    const double common_bottom = std::min(grid.cells[grid.bottom_cells[0]].z_bottom,
                                          grid.cells[grid.bottom_cells[1]].z_bottom);
    EXPECT_NEAR(interface_length, common_bottom - common_top, 1.0e-14);
}

TEST(NonmatchingColumnGrid, AxisRetainsContourSamplesAndPhasesOnlyInsertedLines) {
    const BasinGeometry source = geometry();
    const auto layout = make_column_axis(source.x, 0.75);
    EXPECT_EQ(layout, (std::vector<double>{0.0, 0.5, 1.0, 1.5, 2.0}));

    const auto shifted = make_column_axis(source.x, 0.75, {}, 0.25);
    EXPECT_EQ(shifted, (std::vector<double>{0.0, 0.625, 1.0, 1.625, 2.0}));
}

TEST(NonmatchingColumnGrid, PreservesLayerAreasAndCoversEveryColumnInterfaceAcrossLayouts) {
    const BasinGeometry source = geometry();
    for (const double dx : {0.75, 0.35}) {
        for (const double dz : {0.55, 0.23}) {
            for (const double phase : {-0.25, 0.0, 0.45}) {
                SCOPED_TRACE(testing::Message()
                             << "dx=" << dx << " dz=" << dz << " phase=" << phase);
                const auto grid = make_nonmatching_column_grid(
                    source, make_column_axis(source.x, dx, {}, phase), dz);
                std::map<int, double> area;
                std::vector<double> coverage(grid.columns.size() - 1, 0.0);
                for (const auto& cell : grid.cells) {
                    ASSERT_GT(cell.area(), 0.0);
                    area[cell.layer_id] += cell.area();
                }
                EXPECT_NEAR(area[1], 1.7, 1.0e-12);
                EXPECT_NEAR(area[2], 2.4, 1.0e-12);
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

TEST(NonmatchingColumnGrid, RepresentsVerticalOuterBoundarySteps) {
    const std::vector<double> x{0.0, 1.0, 2.0};
    const BasinGeometry source{.age_ma = 0.0,
                               .x = x,
                               .boundaries = {{0.0, 0.4, 0.8}, {1.0, 1.4, 1.8}, {2.0, 2.4, 2.8}},
                               .layers = {{1, 0.0, 2.0}, {2, 0.0, 2.0}}};
    const std::vector<double> layout = {0.0, 1.0, 2.0};
    const double vertical_step = 0.55;
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout, vertical_step);

    ASSERT_EQ(grid.surface_vertical_faces.size(), 1U);
    EXPECT_NEAR(grid.surface_vertical_faces.front().length(), 0.4, 1.0e-14);
    EXPECT_DOUBLE_EQ(grid.surface_vertical_faces.front().x, 1.0);
    EXPECT_DOUBLE_EQ(grid.surface_vertical_faces.front().outward_x, 1.0);
    ASSERT_EQ(grid.basal_vertical_faces.size(), 1U);
    EXPECT_NEAR(grid.basal_vertical_faces.front().length(), 0.4, 1.0e-14);
    EXPECT_DOUBLE_EQ(grid.basal_vertical_faces.front().outward_x, -1.0);
}

TEST(NonmatchingColumnHeatSystem, ReproducesAffineFieldOnSurfaceStaircase) {
    const std::vector<double> x{0.0, 1.0, 2.0};
    const BasinGeometry source{.age_ma = 0.0,
                               .x = x,
                               .boundaries = {{0.0, 0.4, 0.8}, {3.0, 3.0, 3.0}},
                               .layers = {{1, 0.0, 2.0}}};
    const std::vector<double> layout = {0.0, 1.0, 2.0};
    const double vertical_step = 0.45;
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout, vertical_step);
    ASSERT_FALSE(grid.surface_vertical_faces.empty());
    ASSERT_TRUE(grid.basal_vertical_faces.empty());

    constexpr double conductivity_value = 2.5;
    constexpr double x_gradient = 3.0;
    constexpr double z_gradient = -1.25;
    const auto exact = [](const Point2D point) {
        return 300.0 + x_gradient * point.x + z_gradient * point.z;
    };
    const std::vector<double> conductivity(grid.cells.size(), conductivity_value);
    const std::vector<double> production(grid.cells.size(), 0.0);
    Vector temperature(static_cast<Vector::Index>(grid.cells.size()));
    for (const NonmatchingColumnCell& cell : grid.cells) {
        temperature[static_cast<Vector::Index>(cell.index)] = exact(cell.center());
    }
    NonmatchingColumnHeatSystem system(
        grid, conductivity, production,
        {.surface_temperature = exact,
         .basal_inward_flux = [](Point2D) { return conductivity_value * z_gradient; },
         .left_inward_flux = [](Point2D) { return -conductivity_value * x_gradient; },
         .right_inward_flux = [](Point2D) { return conductivity_value * x_gradient; }});

    for (const MultipointFluxScheme scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        system.set_flux_scheme(scheme);
        Vector residual;
        system.assemble_residual(temperature, residual);
        EXPECT_LT(residual.infinity_norm(), 2.0e-11);
        const NonmatchingColumnEnergyRates rates = system.energy_rates(temperature);
        EXPECT_NEAR(rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow, 0.0,
                    2.0e-11);
    }
}

TEST(NonmatchingColumnFlux, ReproducesAffineFluxOnMergedSubfaces) {
    const BasinGeometry source = geometry();
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
    const NonmatchingColumnFlux flux(grid);
    for (const MultipointFluxScheme scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        const NonmatchingColumnFluxes values =
            flux.compute(scheme, conductivity, temperature, boundary);
        ASSERT_EQ(values.vertical.size(), grid.vertical_subfaces.size());
        for (std::size_t i = 0; i < values.vertical.size(); ++i) {
            const double expected = -2.5 * x_gradient * grid.vertical_subfaces[i].length();
            EXPECT_NEAR(values.vertical[i].left_expression.value(temperature), expected, 2.0e-12);
            EXPECT_NEAR(values.vertical[i].right_expression.value(temperature), expected, 2.0e-12);
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
    const NonmatchingColumnFlux flux(grid);
    for (const MultipointFluxScheme scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        const NonmatchingColumnFluxes values =
            flux.compute(scheme, conductivity, temperature, boundary);
        ASSERT_EQ(values.vertical.size(), 1U);
        EXPECT_NEAR(values.vertical[0].left_expression.value(temperature), -flux_scale, 2.0e-11);
        EXPECT_NEAR(values.vertical[0].right_expression.value(temperature), -flux_scale, 2.0e-11);
    }
}

TEST(NonmatchingColumnHeatSystem, LinearMatrixMatchesResidual) {
    const BasinGeometry source = geometry();
    const auto layout = make_column_axis(source.x, 0.55);
    const double vertical_step = 0.55;
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout, vertical_step);
    const std::vector<double> conductivity(grid.cells.size(), 2.5);
    const std::vector<double> production(grid.cells.size(), 0.01);
    NonmatchingColumnHeatSystem system(
        grid, conductivity, production,
        {.surface_temperature = [](const Point2D point) { return 285.0 + point.x; },
         .basal_inward_flux = [](Point2D) { return 0.08; }});
    Vector zero(static_cast<Vector::Index>(grid.cells.size()));
    zero.set_zero();
    Vector value(static_cast<Vector::Index>(grid.cells.size()));
    for (Vector::Index i = 0; i < value.size(); ++i) {
        value[i] = 290.0 + static_cast<double>(i);
    }
    Vector zero_residual;
    Vector residual;
    system.assemble_residual(zero, zero_residual);
    system.assemble_residual(value, residual);
    SparseMatrix matrix(value.size(), value.size(), SparseStorageOrder::csc);
    system.assemble_matrix(value, matrix);
    const Eigen::VectorXd predicted = matrix.native_csc() * value.native();
    EXPECT_LT((residual.native() - zero_residual.native() - predicted).norm(), 1.0e-10);
}
