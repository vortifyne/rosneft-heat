#include "basin/nonmatching_column_grid.hpp"
#include "discretization/nonmatching_column_flux.hpp"
#include "discretization/nonmatching_column_heat_system.hpp"
#include "discretization/nonmatching_column_hfv_system.hpp"
#include "linear/umfpack_linear_solver.hpp"

#include <gtest/gtest.h>
#include <map>

namespace {

ScalarProfile profile(std::vector<double> x, std::vector<double> values) {
    return {.x = std::move(x), .values = std::move(values)};
}

BasinGeometry geometry() {
    const std::vector<double> x{0.0, 1.0, 2.0};
    return {.age_ma = 0.0,
            .boundaries = {{profile(x, {0.0, 0.4, 0.0})},
                           {profile(x, {0.8, 1.2, 1.0})},
                           {profile(x, {2.0, 2.4, 2.2})}},
            .layers = {{1, 0, 1, 0.0, 2.0}, {2, 1, 2, 0.0, 2.0}}};
}

} // namespace

TEST(NonmatchingColumnGrid, PreservesEveryLayerAreaWithMeanHorizontalBoundaries) {
    const BasinGeometry source = geometry();
    const LayeredMeshLayout layout{.x = {0.0, 1.0, 2.0}, .cell_size = 0.55};
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout);

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
    const LayeredMeshLayout layout{.x = {0.0, 1.0, 2.0}, .cell_size = 0.55};
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout);

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

TEST(NonmatchingColumnGrid, LayoutRetainsContourSamplesAndPhasesOnlyInsertedLines) {
    const BasinGeometry source = geometry();
    const LayeredMeshLayout layout = make_nonmatching_column_layout(source, 0.75);
    EXPECT_EQ(layout.x, (std::vector<double>{0.0, 0.5, 1.0, 1.5, 2.0}));

    const LayeredMeshLayout shifted = make_nonmatching_column_layout(source, 0.75, 0.4, 0.25);
    EXPECT_EQ(shifted.x, (std::vector<double>{0.0, 0.625, 1.0, 1.625, 2.0}));
    EXPECT_DOUBLE_EQ(shifted.cell_size, 0.4);
}

TEST(NonmatchingColumnGrid, RepresentsVerticalOuterBoundarySteps) {
    const std::vector<double> x{0.0, 1.0, 2.0};
    const BasinGeometry source{.age_ma = 0.0,
                               .boundaries = {{profile(x, {0.0, 0.4, 0.8})},
                                              {profile(x, {1.0, 1.4, 1.8})},
                                              {profile(x, {2.0, 2.4, 2.8})}},
                               .layers = {{1, 0, 1, 0.0, 2.0}, {2, 1, 2, 0.0, 2.0}}};
    const LayeredMeshLayout layout{.x = {0.0, 1.0, 2.0}, .cell_size = 0.55};
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout);

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
    const BasinGeometry source{
        .age_ma = 0.0,
        .boundaries = {{profile(x, {0.0, 0.4, 0.8})}, {profile(x, {3.0, 3.0, 3.0})}},
        .layers = {{1, 0, 1, 0.0, 2.0}}};
    const LayeredMeshLayout layout{.x = {0.0, 1.0, 2.0}, .cell_size = 0.45};
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout);
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
         {MultipointFluxScheme::linear, MultipointFluxScheme::linear_equal_weight,
          MultipointFluxScheme::nonlinear_conservative}) {
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
    const LayeredMeshLayout layout{.x = {0.0, 1.0, 2.0}, .cell_size = 0.55};
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout);
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
         {MultipointFluxScheme::linear, MultipointFluxScheme::linear_equal_weight,
          MultipointFluxScheme::nonlinear_conservative, MultipointFluxScheme::nonlinear_dmp}) {
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
         {MultipointFluxScheme::linear, MultipointFluxScheme::linear_equal_weight,
          MultipointFluxScheme::nonlinear_conservative, MultipointFluxScheme::nonlinear_dmp}) {
        const NonmatchingColumnFluxes values =
            flux.compute(scheme, conductivity, temperature, boundary);
        ASSERT_EQ(values.vertical.size(), 1U);
        EXPECT_NEAR(values.vertical[0].left_expression.value(temperature), -flux_scale, 2.0e-11);
        EXPECT_NEAR(values.vertical[0].right_expression.value(temperature), -flux_scale, 2.0e-11);
    }
}

TEST(NonmatchingColumnHeatSystem, LinearMatrixMatchesResidualAndDmpMatrixIsMonotone) {
    const BasinGeometry source = geometry();
    const LayeredMeshLayout layout = make_nonmatching_column_layout(source, 0.55);
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout);
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

    system.set_flux_scheme(MultipointFluxScheme::nonlinear_dmp);
    system.assemble_matrix(value, matrix);
    const auto& native = matrix.native_csc();
    for (Eigen::Index column = 0; column < native.outerSize(); ++column) {
        for (SparseMatrix::CscNativeType::InnerIterator entry(native, column); entry; ++entry) {
            if (entry.row() != entry.col()) {
                EXPECT_LE(entry.value(), 1.0e-12);
            }
        }
    }
}

TEST(NonmatchingColumnHfvSystem, ReproducesAffineFieldOnNonmatchingSurfaceStaircase) {
    BasinGeometry source = geometry();
    source.boundaries.back().depth.values = {2.4, 2.4, 2.4};
    const LayeredMeshLayout layout{.x = {0.0, 0.6, 1.0, 2.0}, .cell_size = 0.43};
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout);
    constexpr double conductivity_value = 2.5;
    constexpr double x_gradient = 3.0;
    constexpr double z_gradient = 1.25;
    const auto exact = [](const Point2D point) {
        return 300.0 + x_gradient * point.x + z_gradient * point.z;
    };
    const std::vector<double> conductivity(grid.cells.size(), conductivity_value);
    const std::vector<double> production(grid.cells.size(), 0.0);
    NonmatchingColumnHfvSystem system(
        grid, conductivity, production,
        {.surface_temperature = exact,
         .basal_inward_flux = [](Point2D) { return conductivity_value * z_gradient; },
         .left_inward_flux = [](Point2D) { return -conductivity_value * x_gradient; },
         .right_inward_flux = [](Point2D) { return conductivity_value * x_gradient; }});
    const Vector interpolated = system.interpolate(exact);
    Vector residual;
    system.assemble_residual(interpolated, residual);
    EXPECT_LT(residual.infinity_norm(), 2.0e-11);

    SparseMatrix matrix;
    Vector rhs;
    system.assemble_system(interpolated, matrix, rhs);
    const Eigen::MatrixXd dense(matrix.native_csc());
    EXPECT_LT((dense - dense.transpose()).norm(), 2.0e-12);
    Vector solved;
    const LinearSolveResult solve = UmfpackLinearSolver{}.solve(
        matrix, rhs, solved,
        {.relative_tolerance = 1.0e-12, .absolute_tolerance = 1.0e-12, .max_iterations = 1});
    ASSERT_EQ(solve.status, LinearSolveStatus::converged);
    EXPECT_LT((solved - interpolated).infinity_norm(), 5.0e-11);

    const NonmatchingHfvEnergyRates rates = system.energy_rates(solved);
    EXPECT_NEAR(rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow, 0.0, 2.0e-11);
    const NonmatchingHfvDiagnostics diagnostics = system.diagnostics(solved);
    EXPECT_GT(diagnostics.face_unknowns, 0U);
    EXPECT_LT(diagnostics.internal_conservation_defect_linf, 2.0e-11);
}

TEST(NonmatchingColumnHfvSystem, ReproducesPiecewiseAffineFieldAcrossConductivityJump) {
    NonmatchingColumnGrid grid{.x = {0.0, 1.0},
                               .cells = {{.index = 0,
                                          .column = 0,
                                          .layer_id = 1,
                                          .x_left = 0.0,
                                          .x_right = 1.0,
                                          .z_top = 0.0,
                                          .z_bottom = 1.0,
                                          .upward = std::nullopt,
                                          .downward = 1},
                                         {.index = 1,
                                          .column = 0,
                                          .layer_id = 2,
                                          .x_left = 0.0,
                                          .x_right = 1.0,
                                          .z_top = 1.0,
                                          .z_bottom = 2.0,
                                          .upward = 0,
                                          .downward = std::nullopt}},
                               .columns = {{0, 1}},
                               .horizontal_faces = {{0, 1, 1.0, 1.0}},
                               .vertical_subfaces = {},
                               .surface_vertical_faces = {},
                               .basal_vertical_faces = {},
                               .top_cells = {0},
                               .bottom_cells = {1}};
    const std::vector<double> conductivity{1.0, 100.0};
    const std::vector<double> production{0.0, 0.0};
    constexpr double inward_flux = 5.0;
    const auto exact = [](const Point2D point) {
        return point.z <= 1.0 ? 300.0 + inward_flux * point.z
                              : 305.0 + inward_flux / 100.0 * (point.z - 1.0);
    };
    NonmatchingColumnHfvSystem system(
        grid, conductivity, production,
        {.surface_temperature = exact, .basal_inward_flux = [](Point2D) { return inward_flux; }});
    const Vector interpolated = system.interpolate(exact);
    Vector residual;
    system.assemble_residual(interpolated, residual);
    EXPECT_LT(residual.infinity_norm(), 2.0e-11);

    SparseMatrix matrix;
    Vector rhs;
    system.assemble_system(interpolated, matrix, rhs);
    Vector solved;
    const LinearSolveResult solve = UmfpackLinearSolver{}.solve(
        matrix, rhs, solved,
        {.relative_tolerance = 1.0e-12, .absolute_tolerance = 1.0e-12, .max_iterations = 1});
    ASSERT_EQ(solve.status, LinearSolveStatus::converged);
    EXPECT_LT((solved - interpolated).infinity_norm(), 5.0e-11);
}
