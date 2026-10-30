#include "basin/discretization/col_heat_system.hpp"
#include "basin/mesh/column_test_utils.hpp"
#include "basin/postprocessing/energy_rates.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(ColHeatSystem, ReproducesAffineFieldOnSurfaceStaircase) {
    const std::vector<double> x{0.0, 1.0, 2.0};
    const BasinGeometry source{
        .x = x, .z = {{0.0, 0.4, 0.8}, {3.0, 3.0, 3.0}}, .layers = {{1, 0.0, 2.0}}};
    const std::vector<double> layout = {0.0, 1.0, 2.0};
    const double dz = 0.45;
    const ColGrid grid = make_col_grid(source, layout, dz);
    ASSERT_FALSE(grid.surface_vertical_faces.empty());
    ASSERT_TRUE(grid.basal_vertical_faces.empty());

    constexpr double conductivity_value = 2.5;
    constexpr double x_gradient = 3.0;
    constexpr double z_gradient = -1.25;
    const auto exact = [](const Point2D point) {
        return 300.0 + x_gradient * point.x + z_gradient * point.z;
    };
    const std::vector<double> lambda(grid.cells.size(), conductivity_value);
    const std::vector<double> production(grid.cells.size(), 0.0);
    Vector T(static_cast<Vector::Index>(grid.cells.size()));
    for (const ColCell& cell : grid.cells) {
        T[static_cast<Vector::Index>(cell.index)] = exact(cell.center());
    }
    const HeatBoundaryData bc{.T_surf = exact,
                              .q_basal = [](Point2D) { return conductivity_value * z_gradient; },
                              .q_left = [](Point2D) { return -conductivity_value * x_gradient; },
                              .q_right = [](Point2D) { return conductivity_value * x_gradient; }};
    ColHeatSystem system(grid, lambda, production, bc);

    for (const MultipointFluxScheme scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        system.set_flux_scheme(scheme);
        Vector r;
        system.assemble_residual(T, r);
        EXPECT_LT(r.infinity_norm(), 2.0e-11);
        const auto rates =
            energy_rates(grid, {T.native().data(), grid.cells.size()}, lambda, production, bc);
        EXPECT_NEAR(rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow, 0.0,
                    2.0e-11);
    }
}

TEST(ColHeatSystem, PicardMatrixMatchesResidualForBothSchemes) {
    const BasinGeometry source = make_test_layered_geometry();
    const auto layout = make_column_axis(source.x, 0.55);
    const double dz = 0.55;
    const ColGrid grid = make_col_grid(source, layout, dz);
    const std::vector<double> lambda(grid.cells.size(), 2.5);
    const std::vector<double> production(grid.cells.size(), 0.01);
    for (const MultipointFluxScheme scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        // Homogeneous boundary data make the constant term independent of NMPFA weights.
        ColHeatSystem system(grid, lambda, production,
                             {.T_surf =
                                  [scheme](const Point2D point) {
                                      return scheme == MultipointFluxScheme::linear
                                                 ? 285.0 + point.x
                                                 : 0.0;
                                  },
                              .q_basal =
                                  [scheme](Point2D) {
                                      return scheme == MultipointFluxScheme::linear ? 0.08 : 0.0;
                                  }});
        system.set_flux_scheme(scheme);
        Vector zero(static_cast<Vector::Index>(grid.cells.size()));
        zero.set_zero();
        Vector value(static_cast<Vector::Index>(grid.cells.size()));
        for (Vector::Index i = 0; i < value.size(); ++i) {
            value[i] = 290.0 + static_cast<double>(i);
        }
        Vector zero_residual;
        Vector r;
        system.assemble_residual(zero, zero_residual);
        system.assemble_residual(value, r);
        SparseMatrix M(value.size(), value.size(), SparseStorageOrder::csc);
        system.assemble_matrix(value, M);
        const Eigen::VectorXd predicted = M.native_csc() * value.native();
        EXPECT_LT((r.native() - zero_residual.native() - predicted).norm(), 1.0e-10);
    }
}

TEST(ColHeatSystem, UsesThermalFieldsAndConservesEnergy) {
    const auto grid = make_test_column_grid(3, 2, 3.0, 2.0);
    const LithotypeProperties lithotype{
        .code = 1003, .rho_s = 2720.0, .lambda20 = 3.95, .c20 = 854.1, .A_s = 7.034e-7};
    constexpr FluidProperties fluid{.lambda = 0.7, .rho = 1040.0, .c = 4184.0};
    const std::size_t cell_count = grid.cells.size();
    std::vector<double> T_values(cell_count, 373.15);
    std::vector<double> phi(cell_count, 0.2);
    std::vector<const LithotypeProperties*> lithotypes(cell_count, &lithotype);
    std::vector<double> lambda(cell_count);
    std::vector<double> rho_c(cell_count);
    std::vector<double> A(cell_count);
    update_thermal_fields(T_values, phi, lithotypes, fluid, lambda, rho_c, A);

    const HeatBoundaryData bc{.T_surf = [](Point2D) { return 300.0; },
                              .q_basal = [](Point2D) { return 0.06; }};
    const ColHeatSystem system(grid, lambda, A, bc, rho_c);
    Vector T(static_cast<Vector::Index>(cell_count));
    T.set_constant(373.15);
    Vector dT_dt(T.size());
    dT_dt.set_zero();
    Vector r;
    system.assemble_residual(0.0, T, dT_dt, r);
    SparseMatrix M(T.size(), T.size(), SparseStorageOrder::csc);
    system.assemble_matrix(NonlinearMethod::picard, 0.0, T, dT_dt, 1.0, M);

    EXPECT_TRUE(r.all_finite());
    EXPECT_TRUE(M.is_compressed());
    EXPECT_GT(M.nonzero_count(), T.size());
    const auto rates = energy_rates(grid, {T.native().data(), cell_count}, lambda, A, bc);
    double expected_production = 0.0;
    for (const auto& cell : grid.cells) {
        expected_production += cell.area() * A[cell.index];
    }
    EXPECT_NEAR(rates.source, expected_production, 1.0e-12);
    EXPECT_NEAR(r.native().sum(),
                rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow - rates.source,
                1.0e-9);
}
