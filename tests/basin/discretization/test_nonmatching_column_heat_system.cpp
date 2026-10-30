#include "basin/discretization/nonmatching_column_heat_system.hpp"
#include "basin/mesh/column_test_utils.hpp"
#include "basin/postprocessing/energy_rates.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(NonmatchingColumnHeatSystem, ReproducesAffineFieldOnSurfaceStaircase) {
    const std::vector<double> x{0.0, 1.0, 2.0};
    const BasinGeometry source{
        .x = x, .boundaries = {{0.0, 0.4, 0.8}, {3.0, 3.0, 3.0}}, .layers = {{1, 0.0, 2.0}}};
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
    const NonmatchingColumnBoundaryData boundary_data{
        .surface_temperature = exact,
        .basal_inward_flux = [](Point2D) { return conductivity_value * z_gradient; },
        .left_inward_flux = [](Point2D) { return -conductivity_value * x_gradient; },
        .right_inward_flux = [](Point2D) { return conductivity_value * x_gradient; }};
    NonmatchingColumnHeatSystem system(grid, conductivity, production, boundary_data);

    for (const MultipointFluxScheme scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        system.set_flux_scheme(scheme);
        Vector residual;
        system.assemble_residual(temperature, residual);
        EXPECT_LT(residual.infinity_norm(), 2.0e-11);
        const auto rates =
            nonmatching_column_energy_rates(grid, {temperature.native().data(), grid.cells.size()},
                                            conductivity, production, boundary_data);
        EXPECT_NEAR(rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow, 0.0,
                    2.0e-11);
    }
}

TEST(NonmatchingColumnHeatSystem, PicardMatrixMatchesResidualForBothSchemes) {
    const BasinGeometry source = make_test_layered_geometry();
    const auto layout = make_column_axis(source.x, 0.55);
    const double vertical_step = 0.55;
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout, vertical_step);
    const std::vector<double> conductivity(grid.cells.size(), 2.5);
    const std::vector<double> production(grid.cells.size(), 0.01);
    for (const MultipointFluxScheme scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        // Homogeneous boundary data make the constant term independent of NMPFA weights.
        NonmatchingColumnHeatSystem system(
            grid, conductivity, production,
            {.surface_temperature =
                 [scheme](const Point2D point) {
                     return scheme == MultipointFluxScheme::linear ? 285.0 + point.x : 0.0;
                 },
             .basal_inward_flux =
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
        Vector residual;
        system.assemble_residual(zero, zero_residual);
        system.assemble_residual(value, residual);
        SparseMatrix matrix(value.size(), value.size(), SparseStorageOrder::csc);
        system.assemble_matrix(value, matrix);
        const Eigen::VectorXd predicted = matrix.native_csc() * value.native();
        EXPECT_LT((residual.native() - zero_residual.native() - predicted).norm(), 1.0e-10);
    }
}

TEST(NonmatchingColumnHeatSystem, UsesThermophysicalFieldsAndConservesEnergy) {
    const auto grid = make_test_column_grid(3, 2, 3.0, 2.0);
    const LithotypeThermophysicalProperties lithotype{.code = 1003,
                                                      .solid_density = 2720.0,
                                                      .conductivity_at_20_celsius = 3.95,
                                                      .specific_heat_at_20_celsius = 854.1,
                                                      .solid_heat_production = 7.034e-7};
    constexpr FluidThermophysicalProperties fluid{
        .thermal_conductivity = 0.7, .density = 1040.0, .specific_heat = 4184.0};
    const std::size_t cell_count = grid.cells.size();
    std::vector<double> temperatures(cell_count, 373.15);
    std::vector<double> porosities(cell_count, 0.2);
    std::vector<const LithotypeThermophysicalProperties*> lithotypes(cell_count, &lithotype);
    std::vector<double> conductivities(cell_count);
    std::vector<double> heat_capacities(cell_count);
    std::vector<double> heat_productions(cell_count);
    evaluate_thermophysical_property_fields(temperatures, porosities, lithotypes, fluid,
                                            conductivities, heat_capacities, heat_productions);

    const NonmatchingColumnBoundaryData boundary_data{
        .surface_temperature = [](Point2D) { return 300.0; },
        .basal_inward_flux = [](Point2D) { return 0.06; }};
    const NonmatchingColumnHeatSystem system(grid, conductivities, heat_productions, boundary_data,
                                             heat_capacities);
    Vector temperature(static_cast<Vector::Index>(cell_count));
    temperature.set_constant(373.15);
    Vector temperature_derivative(temperature.size());
    temperature_derivative.set_zero();
    Vector residual;
    system.assemble_residual(0.0, temperature, temperature_derivative, residual);
    SparseMatrix matrix(temperature.size(), temperature.size(), SparseStorageOrder::csc);
    system.assemble_matrix(NonlinearMethod::picard, 0.0, temperature, temperature_derivative, 1.0,
                           matrix);

    EXPECT_TRUE(residual.all_finite());
    EXPECT_TRUE(matrix.is_compressed());
    EXPECT_GT(matrix.nonzero_count(), temperature.size());
    const auto rates =
        nonmatching_column_energy_rates(grid, {temperature.native().data(), cell_count},
                                        conductivities, heat_productions, boundary_data);
    double expected_production = 0.0;
    for (const auto& cell : grid.cells) {
        expected_production += cell.area() * heat_productions[cell.index];
    }
    EXPECT_NEAR(rates.heat_production, expected_production, 1.0e-12);
    EXPECT_NEAR(residual.native().sum(),
                rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow -
                    rates.heat_production,
                1.0e-9);
}
