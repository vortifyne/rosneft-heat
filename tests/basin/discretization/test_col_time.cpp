#include "basin/discretization/col_heat_system.hpp"
#include "basin/geometry/basin_geometry.hpp"
#include "basin/mesh/column_test_utils.hpp"
#include "numerics/time/time_integrator.hpp"
#include "support/petsc_test_session.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <optional>
#include <vector>

namespace {

ColGrid single_cell_grid(double height) {
    ColGrid grid;
    grid.x = {0.0, 1.0};
    grid.cells = {{.index = 0,
                   .column = 0,
                   .layer_id = 1,
                   .x_left = 0.0,
                   .x_right = 1.0,
                   .z_top = 0.0,
                   .z_bottom = height,
                   .upward = std::nullopt,
                   .downward = std::nullopt}};
    grid.columns = {{0}};
    grid.top_cells = {0};
    grid.bottom_cells = {0};
    return grid;
}

HeatBoundaryData boundary(double T_surf) {
    return {.T_surf = [T_surf](Point2D) { return T_surf; }, .q_basal = [](Point2D) { return 0.0; }};
}

constexpr NonlinearSolveRequest kPicard{
    .method = NonlinearMethod::picard,
    .rtol = 1.0e-12,
    .atol = 1.0e-12,
    .maxiter = 20,
};

constexpr LinearSolveRequest kDirect{
    .rtol = 1.0e-12,
    .atol = 1.0e-12,
    .maxiter = 50,
};

double integrate_decay(double timestep) {
    const ColGrid grid = single_cell_grid(1.0);
    const std::vector<double> lambda{1.0};
    const std::vector<double> source{0.0};
    const std::vector<double> capacity{1.0};
    ColHeatSystem system(grid, lambda, source, boundary(0.0), capacity);
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, Vector{1.0});
    integrator.set_timestep(timestep);
    const TimeIntegrationResult result = integrator.advance_to(system, 1.0, kPicard, kDirect);
    EXPECT_TRUE(result.completed());
    return integrator.current_snapshot().u[0];
}

double integrate_moving_quadratic(double timestep) {
    double T = 0.0;
    const int steps = static_cast<int>(std::round(1.0 / timestep));
    for (int step = 1; step <= steps; ++step) {
        const double time = static_cast<double>(step) * timestep;
        const ColGrid grid = single_cell_grid(1.0 + 0.5 * time);
        const std::vector<double> lambda{1.0};
        const std::vector<double> source{2.0 * time};
        const std::vector<double> capacity{1.0};
        ColHeatSystem system(grid, lambda, source, boundary(time * time), capacity);
        TimeIntegrator integrator;
        integrator.set_initial_solution(0.0, Vector{T});
        integrator.set_timestep(timestep);
        const TimeIntegrationResult result =
            integrator.advance_to(system, timestep, kPicard, kDirect);
        EXPECT_TRUE(result.completed());
        T = integrator.current_snapshot().u[0];
    }
    return T;
}

Vector solve_temperature_dependent_case(double timestep, LinearSolverOptions linear_solver) {
    const std::vector<double> x{0.0, 1.0, 2.0};
    const BasinGeometry geometry{
        .x = x, .z = {{0.0, 0.2, 0.0}, {2.0, 2.2, 2.0}}, .layers = {{1, 0.0, 2.0}}};
    const std::vector<double> layout = {0.0, 0.5, 1.0, 1.5, 2.0};
    const double dz = 0.4;
    const ColGrid grid = make_col_grid(geometry, layout, dz);
    const LithotypeProperties lithotype{
        .code = 1, .rho_s = 2500.0, .lambda20 = 2.5, .c20 = 850.0, .A_s = 0.0};
    const FluidProperties fluid{.lambda = 0.7, .rho = 1040.0, .c = 4184.0};
    const std::vector<double> phi(grid.cells.size(), 0.2);
    const std::vector<const LithotypeProperties*> lithotypes(grid.cells.size(), &lithotype);
    ColHeatSystem system(grid, phi, lithotypes, fluid, boundary(280.0));
    Vector initial(static_cast<Vector::Index>(grid.cells.size()));
    for (const ColCell& cell : grid.cells) {
        initial[static_cast<Vector::Index>(cell.index)] =
            320.0 + 0.5 * cell.center().x + 0.25 * cell.center().z;
    }
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, initial);
    integrator.set_timestep(timestep * 1.0e6);
    integrator.set_linear_solver(linear_solver);
    const NonlinearSolveRequest nonlinear{
        .method = NonlinearMethod::picard, .rtol = 1.0e-9, .atol = 1.0e-10, .maxiter = 100};
    const LinearSolveRequest linear{.rtol = 1.0e-10, .atol = 1.0e-12, .maxiter = 300};
    const TimeIntegrationResult result = integrator.advance_to(system, 1.0e6, nonlinear, linear);
    EXPECT_TRUE(result.completed());
    return integrator.current_snapshot().u;
}

double transient_error(double timestep) {
    constexpr double lambda = 2.0;
    constexpr double capacity = 3.0;
    constexpr double amplitude = 10.0;
    constexpr double final_time = 0.8;
    constexpr double kz = std::numbers::pi / 4.0;
    constexpr double decay = lambda * kz * kz / capacity;
    const auto grid = make_test_column_grid(1, 128, 1.0, 2.0);
    const std::vector<double> coefficients(grid.cells.size(), lambda);
    const std::vector<double> capacities(grid.cells.size(), capacity);
    const std::vector<double> source(grid.cells.size(), 0.0);
    ColHeatSystem system(
        grid, coefficients, source,
        {.T_surf = [](Point2D) { return 300.0; }, .q_basal = [](Point2D) { return 0.0; }},
        capacities);
    Vector initial(static_cast<Vector::Index>(grid.cells.size()));
    for (const auto& cell : grid.cells) {
        initial[static_cast<Vector::Index>(cell.index)] =
            300.0 + amplitude * std::sin(kz * cell.center().z);
    }
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, initial);
    integrator.set_timestep(timestep);
    const auto result = integrator.advance_to(
        system, final_time,
        {.method = NonlinearMethod::picard, .rtol = 1e-12, .atol = 1e-8, .maxiter = 10}, {});
    EXPECT_TRUE(result.completed()) << "residual=" << result.last_r_norm;
    return area_weighted_error(
        grid, {integrator.current_snapshot().u.native().data(), grid.cells.size()},
        [](double, double z) {
            return 300.0 + amplitude * std::exp(-decay * final_time) * std::sin(kz * z);
        });
}

} // namespace

TEST(ColHeatSystem, Bdf1ConvergesAtFirstOrderForDiscreteDecayMode) {
    // One cell gives the exact semi-discrete equation dT/dt + 2T = 0.
    const double exact = std::exp(-2.0);
    const double coarse_error = std::abs(integrate_decay(0.1) - exact);
    const double fine_error = std::abs(integrate_decay(0.05) - exact);
    EXPECT_GT(coarse_error / fine_error, 1.7);
    EXPECT_LT(coarse_error / fine_error, 2.1);
}

TEST(ColHeatSystem, MovingCellBdf1ConvergesAtFirstOrder) {
    // T=t^2, source=2t, surface T=t^2, basal flux=0 on a growing cell.
    const double coarse_error = std::abs(integrate_moving_quadratic(0.1) - 1.0);
    const double fine_error = std::abs(integrate_moving_quadratic(0.05) - 1.0);
    EXPECT_GT(coarse_error / fine_error, 1.7);
    EXPECT_LT(coarse_error / fine_error, 2.1);
}

TEST(ColHeatSystem, AccumulationUsesCurrentMovingCellAndPicardCapacity) {
    const ColGrid grid = single_cell_grid(2.0);
    const LithotypeProperties lithotype{
        .code = 1, .rho_s = 2500.0, .lambda20 = 2.5, .c20 = 850.0, .A_s = 0.0};
    const FluidProperties fluid{.lambda = 0.7, .rho = 1040.0, .c = 4184.0};
    const std::vector<double> phi{0.2};
    const std::vector<const LithotypeProperties*> lithotypes{&lithotype};
    ColHeatSystem system(grid, phi, lithotypes, fluid, boundary(300.0));
    for (const double value : {323.15, 373.15}) {
        const auto properties = thermal_properties(value, phi[0], lithotype, fluid);
        const std::vector<double> lambda{properties.lambda};
        const std::vector<double> source{properties.A};
        const ColHeatSystem spatial(grid, lambda, source, boundary(300.0));
        const Vector T{value};
        const Vector dT_dt{0.4};
        Vector spatial_residual;
        Vector transient_residual;
        spatial.assemble_residual(T, spatial_residual);
        system.assemble_residual(1.0, T, dT_dt, transient_residual);
        EXPECT_NEAR(transient_residual[0] - spatial_residual[0], 2.0 * properties.rho_c * 0.4,
                    1.0e-8);

        SparseMatrix M;
        SparseMatrix spatial_matrix;
        spatial.assemble_matrix(T, spatial_matrix);
        system.assemble_matrix(NonlinearMethod::picard, 1.0, T, dT_dt, 4.0, M);
        EXPECT_NEAR(M.coefficient(0, 0) - spatial_matrix.coefficient(0, 0),
                    2.0 * properties.rho_c * 4.0, 1.0e-8);

        system.assemble_residual(1.0, T, Vector{0.0}, transient_residual);
        EXPECT_NEAR(transient_residual[0], spatial_residual[0], 1.0e-12);
    }
}

TEST(ColHeatSystem, AdvancesAffineFieldOnIndependentColumns) {
    const std::vector<double> x{0.0, 1.0, 2.0};
    const BasinGeometry source{
        .x = x, .z = {{0.0, 0.4, 0.8}, {3.0, 3.0, 3.0}}, .layers = {{1, 0.0, 2.0}}};
    const std::vector<double> layout = x;
    const double dz = 0.45;
    const ColGrid grid = make_col_grid(source, layout, dz);
    constexpr double conductivity_value = 2.5;
    constexpr double capacity_value = 1.7;
    constexpr double x_gradient = 3.0;
    constexpr double z_gradient = -1.25;
    constexpr double time_gradient = 0.4;
    constexpr double final_time = 0.2;
    const auto exact = [](Point2D point, double time) {
        return 300.0 + x_gradient * point.x + z_gradient * point.z + time_gradient * time;
    };
    const std::vector<double> lambda(grid.cells.size(), conductivity_value);
    const std::vector<double> source_rate(grid.cells.size(), capacity_value * time_gradient);
    const std::vector<double> capacity(grid.cells.size(), capacity_value);
    Vector initial(static_cast<Vector::Index>(grid.cells.size()));
    for (const ColCell& cell : grid.cells) {
        initial[static_cast<Vector::Index>(cell.index)] = exact(cell.center(), 0.0);
    }

    for (const MultipointFluxScheme scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        ColHeatSystem system(grid, lambda, source_rate,
                             {.T_surf = [exact](Point2D point) { return exact(point, final_time); },
                              .q_basal = [](Point2D) { return conductivity_value * z_gradient; },
                              .q_left = [](Point2D) { return -conductivity_value * x_gradient; },
                              .q_right = [](Point2D) { return conductivity_value * x_gradient; }},
                             capacity);
        system.set_flux_scheme(scheme);
        TimeIntegrator integrator;
        integrator.set_initial_solution(0.0, initial);
        integrator.set_timestep(final_time);
        const TimeIntegrationResult result =
            integrator.advance_to(system, final_time, kPicard, kDirect);
        ASSERT_TRUE(result.completed());
        const Vector& final = integrator.current_snapshot().u;
        Vector dT_dt(final.size());
        for (const ColCell& cell : grid.cells) {
            const auto index = static_cast<Vector::Index>(cell.index);
            EXPECT_NEAR(final[index], exact(cell.center(), final_time), 1.0e-8);
            dT_dt[index] = (final[index] - initial[index]) / final_time;
        }
        Vector r;
        system.assemble_residual(final_time, final, dT_dt, r);
        EXPECT_LT(r.infinity_norm(), 1.0e-8);
    }
}

TEST(ColHeatSystem, TemperatureDependentCaseRefinesAndAgreesWithGamg) {
    ensure_petsc_test_session();
    const Vector coarse = solve_temperature_dependent_case(0.25, LinearSolverOptions{});
    const Vector medium = solve_temperature_dependent_case(0.125, LinearSolverOptions{});
    const Vector fine = solve_temperature_dependent_case(0.0625, LinearSolverOptions{});
    const double first_change = (medium - coarse).infinity_norm();
    const double second_change = (fine - medium).infinity_norm();
    EXPECT_GT(first_change, 1.0e-5);
    EXPECT_LT(second_change, 0.8 * first_change);

    const Vector gamg = solve_temperature_dependent_case(
        0.125,
        LinearSolverOptions{.method = LinearMethod::gmres, .preconditioner = Preconditioner::gamg});
    EXPECT_LT((gamg - medium).infinity_norm(), 1.0e-5);
}

TEST(ColHeatSystem, HasFirstOrderBdf1Convergence) {
    const double coarse = transient_error(0.1);
    const double medium = transient_error(0.05);
    const double fine = transient_error(0.025);
    EXPECT_GT(coarse / medium, 1.7);
    EXPECT_GT(medium / fine, 1.7);
}
