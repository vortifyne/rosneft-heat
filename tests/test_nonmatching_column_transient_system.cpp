#include "discretization/nonmatching_column_transient_system.hpp"
#include "petsc_test_session.hpp"
#include "time/time_integrator.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <optional>
#include <vector>

namespace {

NonmatchingColumnGrid single_cell_grid(double height) {
    NonmatchingColumnGrid grid;
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

NonmatchingColumnBoundaryData boundary(double surface_temperature) {
    return {.surface_temperature = [surface_temperature](Point2D) {
                return surface_temperature;
            },
            .basal_inward_flux = [](Point2D) { return 0.0; }};
}

constexpr NonlinearSolveRequest kPicard{
    .nonlinear_method = NonlinearMethod::picard,
    .relative_tolerance = 1.0e-12,
    .absolute_tolerance = 1.0e-12,
    .max_iterations = 20,
};

constexpr LinearSolveRequest kDirect{
    .relative_tolerance = 1.0e-12,
    .absolute_tolerance = 1.0e-12,
    .max_iterations = 50,
};

double integrate_decay(double timestep) {
    const NonmatchingColumnGrid grid = single_cell_grid(1.0);
    const std::vector<double> conductivity{1.0};
    const std::vector<double> source{0.0};
    const std::vector<double> capacity{1.0};
    NonmatchingColumnTransientSystem system(grid, conductivity, source, capacity, boundary(0.0));
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, Vector{1.0});
    integrator.set_timestep(timestep);
    const TimeIntegrationResult result = integrator.advance_to(system, 1.0, kPicard, kDirect);
    EXPECT_TRUE(result.completed());
    return integrator.current_snapshot().solution[0];
}

double integrate_moving_quadratic(double timestep) {
    double temperature = 0.0;
    const int steps = static_cast<int>(std::round(1.0 / timestep));
    for (int step = 1; step <= steps; ++step) {
        const double time = static_cast<double>(step) * timestep;
        const NonmatchingColumnGrid grid = single_cell_grid(1.0 + 0.5 * time);
        const std::vector<double> conductivity{1.0};
        const std::vector<double> source{2.0 * time};
        const std::vector<double> capacity{1.0};
        NonmatchingColumnTransientSystem system(grid, conductivity, source, capacity,
                                                boundary(time * time));
        TimeIntegrator integrator;
        integrator.set_initial_solution(0.0, Vector{temperature});
        integrator.set_timestep(timestep);
        const TimeIntegrationResult result =
            integrator.advance_to(system, timestep, kPicard, kDirect);
        EXPECT_TRUE(result.completed());
        temperature = integrator.current_snapshot().solution[0];
    }
    return temperature;
}

Vector solve_temperature_dependent_case(double timestep, LinearSolverKind linear_solver) {
    const std::vector<double> x{0.0, 1.0, 2.0};
    const BasinGeometry geometry{
        .age_ma = 0.0,
        .boundaries = {{{x, {0.0, 0.2, 0.0}}}, {{x, {2.0, 2.2, 2.0}}}},
        .layers = {{1, 0, 1, 0.0, 2.0}}};
    const LayeredMeshLayout layout{.x = {0.0, 0.5, 1.0, 1.5, 2.0}, .cell_size = 0.4};
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(geometry, layout);
    const std::vector<double> conductivity(grid.cells.size(), 1.0);
    const std::vector<double> source(grid.cells.size(), 0.0);
    const std::vector<double> capacity(grid.cells.size(), 1.5);
    NonmatchingColumnTransientSystem system(
        grid, conductivity, source, capacity, boundary(280.0),
        [](std::span<const double> temperature, std::span<double> updated_conductivity,
           std::span<double>) {
            for (std::size_t i = 0; i < temperature.size(); ++i) {
                updated_conductivity[i] = 1.0 + 0.002 * (temperature[i] - 280.0);
            }
        });
    Vector initial(static_cast<Vector::Index>(grid.cells.size()));
    for (const NonmatchingColumnCell& cell : grid.cells) {
        initial[static_cast<Vector::Index>(cell.index)] =
            320.0 + 0.5 * cell.center().x + 0.25 * cell.center().z;
    }
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, initial);
    integrator.set_timestep(timestep);
    integrator.set_linear_solver(linear_solver);
    const NonlinearSolveRequest nonlinear{.nonlinear_method = NonlinearMethod::picard,
                                          .relative_tolerance = 1.0e-9,
                                          .absolute_tolerance = 1.0e-10,
                                          .max_iterations = 100};
    const LinearSolveRequest linear{.relative_tolerance = 1.0e-10,
                                    .absolute_tolerance = 1.0e-12,
                                    .max_iterations = 300};
    const TimeIntegrationResult result = integrator.advance_to(system, 1.0, nonlinear, linear);
    EXPECT_TRUE(result.completed());
    return integrator.current_snapshot().solution;
}

} // namespace

TEST(NonmatchingColumnTransientSystem, Bdf1ConvergesAtFirstOrderForDiscreteDecayMode) {
    // One cell gives the exact semi-discrete equation dT/dt + 2T = 0.
    const double exact = std::exp(-2.0);
    const double coarse_error = std::abs(integrate_decay(0.1) - exact);
    const double fine_error = std::abs(integrate_decay(0.05) - exact);
    EXPECT_GT(coarse_error / fine_error, 1.7);
    EXPECT_LT(coarse_error / fine_error, 2.1);
}

TEST(NonmatchingColumnTransientSystem, MovingCellBdf1ConvergesAtFirstOrder) {
    // T=t^2, source=2t, surface T=t^2, basal flux=0 on a growing cell.
    const double coarse_error = std::abs(integrate_moving_quadratic(0.1) - 1.0);
    const double fine_error = std::abs(integrate_moving_quadratic(0.05) - 1.0);
    EXPECT_GT(coarse_error / fine_error, 1.7);
    EXPECT_LT(coarse_error / fine_error, 2.1);
}

TEST(NonmatchingColumnTransientSystem, AccumulationUsesCurrentMovingCellAndPicardCapacity) {
    const NonmatchingColumnGrid grid = single_cell_grid(2.0);
    const std::vector<double> conductivity{1.0};
    const std::vector<double> source{0.0};
    const std::vector<double> capacity{1.0};
    NonmatchingColumnTransientSystem system(
        grid, conductivity, source, capacity, boundary(3.0), {},
        [](double, std::span<const double> temperature, std::span<double> updated) {
            updated[0] = 2.0 + temperature[0];
        });
    const Vector temperature{3.0};
    const Vector derivative{0.4};
    Vector spatial_residual;
    Vector transient_residual;
    system.spatial_system().assemble_residual(temperature, spatial_residual);
    system.assemble_residual(1.0, temperature, derivative, transient_residual);
    EXPECT_NEAR(transient_residual[0] - spatial_residual[0], 2.0 * 5.0 * 0.4, 1.0e-12);

    SparseMatrix matrix;
    SparseMatrix spatial_matrix;
    system.spatial_system().assemble_matrix(temperature, spatial_matrix);
    system.assemble_matrix(NonlinearMethod::picard, 1.0, temperature, derivative, 4.0, matrix);
    EXPECT_NEAR(matrix.coefficient(0, 0) - spatial_matrix.coefficient(0, 0), 2.0 * 5.0 * 4.0,
                1.0e-12);

    system.assemble_residual(1.0, temperature, Vector{0.0}, transient_residual);
    EXPECT_NEAR(transient_residual[0], spatial_residual[0], 1.0e-12);
}

TEST(NonmatchingColumnTransientSystem, AdvancesAffineFieldOnIndependentColumns) {
    const std::vector<double> x{0.0, 1.0, 2.0};
    const BasinGeometry source{
        .age_ma = 0.0,
        .boundaries = {{{x, {0.0, 0.4, 0.8}}}, {{x, {3.0, 3.0, 3.0}}}},
        .layers = {{1, 0, 1, 0.0, 2.0}}};
    const LayeredMeshLayout layout{.x = x, .cell_size = 0.45};
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(source, layout);
    constexpr double conductivity_value = 2.5;
    constexpr double capacity_value = 1.7;
    constexpr double x_gradient = 3.0;
    constexpr double z_gradient = -1.25;
    constexpr double time_gradient = 0.4;
    constexpr double final_time = 0.2;
    const auto exact = [](Point2D point, double time) {
        return 300.0 + x_gradient * point.x + z_gradient * point.z + time_gradient * time;
    };
    const std::vector<double> conductivity(grid.cells.size(), conductivity_value);
    const std::vector<double> source_rate(grid.cells.size(), capacity_value * time_gradient);
    const std::vector<double> capacity(grid.cells.size(), capacity_value);
    Vector initial(static_cast<Vector::Index>(grid.cells.size()));
    for (const NonmatchingColumnCell& cell : grid.cells) {
        initial[static_cast<Vector::Index>(cell.index)] = exact(cell.center(), 0.0);
    }

    for (const MultipointFluxScheme scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        NonmatchingColumnTransientSystem system(
            grid, conductivity, source_rate, capacity,
            {.surface_temperature = [exact](Point2D point) { return exact(point, final_time); },
             .basal_inward_flux = [](Point2D) { return conductivity_value * z_gradient; },
             .left_inward_flux = [](Point2D) { return -conductivity_value * x_gradient; },
             .right_inward_flux = [](Point2D) { return conductivity_value * x_gradient; }});
        system.set_flux_scheme(scheme);
        TimeIntegrator integrator;
        integrator.set_initial_solution(0.0, initial);
        integrator.set_timestep(final_time);
        const TimeIntegrationResult result =
            integrator.advance_to(system, final_time, kPicard, kDirect);
        ASSERT_TRUE(result.completed());
        const Vector& final = integrator.current_snapshot().solution;
        Vector derivative(final.size());
        for (const NonmatchingColumnCell& cell : grid.cells) {
            const auto index = static_cast<Vector::Index>(cell.index);
            EXPECT_NEAR(final[index], exact(cell.center(), final_time), 1.0e-8);
            derivative[index] = (final[index] - initial[index]) / final_time;
        }
        Vector residual;
        system.assemble_residual(final_time, final, derivative, residual);
        EXPECT_LT(residual.infinity_norm(), 1.0e-8);
    }
}

TEST(NonmatchingColumnTransientSystem, TemperatureDependentCaseRefinesAndAgreesWithGamg) {
    ensure_petsc_test_session();
    const Vector coarse = solve_temperature_dependent_case(0.25, LinearSolverKind::umfpack_lu);
    const Vector medium = solve_temperature_dependent_case(0.125, LinearSolverKind::umfpack_lu);
    const Vector fine = solve_temperature_dependent_case(0.0625, LinearSolverKind::umfpack_lu);
    const double first_change = (medium - coarse).infinity_norm();
    const double second_change = (fine - medium).infinity_norm();
    EXPECT_GT(first_change, 1.0e-5);
    EXPECT_LT(second_change, 0.8 * first_change);

    const Vector gamg =
        solve_temperature_dependent_case(0.125, LinearSolverKind::petsc_gmres_gamg);
    EXPECT_LT((gamg - medium).infinity_norm(), 1.0e-5);
}
