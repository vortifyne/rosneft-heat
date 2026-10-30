#include "basin/discretization/nonmatching_column_transient_system.hpp"
#include "basin/mesh/column_field_transfer.hpp"
#include "column_test_utils.hpp"
#include "numerics/nonlinear/nonlinear_solver.hpp"
#include "numerics/time/time_integrator.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kBaseTemperature = 300.0;

class SteadySystem final : public NonlinearSystem {
public:
    explicit SteadySystem(const NonmatchingColumnHeatSystem& system) : system_(system) {}

    void assemble_residual(const Vector& solution, Vector& residual) const override {
        system_.assemble_residual(solution, residual);
    }

private:
    void assemble_picard_matrix(const Vector& solution, SparseMatrix& matrix) const override {
        system_.assemble_matrix(solution, matrix);
    }

    void assemble_newton_matrix(const Vector&, SparseMatrix&) const override {
        throw std::invalid_argument("Stationary column verification supports Picard only");
    }

    const NonmatchingColumnHeatSystem& system_;
};

double area_weighted_error(const NonmatchingColumnGrid& grid, const Vector& solution,
                           const auto& exact) {
    double squared_error = 0.0;
    for (const auto& cell : grid.cells) {
        const Point2D point = cell.center();
        const double difference =
            solution[static_cast<Vector::Index>(cell.index)] - exact(point.x, point.z);
        squared_error += cell.area() * difference * difference;
    }
    return std::sqrt(squared_error / grid.area());
}

double steady_spatial_error(std::size_t nx, std::size_t nz, MultipointFluxScheme scheme) {
    constexpr double width = 4.0;
    constexpr double depth = 2.0;
    constexpr double conductivity = 2.0;
    constexpr double amplitude = 20.0;
    constexpr double kx = std::numbers::pi / width;
    constexpr double kz = std::numbers::pi / (2.0 * depth);
    const auto grid = make_test_column_grid(nx, nz, width, depth);
    const auto exact = [](double x, double z) {
        return kBaseTemperature + amplitude * std::cos(kx * x) * std::sin(kz * z);
    };
    std::vector<double> source(grid.cells.size());
    for (const auto& cell : grid.cells) {
        const auto point = cell.center();
        source[cell.index] =
            conductivity * (kx * kx + kz * kz) * (exact(point.x, point.z) - kBaseTemperature);
    }
    const std::vector<double> coefficients(grid.cells.size(), conductivity);
    NonmatchingColumnHeatSystem system(
        grid, coefficients, source,
        {.surface_temperature = [](Point2D) { return kBaseTemperature; },
         .basal_inward_flux = [](Point2D) { return 0.0; }});
    system.set_flux_scheme(scheme);
    Vector temperature(static_cast<Vector::Index>(grid.cells.size()));
    temperature.set_constant(kBaseTemperature);
    NonlinearSolver solver;
    const auto result = solver.solve(SteadySystem(system), temperature,
                                     {.nonlinear_method = NonlinearMethod::picard,
                                      .relative_tolerance = 1e-12,
                                      .absolute_tolerance = 1e-10,
                                      .max_iterations = 100},
                                     {});
    EXPECT_TRUE(result.converged()) << "residual=" << result.final_residual_norm;
    const auto rates = system.energy_rates(temperature);
    EXPECT_NEAR(rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow,
                rates.heat_production, 1e-8);
    EXPECT_LT(system.flux_diagnostics(temperature).internal_conservation_defect_linf, 1e-12);
    return area_weighted_error(grid, temperature, exact);
}

double transient_error(double timestep) {
    constexpr double conductivity = 2.0;
    constexpr double capacity = 3.0;
    constexpr double amplitude = 10.0;
    constexpr double final_time = 0.8;
    constexpr double kz = std::numbers::pi / 4.0;
    constexpr double decay = conductivity * kz * kz / capacity;
    const auto grid = make_test_column_grid(1, 128, 1.0, 2.0);
    const std::vector<double> coefficients(grid.cells.size(), conductivity);
    const std::vector<double> capacities(grid.cells.size(), capacity);
    const std::vector<double> source(grid.cells.size(), 0.0);
    NonmatchingColumnTransientSystem system(
        grid, coefficients, source, capacities,
        {.surface_temperature = [](Point2D) { return kBaseTemperature; },
         .basal_inward_flux = [](Point2D) { return 0.0; }});
    Vector initial(static_cast<Vector::Index>(grid.cells.size()));
    for (const auto& cell : grid.cells) {
        initial[static_cast<Vector::Index>(cell.index)] =
            kBaseTemperature + amplitude * std::sin(kz * cell.center().z);
    }
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, initial);
    integrator.set_timestep(timestep);
    const auto result = integrator.advance_to(system, final_time,
                                              {.nonlinear_method = NonlinearMethod::picard,
                                               .relative_tolerance = 1e-12,
                                               .absolute_tolerance = 1e-8,
                                               .max_iterations = 10},
                                              {});
    EXPECT_TRUE(result.completed()) << "residual=" << result.last_residual_norm;
    return area_weighted_error(grid, integrator.current_snapshot().solution, [](double, double z) {
        return kBaseTemperature + amplitude * std::exp(-decay * final_time) * std::sin(kz * z);
    });
}

double transfer_error(std::size_t nx, std::size_t nz) {
    constexpr double kx = std::numbers::pi / 4.0;
    constexpr double kz = std::numbers::pi / 2.0;
    const auto source = make_test_column_grid(nx, nz, 4.0, 2.0);
    const auto target = make_test_column_grid(nx * 2, nz * 2, 4.0, 2.0);
    const auto exact = [](double x, double z) {
        return kBaseTemperature + 15.0 * std::cos(kx * x) * std::sin(kz * z);
    };
    std::vector<double> values(source.cells.size());
    for (const auto& cell : source.cells)
        values[cell.index] = exact(cell.center().x, cell.center().z);
    const auto transferred =
        make_column_field_transfer(source, target).interpolate_temperature(values);
    Vector result(static_cast<Vector::Index>(transferred.size()));
    for (std::size_t i = 0; i < transferred.size(); ++i)
        result[static_cast<Vector::Index>(i)] = transferred[i];
    return area_weighted_error(target, result, exact);
}

} // namespace

TEST(ManufacturedHeatSolution, HasSecondOrderSpatialConvergence) {
    for (auto scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        SCOPED_TRACE(static_cast<int>(scheme));
        const double coarse = steady_spatial_error(16, 8, scheme);
        const double medium = steady_spatial_error(32, 16, scheme);
        const double fine = steady_spatial_error(64, 32, scheme);
        const double finer = steady_spatial_error(128, 64, scheme);
        const std::string prefix = scheme == MultipointFluxScheme::linear ? "linear" : "nmpfa";
        testing::Test::RecordProperty(prefix + "_coarse_error", std::to_string(coarse));
        testing::Test::RecordProperty(prefix + "_medium_error", std::to_string(medium));
        testing::Test::RecordProperty(prefix + "_fine_error", std::to_string(fine));
        testing::Test::RecordProperty(prefix + "_finer_error", std::to_string(finer));
        // The coarsest NMPFA grid precedes the asymptotic second-order range.
        EXPECT_GT(coarse, medium);
        EXPECT_GT(medium / fine, 3.5) << medium << ", " << fine;
        EXPECT_GT(fine / finer, 3.5) << fine << ", " << finer;
    }
}

TEST(ManufacturedHeatSolution, HasFirstOrderBdf1Convergence) {
    const double coarse = transient_error(0.1);
    const double medium = transient_error(0.05);
    const double fine = transient_error(0.025);
    EXPECT_GT(coarse / medium, 1.7);
    EXPECT_GT(medium / fine, 1.7);
}

TEST(ManufacturedHeatSolution, SmoothFieldTransferConvergesUnderRefinement) {
    const double coarse = transfer_error(8, 4);
    const double fine = transfer_error(16, 8);
    EXPECT_GT(coarse / fine, 1.7) << coarse << ", " << fine;
}
