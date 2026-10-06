#include "discretization/heat_system.hpp"
#include "linear/umfpack_linear_solver.hpp"
#include "mesh/mesh_transfer.hpp"
#include "time/time_integrator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr double kBaseTemperature = 300.0;

Mesh2D make_regular_mesh(const std::size_t nx, const std::size_t nz, const double width,
                         const double depth) {
    std::vector<Mesh2D::Cell> cells;
    cells.reserve(nx * nz);
    for (std::size_t iz = 0; iz < nz; ++iz) {
        const double top = depth * static_cast<double>(iz) / static_cast<double>(nz);
        const double bottom = depth * static_cast<double>(iz + 1) / static_cast<double>(nz);
        for (std::size_t ix = 0; ix < nx; ++ix) {
            const double left = width * static_cast<double>(ix) / static_cast<double>(nx);
            const double right = width * static_cast<double>(ix + 1) / static_cast<double>(nx);
            cells.push_back(
                {.vertices = {{left, top}, {right, top}, {right, bottom}, {left, bottom}},
                 .edge_kinds =
                     {
                         iz == 0 ? BoundaryKind::top : BoundaryKind::interface,
                         ix + 1 == nx ? BoundaryKind::right : BoundaryKind::interface,
                         iz + 1 == nz ? BoundaryKind::bottom : BoundaryKind::interface,
                         ix == 0 ? BoundaryKind::left : BoundaryKind::interface,
                     },
                 .id = 1});
        }
    }
    return Mesh2D::from_cells(cells);
}

double area_weighted_error(const Mesh2D& mesh, const Vector& solution, const auto& exact_solution) {
    double squared_error = 0.0;
    double area = 0.0;
    for (const TQMesh::Facet* cell : mesh.cells()) {
        const double difference =
            solution[cell->index()] - exact_solution(cell->xy().x, cell->xy().y);
        squared_error += cell->area() * difference * difference;
        area += cell->area();
    }
    return std::sqrt(squared_error / area);
}

double steady_spatial_error(const std::size_t nx, const std::size_t nz) {
    constexpr double width = 4.0;
    constexpr double depth = 2.0;
    constexpr double conductivity = 2.0;
    constexpr double amplitude = 20.0;
    constexpr double kx = std::numbers::pi / width;
    constexpr double kz = std::numbers::pi / (2.0 * depth);
    Mesh2D mesh = make_regular_mesh(nx, nz, width, depth);
    const auto exact = [](const double x, const double z) {
        return kBaseTemperature + amplitude * std::cos(kx * x) * std::sin(kz * z);
    };
    const std::size_t count = mesh.cells().size();
    const std::vector<double> conductivity_field(count, conductivity);
    const std::vector<double> heat_capacity(count, 1.0);
    std::vector<double> source(count);
    for (const TQMesh::Facet* cell : mesh.cells()) {
        source[static_cast<std::size_t>(cell->index())] =
            conductivity * (kx * kx + kz * kz) *
            (exact(cell->xy().x, cell->xy().y) - kBaseTemperature);
    }
    HeatSystem system(mesh, conductivity_field, heat_capacity, source,
                      {.surface_temperature = [](Point2D, double) { return kBaseTemperature; },
                       .basal_heat_flux = [](Point2D, double) { return 0.0; }});
    system.set_implicit_nonorthogonal_correction(true);
    Vector temperature(static_cast<Vector::Index>(count));
    temperature.set_constant(kBaseTemperature);
    Vector derivative(static_cast<Vector::Index>(count));
    derivative.set_zero();
    Vector residual;
    system.assemble_residual(0.0, temperature, derivative, residual);
    SparseMatrix matrix(static_cast<Vector::Index>(count), static_cast<Vector::Index>(count),
                        SparseStorageOrder::csc);
    system.assemble_matrix(NonlinearMethod::picard, 0.0, temperature, derivative, 0.0, matrix);
    UmfpackLinearSolver solver;
    Vector correction;
    const LinearSolveResult result = solver.solve(
        matrix, -residual, correction,
        {.relative_tolerance = 1.0e-12, .absolute_tolerance = 1.0e-12, .max_iterations = 1});
    EXPECT_EQ(result.status, LinearSolveStatus::converged);
    temperature += correction;
    return area_weighted_error(mesh, temperature, exact);
}

double transient_error(const double timestep) {
    constexpr double width = 1.0;
    constexpr double depth = 1.0;
    constexpr double conductivity = 1.0;
    constexpr double heat_capacity = 1.0;
    constexpr double amplitude = 10.0;
    constexpr double kz = std::numbers::pi / (2.0 * depth);
    constexpr double final_time = 0.2;
    constexpr double decay = conductivity * kz * kz / heat_capacity;
    Mesh2D mesh = make_regular_mesh(1, 128, width, depth);
    const std::size_t count = mesh.cells().size();
    const std::vector<double> conductivity_field(count, conductivity);
    const std::vector<double> heat_capacity_field(count, heat_capacity);
    const std::vector<double> source(count, 0.0);
    HeatSystem system(mesh, conductivity_field, heat_capacity_field, source,
                      {.surface_temperature = [](Point2D, double) { return kBaseTemperature; },
                       .basal_heat_flux = [](Point2D, double) { return 0.0; }});
    Vector initial(static_cast<Vector::Index>(count));
    for (const TQMesh::Facet* cell : mesh.cells()) {
        initial[cell->index()] = kBaseTemperature + amplitude * std::sin(kz * cell->xy().y);
    }
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, initial);
    integrator.set_timestep(timestep);
    const TimeIntegrationResult result = integrator.advance_to(
        system, final_time,
        {.nonlinear_method = NonlinearMethod::picard,
         .relative_tolerance = 1.0e-12,
         .absolute_tolerance = 1.0e-12,
         .step_relative_tolerance = 1.0e-13,
         .max_iterations = 10},
        {.relative_tolerance = 1.0e-12, .absolute_tolerance = 1.0e-12, .max_iterations = 1});
    EXPECT_TRUE(result.completed());
    const auto exact = [](double, const double z) {
        return kBaseTemperature + amplitude * std::exp(-decay * final_time) * std::sin(kz * z);
    };
    return area_weighted_error(mesh, integrator.current_snapshot().solution, exact);
}

double transfer_error(const std::size_t source_nx, const std::size_t source_nz) {
    constexpr double width = 4.0;
    constexpr double depth = 2.0;
    constexpr double kx = std::numbers::pi / width;
    constexpr double kz = std::numbers::pi / depth;
    Mesh2D source = make_regular_mesh(source_nx, source_nz, width, depth);
    Mesh2D target = make_regular_mesh(source_nx * 2, source_nz * 2, width, depth);
    const auto exact = [](const double x, const double z) {
        return kBaseTemperature + 15.0 * std::cos(kx * x) * std::sin(kz * z);
    };
    std::vector<double> values(source.cells().size());
    for (const TQMesh::Facet* cell : source.cells()) {
        values[static_cast<std::size_t>(cell->index())] = exact(cell->xy().x, cell->xy().y);
    }
    const CellTransferMap map = make_cell_transfer_map(source, target);
    const std::vector<double> transferred =
        transfer_cell_field(source, values, map, [](std::size_t, Point2D) {
            return std::numeric_limits<double>::quiet_NaN();
        });
    Vector result(static_cast<Vector::Index>(transferred.size()));
    for (std::size_t index = 0; index < transferred.size(); ++index) {
        result[static_cast<Vector::Index>(index)] = transferred[index];
    }
    return area_weighted_error(target, result, exact);
}

} // namespace

TEST(ManufacturedHeatSolution, HasSecondOrderSpatialConvergence) {
    const double coarse = steady_spatial_error(16, 8);
    const double medium = steady_spatial_error(32, 16);
    const double fine = steady_spatial_error(64, 32);

    testing::Test::RecordProperty("coarse_error", std::to_string(coarse));
    testing::Test::RecordProperty("medium_error", std::to_string(medium));
    testing::Test::RecordProperty("fine_error", std::to_string(fine));

    EXPECT_GT(coarse / medium, 3.5);
    EXPECT_GT(medium / fine, 3.5);
}

TEST(ManufacturedHeatSolution, HasFirstOrderBdf1Convergence) {
    const double coarse = transient_error(0.1);
    const double medium = transient_error(0.05);
    const double fine = transient_error(0.025);

    testing::Test::RecordProperty("coarse_error", std::to_string(coarse));
    testing::Test::RecordProperty("medium_error", std::to_string(medium));
    testing::Test::RecordProperty("fine_error", std::to_string(fine));

    EXPECT_GT(coarse / medium, 1.7);
    EXPECT_GT(medium / fine, 1.7);
}

TEST(ManufacturedHeatSolution, SmoothFieldTransferConvergesUnderRefinement) {
    const double coarse = transfer_error(8, 4);
    const double fine = transfer_error(16, 8);

    testing::Test::RecordProperty("coarse_error", std::to_string(coarse));
    testing::Test::RecordProperty("fine_error", std::to_string(fine));

    EXPECT_GT(coarse / fine, 1.7);
}
