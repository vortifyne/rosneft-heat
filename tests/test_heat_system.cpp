#include "discretization/heat_system.hpp"
#include "linear/umfpack_linear_solver.hpp"
#include "time/time_integrator.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {

Mesh2D make_rectangular_mesh() {
    const Mesh2D::Region region = {
        .vertices = {{0.0, 0.0}, {4.0, 0.0}, {4.0, 2.0}, {0.0, 2.0}},
        .edge_kinds = {BoundaryKind::top, BoundaryKind::right, BoundaryKind::bottom,
                       BoundaryKind::left},
        .id = 1,
    };
    return Mesh2D::generate(std::span<const Mesh2D::Region>(&region, 1),
                            {.cell_size = [](Point2D) { return 0.5; },
                             .smoothing_iterations = 2,
                             .make_quadrilateral = true,
                             .diagnostic_vtu = std::nullopt});
}

Mesh2D make_layered_mesh() {
    const std::array regions = {
        Mesh2D::Region{.vertices = {{0.0, 0.0}, {4.0, 0.0}, {4.0, 1.0}, {0.0, 1.0}},
                       .edge_kinds = {BoundaryKind::top, BoundaryKind::right,
                                      BoundaryKind::interface, BoundaryKind::left},
                       .id = 1},
        Mesh2D::Region{.vertices = {{0.0, 1.0}, {4.0, 1.0}, {4.0, 2.0}, {0.0, 2.0}},
                       .edge_kinds = {BoundaryKind::interface, BoundaryKind::right,
                                      BoundaryKind::bottom, BoundaryKind::left},
                       .id = 2},
    };
    return Mesh2D::generate(regions, {.cell_size = [](Point2D) { return 0.5; },
                                      .smoothing_iterations = 2,
                                      .make_quadrilateral = false,
                                      .diagnostic_vtu = std::nullopt});
}

HeatSystem make_system(const Mesh2D& mesh, const double conductivity = 2.0,
                       const double heat_capacity = 3.0, const double heat_production = 0.0,
                       const double surface_temperature = 300.0, const double basal_flux = 0.0) {
    const std::size_t count = mesh.cells().size();
    const std::vector conductivity_field(count, conductivity);
    const std::vector heat_capacity_field(count, heat_capacity);
    const std::vector source_field(count, heat_production);
    return HeatSystem(mesh, conductivity_field, heat_capacity_field, source_field,
                      {.surface_temperature =
                           [surface_temperature](Point2D, double) { return surface_temperature; },
                       .basal_heat_flux = [basal_flux](Point2D, double) { return basal_flux; }});
}

} // namespace

TEST(HeatSystem, KeepsConstantTemperatureAtSteadyState) {
    const Mesh2D mesh = make_rectangular_mesh();
    const HeatSystem system = make_system(mesh);
    Vector temperature(system.size());
    temperature.set_constant(300.0);
    Vector derivative(system.size());
    derivative.set_zero();
    Vector residual;

    system.assemble_residual(0.0, temperature, derivative, residual);

    EXPECT_LT(residual.infinity_norm(), 1.0e-11);
}

TEST(HeatSystem, ReproducesLinearVerticalTemperatureWithNonorthogonalCorrection) {
    constexpr double conductivity = 2.0;
    constexpr double gradient = 4.0;
    const Mesh2D mesh = make_rectangular_mesh();
    const HeatSystem system =
        make_system(mesh, conductivity, 3.0, 0.0, 300.0, conductivity * gradient);
    Vector temperature(system.size());
    for (const TQMesh::Facet* cell : mesh.cells()) {
        temperature[cell->index()] = 300.0 + gradient * cell->xy().y;
    }
    Vector derivative(system.size());
    derivative.set_zero();
    Vector residual;

    system.assemble_residual(0.0, temperature, derivative, residual);

    EXPECT_LT(residual.infinity_norm(), 1.0e-9);
}

TEST(HeatSystem, UsesConservativeHarmonicFluxAcrossMaterialInterface) {
    const Mesh2D mesh = make_layered_mesh();
    std::vector<double> conductivity(mesh.cells().size());
    std::vector<double> heat_capacity(mesh.cells().size(), 3.0);
    std::vector<double> source(mesh.cells().size(), 0.0);
    for (const TQMesh::Facet* cell : mesh.cells()) {
        conductivity[static_cast<std::size_t>(cell->index())] = cell->color() == 1 ? 2.0 : 8.0;
    }
    const HeatSystem system(mesh, conductivity, heat_capacity, source,
                            {.surface_temperature = [](Point2D, double) { return 300.0; },
                             .basal_heat_flux = [](Point2D, double) { return 0.0; }});
    Vector temperature(system.size());
    temperature.set_constant(300.0);
    Vector derivative(system.size());
    derivative.set_zero();
    SparseMatrix matrix(system.size(), system.size(), SparseStorageOrder::csc);

    system.assemble_matrix(NonlinearMethod::picard, 0.0, temperature, derivative, 0.0, matrix);

    bool checked_interface = false;
    for (const TQMesh::Edge* edge : mesh.internal_edges()) {
        const TQMesh::Facet& owner = mesh.owner(*edge);
        const TQMesh::Facet& neighbor = mesh.neighbor(*edge);
        if (owner.color() == neighbor.color()) {
            continue;
        }
        checked_interface = true;
        const Eigen::Vector2d normal(mesh.normal_from_owner(*edge).x,
                                     mesh.normal_from_owner(*edge).y);
        const Eigen::Vector2d owner_to_face(edge->xy().x - owner.xy().x,
                                            edge->xy().y - owner.xy().y);
        const Eigen::Vector2d face_to_neighbor(neighbor.xy().x - edge->xy().x,
                                               neighbor.xy().y - edge->xy().y);
        const Eigen::Vector2d displacement(neighbor.xy().x - owner.xy().x,
                                           neighbor.xy().y - owner.xy().y);
        const double owner_distance = std::abs(normal.dot(owner_to_face));
        const double neighbor_distance = std::abs(normal.dot(face_to_neighbor));
        const double face_conductivity =
            (owner_distance + neighbor_distance) /
            ((owner_distance / conductivity[static_cast<std::size_t>(owner.index())]) +
             (neighbor_distance / conductivity[static_cast<std::size_t>(neighbor.index())]));
        const double conductance = face_conductivity * edge->length() / normal.dot(displacement);
        const double owner_entry = matrix.coefficient(owner.index(), neighbor.index());
        const double neighbor_entry = matrix.coefficient(neighbor.index(), owner.index());
        EXPECT_NEAR(owner_entry, -conductance / owner.area(), 1.0e-12);
        EXPECT_NEAR(neighbor_entry, -conductance / neighbor.area(), 1.0e-12);
        EXPECT_NEAR(owner.area() * owner_entry, neighbor.area() * neighbor_entry, 1.0e-12);
    }
    EXPECT_TRUE(checked_interface);
}

TEST(HeatSystem, SolvesConstantCoefficientSteadyProblemWithUmfpack) {
    constexpr double conductivity = 2.0;
    constexpr double gradient = 4.0;
    const Mesh2D mesh = make_rectangular_mesh();
    const HeatSystem system =
        make_system(mesh, conductivity, 3.0, 0.0, 300.0, conductivity * gradient);
    Vector temperature(system.size());
    temperature.set_constant(300.0);
    Vector derivative(system.size());
    derivative.set_zero();
    UmfpackLinearSolver solver;

    for (int iteration = 0; iteration < 20; ++iteration) {
        Vector residual;
        system.assemble_residual(0.0, temperature, derivative, residual);
        if (residual.infinity_norm() < 1.0e-9) {
            break;
        }
        SparseMatrix matrix(system.size(), system.size(), SparseStorageOrder::csc);
        system.assemble_matrix(NonlinearMethod::picard, 0.0, temperature, derivative, 0.0, matrix);
        Vector correction;
        const LinearSolveResult result = solver.solve(
            matrix, -residual, correction,
            {.relative_tolerance = 1.0e-12, .absolute_tolerance = 1.0e-12, .max_iterations = 1});
        ASSERT_EQ(result.status, LinearSolveStatus::converged);
        temperature += correction;
    }

    Vector residual;
    system.assemble_residual(0.0, temperature, derivative, residual);
    EXPECT_LT(residual.infinity_norm(), 1.0e-8);
    for (const TQMesh::Facet* cell : mesh.cells()) {
        EXPECT_NEAR(temperature[cell->index()], 300.0 + gradient * cell->xy().y, 1.0e-7);
    }
}

TEST(HeatSystem, AdvancesOneBdf1StepWithPicardMethod) {
    const Mesh2D mesh = make_rectangular_mesh();
    const HeatSystem system = make_system(mesh, 2.0, 3.0, 0.0, 310.0, 0.0);
    Vector initial_temperature(system.size());
    initial_temperature.set_constant(300.0);
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, initial_temperature);
    integrator.set_timestep(0.1);

    const TimeIntegrationResult result = integrator.advance_to(
        system, 0.1,
        {.nonlinear_method = NonlinearMethod::picard,
         .relative_tolerance = 1.0e-10,
         .absolute_tolerance = 1.0e-9,
         .step_relative_tolerance = 1.0e-12,
         .max_iterations = 20},
        {.relative_tolerance = 1.0e-12, .absolute_tolerance = 1.0e-12, .max_iterations = 1});

    EXPECT_TRUE(result.completed());
    EXPECT_EQ(result.accepted_steps, 1);
    EXPECT_GT(integrator.current_snapshot().solution.native().maxCoeff(), 300.0);
}

TEST(HeatSystem, RejectsNewtonMatrix) {
    const Mesh2D mesh = make_rectangular_mesh();
    const HeatSystem system = make_system(mesh);
    Vector temperature(system.size());
    temperature.set_constant(300.0);
    Vector derivative(system.size());
    derivative.set_zero();
    SparseMatrix matrix;

    EXPECT_THROW(
        system.assemble_matrix(NonlinearMethod::newton, 0.0, temperature, derivative, 0.0, matrix),
        std::invalid_argument);
}
