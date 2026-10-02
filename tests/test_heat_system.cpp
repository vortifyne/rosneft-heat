#include "discretization/heat_system.hpp"
#include "linear/umfpack_linear_solver.hpp"
#include "mesh_test_utils.hpp"
#include "time/time_integrator.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {

Mesh2D make_rectangular_mesh(const double cell_size = 0.5) {
    return make_rectangular_test_mesh(4.0, 2.0, cell_size, 1, 0.0, true);
}

Mesh2D make_layered_mesh() {
    std::vector<Mesh2D::Cell> cells;
    for (int iz = 0; iz < 4; ++iz) {
        for (int ix = 0; ix < 8; ++ix) {
            const double x0 = 0.5 * static_cast<double>(ix);
            const double z0 = 0.5 * static_cast<double>(iz);
            cells.push_back({
                .vertices = {{x0, z0}, {x0 + 0.5, z0}, {x0 + 0.5, z0 + 0.5}, {x0, z0 + 0.5}},
                .edge_kinds =
                    {
                        iz == 0 ? BoundaryKind::top : BoundaryKind::interface,
                        ix == 7 ? BoundaryKind::right : BoundaryKind::interface,
                        iz == 3 ? BoundaryKind::bottom : BoundaryKind::interface,
                        ix == 0 ? BoundaryKind::left : BoundaryKind::interface,
                    },
                .id = iz < 2 ? 1 : 2,
            });
        }
    }
    return Mesh2D::from_cells(cells);
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

TEST(HeatSystem, ReportsWholeDomainEnergyRates) {
    constexpr double conductivity = 2.0;
    constexpr double gradient = 4.0;
    constexpr double heat_production = 3.0;
    const Mesh2D mesh = make_rectangular_mesh();
    const HeatSystem system =
        make_system(mesh, conductivity, 3.0, heat_production, 300.0, conductivity * gradient);
    Vector temperature(system.size());
    for (const TQMesh::Facet* cell : mesh.cells()) {
        temperature[cell->index()] = 300.0 + gradient * cell->xy().y;
    }

    const HeatEnergyRates rates = system.energy_rates(0.0, temperature);

    EXPECT_NEAR(rates.surface_outflow, conductivity * gradient * 4.0, 1.0e-9);
    EXPECT_NEAR(rates.basal_inflow, conductivity * gradient * 4.0, 1.0e-12);
    EXPECT_NEAR(rates.heat_production, heat_production * 4.0 * 2.0, 1.0e-12);
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

TEST(HeatSystem, KeepsConstantTemperatureWhileMeshMoves) {
    Mesh2D mesh = make_rectangular_mesh();
    HeatSystem system = make_system(mesh);
    Vector initial_temperature(system.size());
    initial_temperature.set_constant(300.0);
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, initial_temperature);
    integrator.set_timestep(0.1);
    const std::vector<Point2D> initial_coordinates = mesh.vertex_coordinates();

    for (int step = 1; step <= 3; ++step) {
        std::vector<Point2D> coordinates = initial_coordinates;
        const double vertical_scale = 1.0 + (0.05 * static_cast<double>(step));
        for (Point2D& point : coordinates) {
            point.z *= vertical_scale;
        }
        mesh.set_vertex_coordinates(coordinates);
        system.update_geometry(mesh);
        const TimeIntegrationResult result = integrator.advance_to(
            system, 0.1 * static_cast<double>(step),
            {.nonlinear_method = NonlinearMethod::picard,
             .relative_tolerance = 1.0e-10,
             .absolute_tolerance = 1.0e-9,
             .step_relative_tolerance = 1.0e-12,
             .max_iterations = 20},
            {.relative_tolerance = 1.0e-12, .absolute_tolerance = 1.0e-12, .max_iterations = 1});
        ASSERT_TRUE(result.completed());
    }

    for (Vector::Index index = 0; index < system.size(); ++index) {
        EXPECT_NEAR(integrator.current_snapshot().solution[index], 300.0, 1.0e-10);
    }
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

TEST(HeatSystem, ImplicitNonorthogonalMatrixMatchesLinearResidual) {
    const Mesh2D mesh = make_rectangular_mesh();
    HeatSystem system = make_system(mesh, 2.0, 3.0, 0.0, 300.0, 0.0);
    system.set_implicit_nonorthogonal_correction(true);
    Vector temperature(system.size());
    Vector direction(system.size());
    for (Vector::Index index = 0; index < system.size(); ++index) {
        temperature[index] = 300.0 + (0.1 * static_cast<double>(index));
        direction[index] = std::sin(static_cast<double>(index) + 0.5);
    }
    Vector derivative(system.size());
    derivative.set_zero();
    SparseMatrix matrix;
    system.assemble_matrix(NonlinearMethod::picard, 0.0, temperature, derivative, 0.0, matrix);
    Vector residual;
    Vector shifted_residual;
    system.assemble_residual(0.0, temperature, derivative, residual);
    constexpr double epsilon = 1.0e-3;
    system.assemble_residual(0.0, temperature + (direction * epsilon), derivative,
                             shifted_residual);
    const Vector difference =
        ((shifted_residual - residual) * (1.0 / epsilon)) - (matrix * direction);

    EXPECT_LT(difference.infinity_norm(), 1.0e-7);
}

TEST(HeatSystem, RefreshesTemperatureDependentPropertiesForResidualAndMatrix) {
    const Mesh2D mesh = make_rectangular_mesh();
    const std::size_t count = mesh.cells().size();
    const std::vector conductivity(count, 1.0);
    const std::vector heat_capacity(count, 2.0);
    const std::vector heat_production(count, 0.0);
    int updates = 0;
    HeatSystem system(mesh, conductivity, heat_capacity, heat_production,
                      {.surface_temperature = [](Point2D, double) { return 300.0; },
                       .basal_heat_flux = [](Point2D, double) { return 0.0; }},
                      [&updates](const std::span<const double> temperature,
                                 const std::span<double> updated_conductivity,
                                 const std::span<double> updated_heat_capacity,
                                 const std::span<double> updated_heat_production) {
                          ++updates;
                          for (std::size_t index = 0; index < temperature.size(); ++index) {
                              updated_conductivity[index] = 0.01 * temperature[index];
                              updated_heat_capacity[index] = 1000.0 + temperature[index];
                              updated_heat_production[index] = 1.0e-6;
                          }
                      });
    Vector temperature(system.size());
    temperature.set_constant(350.0);
    Vector derivative(system.size());
    derivative.set_zero();
    Vector residual;
    SparseMatrix matrix;

    system.assemble_residual(0.0, temperature, derivative, residual);
    system.assemble_matrix(NonlinearMethod::picard, 0.0, temperature, derivative, 1.0, matrix);

    EXPECT_EQ(updates, 2);
    EXPECT_DOUBLE_EQ(system.thermal_conductivity().front(), 3.5);
    EXPECT_DOUBLE_EQ(system.volumetric_heat_capacity().front(), 1350.0);
    EXPECT_DOUBLE_EQ(system.heat_production().front(), 1.0e-6);
}
