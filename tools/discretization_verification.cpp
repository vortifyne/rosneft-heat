#include "basin/layered_mesh.hpp"
#include "discretization/heat_system.hpp"
#include "linear/umfpack_linear_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr double kWidth = 4.0;
constexpr double kBaseTemperature = 300.0;

enum class MeshFamily {
    fixed_rows,
    variable_rows
};

struct ErrorMetrics {
    double l2 = 0.0;
    double linf = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
};

struct SmoothResult {
    std::size_t cells = 0;
    Eigen::Index nonzeros = 0;
    std::size_t nonlinear_iterations = 0;
    std::size_t positive_offdiagonal_count = 0;
    double maximum_positive_offdiagonal = 0.0;
    std::size_t maximum_stencil_size = 0;
    double internal_conservation_defect_linf = 0.0;
    ErrorMetrics error;
    double lower_bound_violation = 0.0;
    double upper_bound_violation = 0.0;
    double integrated_residual_linf = 0.0;
    double global_balance = 0.0;
};

struct MatrixSignMetrics {
    std::size_t positive_offdiagonal_count = 0;
    double maximum_positive_offdiagonal = 0.0;
};

struct StationarySolution {
    Vector temperature;
    SparseMatrix matrix;
    Vector residual;
    std::size_t nonlinear_iterations = 0;
};

struct PatchResult {
    std::size_t cells = 0;
    std::size_t measured_cells = 0;
    double integrated_residual_linf = 0.0;
    double integrated_residual_l1 = 0.0;
};

ScalarProfile profile(std::vector<double> x, std::vector<double> values) {
    return {.x = std::move(x), .values = std::move(values)};
}

BasinGeometry make_smooth_geometry() {
    const std::vector<double> x{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    std::vector<double> top;
    std::vector<double> contact;
    std::vector<double> bottom;
    top.reserve(x.size());
    contact.reserve(x.size());
    bottom.reserve(x.size());
    for (const double coordinate : x) {
        const double roof = 0.08 * std::sin(2.0 * std::numbers::pi * coordinate / kWidth);
        top.push_back(roof);
        contact.push_back(roof + 0.55 +
                          0.22 * (1.0 + std::sin(std::numbers::pi * coordinate / kWidth)));
        bottom.push_back(2.1 + 0.12 * std::cos(2.0 * std::numbers::pi * coordinate / kWidth));
    }
    return {.age_ma = 0.0,
            .boundaries = {{profile(x, top)}, {profile(x, contact)}, {profile(x, bottom)}},
            .layers = {{1, 0, 1, 0.0, kWidth}, {2, 1, 2, 0.0, kWidth}}};
}

BasinGeometry make_horizontal_contact_geometry() {
    const std::vector<double> x{0.0, 1.0, 2.0, 3.0, 4.0};
    return {.age_ma = 0.0,
            .boundaries = {{profile(x, std::vector<double>(x.size(), 0.0))},
                           {profile(x, std::vector<double>(x.size(), 1.0))},
                           {profile(x, std::vector<double>(x.size(), 2.0))}},
            .layers = {{1, 0, 1, 0.0, kWidth}, {2, 1, 2, 0.0, kWidth}}};
}

BasinGeometry make_tilted_contact_geometry() {
    const std::vector<double> x{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    std::vector<double> top(x.size(), 0.0);
    std::vector<double> contact;
    std::vector<double> bottom(x.size(), 2.6);
    contact.reserve(x.size());
    for (const double coordinate : x) {
        contact.push_back(0.7 + 0.3 * coordinate);
    }
    return {.age_ma = 0.0,
            .boundaries = {{profile(x, top)}, {profile(x, contact)}, {profile(x, bottom)}},
            .layers = {{1, 0, 1, 0.0, kWidth}, {2, 1, 2, 0.0, kWidth}}};
}

Mesh2D make_mesh(const BasinGeometry& geometry, const double cell_size, const MeshFamily family) {
    const LayeredMeshLayout layout = make_layered_mesh_layout({&geometry, 1}, cell_size);
    return family == MeshFamily::fixed_rows ? make_layered_mesh(geometry, layout)
                                            : make_variable_layered_mesh(geometry, layout);
}

std::string_view name(const MeshFamily family) {
    return family == MeshFamily::fixed_rows ? "A" : "B";
}

std::string_view name(const HeatFluxScheme scheme) {
    switch (scheme) {
    case HeatFluxScheme::tpfa_nonorthogonal:
        return "tpfa_correction";
    case HeatFluxScheme::linear_multipoint:
        return "linear_mpfa";
    case HeatFluxScheme::nonlinear_multipoint_conservative:
        return "nmpfa_conservative";
    case HeatFluxScheme::nonlinear_multipoint_dmp:
        return "nmpfa_dmp";
    }
    throw std::logic_error("Unknown heat-flux scheme");
}

double slope_at(const ScalarProfile& curve, const double x) {
    const auto upper = std::upper_bound(curve.x.begin(), curve.x.end(), x);
    const std::size_t right = upper == curve.x.end()
                                  ? curve.x.size() - 1
                                  : static_cast<std::size_t>(upper - curve.x.begin());
    const std::size_t left = right - 1;
    return (curve.values[right] - curve.values[left]) / (curve.x[right] - curve.x[left]);
}

ErrorMetrics error_metrics(const Mesh2D& mesh, const Vector& solution, const auto& exact) {
    ErrorMetrics result;
    double squared_error = 0.0;
    double area = 0.0;
    for (const TQMesh::Facet* cell : mesh.cells()) {
        const double value = solution[cell->index()];
        const double difference = value - exact(cell->xy().x, cell->xy().y);
        squared_error += cell->area() * difference * difference;
        area += cell->area();
        result.linf = std::max(result.linf, std::abs(difference));
        result.minimum = std::min(result.minimum, value);
        result.maximum = std::max(result.maximum, value);
    }
    result.l2 = std::sqrt(squared_error / area);
    return result;
}

double integrated_residual_linf(const Mesh2D& mesh, const Vector& residual) {
    double result = 0.0;
    for (const TQMesh::Facet* cell : mesh.cells()) {
        result = std::max(result, std::abs(residual[cell->index()] * cell->area()));
    }
    return result;
}

MatrixSignMetrics matrix_sign_metrics(const SparseMatrix& matrix) {
    MatrixSignMetrics result;
    const auto& native = matrix.native_csc();
    for (Eigen::Index column = 0; column < native.outerSize(); ++column) {
        for (SparseMatrix::CscNativeType::InnerIterator entry(native, column); entry; ++entry) {
            if (entry.row() == entry.col() || !(entry.value() > 0.0)) {
                continue;
            }
            ++result.positive_offdiagonal_count;
            result.maximum_positive_offdiagonal =
                std::max(result.maximum_positive_offdiagonal, entry.value());
        }
    }
    return result;
}

StationarySolution solve_stationary(const Mesh2D& mesh, HeatSystem& system, Vector temperature,
                                    const HeatFluxScheme scheme) {
    Vector derivative(temperature.size());
    derivative.set_zero();
    SparseMatrix matrix(temperature.size(), temperature.size(), SparseStorageOrder::csc);
    StationarySolution result{
        .temperature = std::move(temperature), .matrix = std::move(matrix), .residual = {}};
    UmfpackLinearSolver solver;
    constexpr std::size_t maximum_iterations = 200;
    for (std::size_t iteration = 0; iteration <= maximum_iterations; ++iteration) {
        system.assemble_residual(0.0, result.temperature, derivative, result.residual);
        const double current_residual = integrated_residual_linf(mesh, result.residual);
        if (current_residual <= 1.0e-8) {
            return result;
        }
        if (iteration == maximum_iterations) {
            throw std::runtime_error("Stationary Picard iteration did not converge");
        }
        system.assemble_matrix(NonlinearMethod::picard, 0.0, result.temperature, derivative, 0.0,
                               result.matrix);
        Vector correction;
        const LinearSolveResult solve = solver.solve(
            result.matrix, -result.residual, correction,
            {.relative_tolerance = 1.0e-12, .absolute_tolerance = 1.0e-12, .max_iterations = 1});
        if (solve.status != LinearSolveStatus::converged) {
            throw std::runtime_error("UMFPACK failed in a stationary verification case");
        }
        if (scheme != HeatFluxScheme::nonlinear_multipoint_conservative &&
            scheme != HeatFluxScheme::nonlinear_multipoint_dmp) {
            result.temperature += correction;
        } else if (scheme == HeatFluxScheme::nonlinear_multipoint_dmp) {
            result.temperature += correction * 0.5;
        } else {
            Vector best_temperature = result.temperature;
            double best_residual = current_residual;
            double relaxation = 1.0;
            for (int line_search_iteration = 0; line_search_iteration < 20;
                 ++line_search_iteration) {
                Vector trial_temperature = result.temperature + (correction * relaxation);
                Vector trial_residual;
                system.assemble_residual(0.0, trial_temperature, derivative, trial_residual);
                const double trial_norm = integrated_residual_linf(mesh, trial_residual);
                if (trial_norm < best_residual) {
                    best_residual = trial_norm;
                    best_temperature = std::move(trial_temperature);
                }
                if (trial_norm <= current_residual * (1.0 - (1.0e-4 * relaxation))) {
                    break;
                }
                relaxation *= 0.5;
            }
            if (!(best_residual < current_residual)) {
                throw std::runtime_error("Stationary Picard line search did not reduce residual " +
                                         std::to_string(current_residual));
            }
            result.temperature = std::move(best_temperature);
        }
        ++result.nonlinear_iterations;
    }
    throw std::logic_error("Unreachable stationary solve state");
}

SmoothResult run_smooth_case(const MeshFamily family, const double cell_size,
                             const bool variable_conductivity, const HeatFluxScheme scheme) {
    constexpr double conductivity = 2.0;
    constexpr double amplitude = 20.0;
    constexpr double kx = std::numbers::pi / kWidth;
    constexpr double kz = std::numbers::pi / 4.0;
    const BasinGeometry geometry = make_smooth_geometry();
    Mesh2D mesh = make_mesh(geometry, cell_size, family);
    const auto exact = [](const double x, const double z) {
        return kBaseTemperature + amplitude * std::cos(kx * x) * std::sin(kz * z);
    };
    const auto gradient = [](const double x, const double z) {
        return Point2D{.x = -amplitude * kx * std::sin(kx * x) * std::sin(kz * z),
                       .z = amplitude * kz * std::cos(kx * x) * std::cos(kz * z)};
    };
    const std::size_t count = mesh.cells().size();
    std::vector<double> conductivity_field(count);
    const std::vector<double> heat_capacity(count, 1.0);
    std::vector<double> source(count);
    for (const TQMesh::Facet* cell : mesh.cells()) {
        const double x = cell->xy().x;
        const double z = cell->xy().y;
        const double local_conductivity =
            variable_conductivity ? 1.0 + 0.2 * x + 0.1 * z : conductivity;
        conductivity_field[static_cast<std::size_t>(cell->index())] = local_conductivity;
        const Point2D value_gradient = gradient(x, z);
        source[static_cast<std::size_t>(cell->index())] =
            local_conductivity * (kx * kx + kz * kz) * (exact(x, z) - kBaseTemperature) -
            (variable_conductivity ? 0.2 * value_gradient.x + 0.1 * value_gradient.z : 0.0);
    }
    const ScalarProfile& floor = geometry.boundaries.back().depth;
    HeatSystem system(mesh, conductivity_field, heat_capacity, source,
                      {.surface_temperature = [exact](const Point2D point,
                                                      double) { return exact(point.x, point.z); },
                       .basal_heat_flux =
                           [&floor, gradient, variable_conductivity](const Point2D point, double) {
                               const double slope = slope_at(floor, point.x);
                               const double normalization = std::hypot(1.0, slope);
                               const Point2D outward_normal{.x = -slope / normalization,
                                                            .z = 1.0 / normalization};
                               const Point2D value = gradient(point.x, point.z);
                               const double local_conductivity =
                                   variable_conductivity ? 1.0 + 0.2 * point.x + 0.1 * point.z
                                                         : conductivity;
                               return local_conductivity *
                                      ((value.x * outward_normal.x) + (value.z * outward_normal.z));
                           }});
    system.set_implicit_nonorthogonal_correction(true);
    system.set_flux_scheme(scheme);

    Vector temperature(static_cast<Vector::Index>(count));
    temperature.set_constant(kBaseTemperature);
    StationarySolution solution = solve_stationary(mesh, system, std::move(temperature), scheme);
    const HeatEnergyRates rates = system.energy_rates(0.0, solution.temperature);
    const HeatFluxDiagnostics flux_diagnostics = system.flux_diagnostics(0.0, solution.temperature);
    const MatrixSignMetrics matrix_sign = matrix_sign_metrics(solution.matrix);
    return {.cells = count,
            .nonzeros = solution.matrix.nonzero_count(),
            .nonlinear_iterations = solution.nonlinear_iterations,
            .positive_offdiagonal_count = matrix_sign.positive_offdiagonal_count,
            .maximum_positive_offdiagonal = matrix_sign.maximum_positive_offdiagonal,
            .maximum_stencil_size = flux_diagnostics.maximum_stencil_size,
            .internal_conservation_defect_linf = flux_diagnostics.internal_conservation_defect_linf,
            .error = error_metrics(mesh, solution.temperature, exact),
            .integrated_residual_linf = integrated_residual_linf(mesh, solution.residual),
            .global_balance = rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow -
                              rates.heat_production};
}

SmoothResult run_horizontal_contact_case(const MeshFamily family, const double cell_size,
                                         const double contrast, const HeatFluxScheme scheme) {
    constexpr double flux = 5.0;
    const BasinGeometry geometry = make_horizontal_contact_geometry();
    Mesh2D mesh = make_mesh(geometry, cell_size, family);
    const auto exact = [contrast](const double, const double z) {
        return z <= 1.0 ? kBaseTemperature + flux * z
                        : kBaseTemperature + flux + (flux / contrast) * (z - 1.0);
    };
    const std::size_t count = mesh.cells().size();
    std::vector<double> conductivity(count);
    for (const TQMesh::Facet* cell : mesh.cells()) {
        conductivity[static_cast<std::size_t>(cell->index())] = cell->color() == 1 ? 1.0 : contrast;
    }
    const std::vector<double> heat_capacity(count, 1.0);
    const std::vector<double> source(count, 0.0);
    HeatSystem system(mesh, conductivity, heat_capacity, source,
                      {.surface_temperature = [exact](const Point2D point,
                                                      double) { return exact(point.x, point.z); },
                       .basal_heat_flux = [](Point2D, double) { return flux; }});
    system.set_implicit_nonorthogonal_correction(true);
    system.set_flux_scheme(scheme);
    Vector temperature(static_cast<Vector::Index>(count));
    temperature.set_constant(kBaseTemperature);
    StationarySolution solution = solve_stationary(mesh, system, std::move(temperature), scheme);
    const HeatEnergyRates rates = system.energy_rates(0.0, solution.temperature);
    const HeatFluxDiagnostics flux_diagnostics = system.flux_diagnostics(0.0, solution.temperature);
    const MatrixSignMetrics matrix_sign = matrix_sign_metrics(solution.matrix);
    return {.cells = count,
            .nonzeros = solution.matrix.nonzero_count(),
            .nonlinear_iterations = solution.nonlinear_iterations,
            .positive_offdiagonal_count = matrix_sign.positive_offdiagonal_count,
            .maximum_positive_offdiagonal = matrix_sign.maximum_positive_offdiagonal,
            .maximum_stencil_size = flux_diagnostics.maximum_stencil_size,
            .internal_conservation_defect_linf = flux_diagnostics.internal_conservation_defect_linf,
            .error = error_metrics(mesh, solution.temperature, exact),
            .integrated_residual_linf = integrated_residual_linf(mesh, solution.residual),
            .global_balance = rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow -
                              rates.heat_production};
}

SmoothResult run_tilted_contact_case(const MeshFamily family, const double cell_size,
                                     const double contrast, const HeatFluxScheme scheme) {
    constexpr double slope = 0.3;
    constexpr double flux_scale = 5.0;
    const BasinGeometry geometry = make_tilted_contact_geometry();
    Mesh2D mesh = make_mesh(geometry, cell_size, family);
    const auto exact = [contrast](const double x, const double z) {
        const double signed_coordinate = z - (0.7 + slope * x);
        const double conductivity = signed_coordinate <= 0.0 ? 1.0 : contrast;
        return kBaseTemperature + flux_scale * signed_coordinate / conductivity;
    };
    const std::size_t count = mesh.cells().size();
    std::vector<double> conductivity(count);
    for (const TQMesh::Facet* cell : mesh.cells()) {
        conductivity[static_cast<std::size_t>(cell->index())] = cell->color() == 1 ? 1.0 : contrast;
    }
    const std::vector<double> heat_capacity(count, 1.0);
    const std::vector<double> source(count, 0.0);
    HeatSystem system(mesh, conductivity, heat_capacity, source,
                      {.surface_temperature = [exact](const Point2D point,
                                                      double) { return exact(point.x, point.z); },
                       .basal_heat_flux = [](Point2D, double) { return flux_scale; },
                       .left_heat_flux = [](Point2D, double) { return slope * flux_scale; },
                       .right_heat_flux = [](Point2D, double) { return -slope * flux_scale; }});
    system.set_implicit_nonorthogonal_correction(true);
    system.set_flux_scheme(scheme);
    Vector temperature(static_cast<Vector::Index>(count));
    temperature.set_constant(kBaseTemperature);
    StationarySolution solution = solve_stationary(mesh, system, std::move(temperature), scheme);
    const HeatEnergyRates rates = system.energy_rates(0.0, solution.temperature);
    const HeatFluxDiagnostics flux_diagnostics = system.flux_diagnostics(0.0, solution.temperature);
    const MatrixSignMetrics matrix_sign = matrix_sign_metrics(solution.matrix);
    return {.cells = count,
            .nonzeros = solution.matrix.nonzero_count(),
            .nonlinear_iterations = solution.nonlinear_iterations,
            .positive_offdiagonal_count = matrix_sign.positive_offdiagonal_count,
            .maximum_positive_offdiagonal = matrix_sign.maximum_positive_offdiagonal,
            .maximum_stencil_size = flux_diagnostics.maximum_stencil_size,
            .internal_conservation_defect_linf = flux_diagnostics.internal_conservation_defect_linf,
            .error = error_metrics(mesh, solution.temperature, exact),
            .integrated_residual_linf = integrated_residual_linf(mesh, solution.residual),
            .global_balance = rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow -
                              rates.heat_production};
}

SmoothResult run_dmp_case(const MeshFamily family, const double cell_size,
                          const HeatFluxScheme scheme) {
    constexpr double lower_bound = 280.0;
    constexpr double upper_bound = 320.0;
    const BasinGeometry geometry = make_smooth_geometry();
    Mesh2D mesh = make_mesh(geometry, cell_size, family);
    const std::size_t count = mesh.cells().size();
    const std::vector<double> conductivity(count, 2.0);
    const std::vector<double> heat_capacity(count, 1.0);
    const std::vector<double> source(count, 0.0);
    HeatSystem system(mesh, conductivity, heat_capacity, source,
                      {.surface_temperature =
                           [](const Point2D point, double) {
                               return kBaseTemperature +
                                      20.0 * std::cos(std::numbers::pi * point.x / kWidth);
                           },
                       .basal_heat_flux = [](Point2D, double) { return 0.0; }});
    system.set_implicit_nonorthogonal_correction(true);
    system.set_flux_scheme(scheme);
    Vector temperature(static_cast<Vector::Index>(count));
    temperature.set_constant(kBaseTemperature);
    StationarySolution solution = solve_stationary(mesh, system, std::move(temperature), scheme);
    ErrorMetrics values;
    for (const TQMesh::Facet* cell : mesh.cells()) {
        values.minimum = std::min(values.minimum, solution.temperature[cell->index()]);
        values.maximum = std::max(values.maximum, solution.temperature[cell->index()]);
    }
    const HeatEnergyRates rates = system.energy_rates(0.0, solution.temperature);
    const HeatFluxDiagnostics flux_diagnostics = system.flux_diagnostics(0.0, solution.temperature);
    const MatrixSignMetrics matrix_sign = matrix_sign_metrics(solution.matrix);
    return {.cells = count,
            .nonzeros = solution.matrix.nonzero_count(),
            .nonlinear_iterations = solution.nonlinear_iterations,
            .positive_offdiagonal_count = matrix_sign.positive_offdiagonal_count,
            .maximum_positive_offdiagonal = matrix_sign.maximum_positive_offdiagonal,
            .maximum_stencil_size = flux_diagnostics.maximum_stencil_size,
            .internal_conservation_defect_linf = flux_diagnostics.internal_conservation_defect_linf,
            .error = values,
            .lower_bound_violation = std::max(0.0, lower_bound - values.minimum),
            .upper_bound_violation = std::max(0.0, values.maximum - upper_bound),
            .integrated_residual_linf = integrated_residual_linf(mesh, solution.residual),
            .global_balance = rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow -
                              rates.heat_production};
}

PatchResult run_tilted_contact_patch(const MeshFamily family, const double cell_size,
                                     const double contrast, const HeatFluxScheme scheme) {
    constexpr double slope = 0.3;
    constexpr double flux_scale = 5.0;
    const BasinGeometry geometry = make_tilted_contact_geometry();
    Mesh2D mesh = make_mesh(geometry, cell_size, family);
    const auto exact = [contrast](const double x, const double z) {
        const double signed_coordinate = z - (0.7 + slope * x);
        const double conductivity = signed_coordinate <= 0.0 ? 1.0 : contrast;
        return kBaseTemperature + flux_scale * signed_coordinate / conductivity;
    };
    const std::size_t count = mesh.cells().size();
    std::vector<double> conductivity(count);
    for (const TQMesh::Facet* cell : mesh.cells()) {
        conductivity[static_cast<std::size_t>(cell->index())] = cell->color() == 1 ? 1.0 : contrast;
    }
    const std::vector<double> heat_capacity(count, 1.0);
    const std::vector<double> source(count, 0.0);
    HeatSystem system(mesh, conductivity, heat_capacity, source,
                      {.surface_temperature = [exact](const Point2D point,
                                                      double) { return exact(point.x, point.z); },
                       .basal_heat_flux = [](Point2D, double) { return flux_scale; }});
    system.set_implicit_nonorthogonal_correction(true);
    system.set_flux_scheme(scheme);
    Vector temperature(static_cast<Vector::Index>(count));
    for (const TQMesh::Facet* cell : mesh.cells()) {
        temperature[cell->index()] = exact(cell->xy().x, cell->xy().y);
    }
    Vector derivative(static_cast<Vector::Index>(count));
    derivative.set_zero();
    Vector residual;
    system.assemble_residual(0.0, temperature, derivative, residual);

    std::vector<bool> boundary_cell(count, false);
    for (const BoundaryKind kind :
         {BoundaryKind::top, BoundaryKind::bottom, BoundaryKind::left, BoundaryKind::right}) {
        for (const TQMesh::Edge* edge : mesh.boundary_edges(kind)) {
            boundary_cell[static_cast<std::size_t>(mesh.owner(*edge).index())] = true;
        }
    }
    std::vector<bool> contact_cell(count, false);
    for (const TQMesh::Edge* edge : mesh.internal_edges()) {
        const TQMesh::Facet& owner = mesh.owner(*edge);
        const TQMesh::Facet& neighbor = mesh.neighbor(*edge);
        if (owner.color() != neighbor.color()) {
            contact_cell[static_cast<std::size_t>(owner.index())] = true;
            contact_cell[static_cast<std::size_t>(neighbor.index())] = true;
        }
    }
    PatchResult result{.cells = count};
    for (const TQMesh::Facet* cell : mesh.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        if (!contact_cell[index] || boundary_cell[index]) {
            continue;
        }
        const double imbalance = std::abs(residual[cell->index()] * cell->area());
        ++result.measured_cells;
        result.integrated_residual_linf = std::max(result.integrated_residual_linf, imbalance);
        result.integrated_residual_l1 += imbalance;
    }
    return result;
}

void print_smooth(const std::string_view test, const HeatFluxScheme scheme, const MeshFamily family,
                  const double cell_size, const double parameter, const SmoothResult& result) {
    std::cout << test << ',' << name(scheme) << ',' << name(family) << ',' << cell_size << ','
              << parameter << ',' << result.cells << ',' << result.nonzeros << ','
              << result.nonlinear_iterations << ',' << result.positive_offdiagonal_count << ','
              << result.maximum_positive_offdiagonal << ',' << result.maximum_stencil_size << ','
              << result.internal_conservation_defect_linf << ',' << result.error.l2 << ','
              << result.error.linf << ',' << result.error.minimum << ',' << result.error.maximum
              << ',' << result.lower_bound_violation << ',' << result.upper_bound_violation << ','
              << result.integrated_residual_linf << ',' << result.global_balance << ",,\n";
}

void print_patch(const HeatFluxScheme scheme, const MeshFamily family, const double cell_size,
                 const double contrast, const PatchResult& result) {
    std::cout << "tilted_contact_patch," << name(scheme) << ',' << name(family) << ',' << cell_size
              << ',' << contrast << ',' << result.cells << ",,,,"
              << ",,,,"
              << ",,,,"
              << "," << result.integrated_residual_linf << ",," << result.measured_cells << ','
              << result.integrated_residual_l1 << '\n';
}

} // namespace

int main() {
    try {
        std::cout << std::setprecision(17);
        std::cout << "test,scheme,mesh,h,parameter,cells,nnz,nonlinear_iterations,"
                     "positive_offdiagonal_count,maximum_positive_offdiagonal,"
                     "maximum_stencil_size,internal_conservation_defect_linf,l2,linf,min,max,"
                     "lower_bound_violation,"
                     "upper_bound_violation,integrated_residual_linf,global_balance,"
                     "measured_cells,integrated_residual_l1\n";
        for (const HeatFluxScheme scheme :
             {HeatFluxScheme::tpfa_nonorthogonal, HeatFluxScheme::linear_multipoint,
              HeatFluxScheme::nonlinear_multipoint_conservative,
              HeatFluxScheme::nonlinear_multipoint_dmp}) {
            for (const MeshFamily family : {MeshFamily::fixed_rows, MeshFamily::variable_rows}) {
                for (const double cell_size : {0.4, 0.2, 0.1}) {
                    print_smooth("smooth_mms_constant_k", scheme, family, cell_size, 2.0,
                                 run_smooth_case(family, cell_size, false, scheme));
                    print_smooth("smooth_mms_variable_k", scheme, family, cell_size, 0.0,
                                 run_smooth_case(family, cell_size, true, scheme));
                    print_smooth("dmp", scheme, family, cell_size, 0.0,
                                 run_dmp_case(family, cell_size, scheme));
                    for (const double contrast : {100.0, 1000.0}) {
                        print_smooth(
                            "horizontal_contact", scheme, family, cell_size, contrast,
                            run_horizontal_contact_case(family, cell_size, contrast, scheme));
                        print_patch(scheme, family, cell_size, contrast,
                                    run_tilted_contact_patch(family, cell_size, contrast, scheme));
                        print_smooth("tilted_contact", scheme, family, cell_size, contrast,
                                     run_tilted_contact_case(family, cell_size, contrast, scheme));
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "discretization verification failed: " << error.what() << '\n';
        return 1;
    }
}
