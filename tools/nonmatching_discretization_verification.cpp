#include "basin/nonmatching_column_grid.hpp"
#include "discretization/nonmatching_column_heat_system.hpp"
#include "linear/umfpack_linear_solver.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr double kWidth = 4.0;
constexpr double kBaseTemperature = 300.0;

struct Solution {
    Vector temperature;
    Vector residual;
    SparseMatrix matrix;
    std::size_t nonlinear_iterations = 0;
};

struct Metrics {
    double l2 = 0.0;
    double linf = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
};

ScalarProfile profile(const std::vector<double>& x, const std::vector<double>& values) {
    return {.x = x, .values = values};
}

BasinGeometry smooth_geometry() {
    const std::vector<double> x{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    std::vector<double> top(x.size(), 0.0);
    std::vector<double> contact;
    std::vector<double> bottom(x.size(), 4.0);
    for (const double coordinate : x) {
        contact.push_back(1.7 + 0.25 * std::sin(std::numbers::pi * coordinate / 2.0));
    }
    return {.age_ma = 0.0,
            .boundaries = {{profile(x, top)}, {profile(x, contact)}, {profile(x, bottom)}},
            .layers = {{1, 0, 1, 0.0, kWidth}, {2, 1, 2, 0.0, kWidth}}};
}

BasinGeometry tilted_contact_geometry() {
    const std::vector<double> x{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    std::vector<double> top(x.size(), 0.0);
    std::vector<double> contact;
    std::vector<double> bottom(x.size(), 2.6);
    for (const double coordinate : x) {
        contact.push_back(0.7 + 0.3 * coordinate);
    }
    return {.age_ma = 0.0,
            .boundaries = {{profile(x, top)}, {profile(x, contact)}, {profile(x, bottom)}},
            .layers = {{1, 0, 1, 0.0, kWidth}, {2, 1, 2, 0.0, kWidth}}};
}

double residual_linf(const Vector& residual) {
    return residual.infinity_norm();
}

Solution solve(NonmatchingColumnHeatSystem& system, Vector temperature,
               const MultipointFluxScheme scheme) {
    UmfpackLinearSolver solver;
    Solution result{.temperature = std::move(temperature),
                    .residual = {},
                    .matrix = SparseMatrix(system.size(), system.size(), SparseStorageOrder::csc)};
    constexpr std::size_t maximum_iterations = 300;
    for (std::size_t iteration = 0; iteration < maximum_iterations; ++iteration) {
        system.assemble_residual(result.temperature, result.residual);
        const double current = residual_linf(result.residual);
        system.assemble_matrix(result.temperature, result.matrix);
        Vector correction;
        const LinearSolveResult linear = solver.solve(
            result.matrix, -result.residual, correction,
            {.relative_tolerance = 1.0e-12, .absolute_tolerance = 1.0e-12, .max_iterations = 1});
        if (linear.status != LinearSolveStatus::converged) {
            throw std::runtime_error("UMFPACK failed on nonmatching-column verification");
        }
        if (current <= 1.0e-10) {
            return result;
        }
        if (scheme == MultipointFluxScheme::nonlinear_dmp) {
            result.temperature += correction * 0.5;
        } else if (scheme == MultipointFluxScheme::nonlinear_conservative) {
            Vector best = result.temperature;
            double best_residual = current;
            double relaxation = 1.0;
            for (int search = 0; search < 24; ++search) {
                Vector trial = result.temperature + correction * relaxation;
                Vector trial_residual;
                system.assemble_residual(trial, trial_residual);
                const double trial_norm = residual_linf(trial_residual);
                if (trial_norm < best_residual) {
                    best_residual = trial_norm;
                    best = std::move(trial);
                }
                if (trial_norm <= current * (1.0 - 1.0e-4 * relaxation)) {
                    break;
                }
                relaxation *= 0.5;
            }
            if (!(best_residual < current)) {
                throw std::runtime_error("Nonmatching conservative NMPFA line search failed");
            }
            result.temperature = std::move(best);
        } else {
            result.temperature += correction;
        }
        ++result.nonlinear_iterations;
    }
    throw std::runtime_error("Nonmatching-column Picard iteration did not converge");
}

Metrics error_metrics(const NonmatchingColumnGrid& grid, const Vector& solution,
                      const auto& exact) {
    Metrics result;
    double squared = 0.0;
    double area = 0.0;
    for (const NonmatchingColumnCell& cell : grid.cells) {
        const Point2D center = cell.center();
        const double value = solution[static_cast<Vector::Index>(cell.index)];
        const double difference = value - exact(center.x, center.z, cell.layer_id);
        squared += cell.area() * difference * difference;
        area += cell.area();
        result.linf = std::max(result.linf, std::abs(difference));
        result.minimum = std::min(result.minimum, value);
        result.maximum = std::max(result.maximum, value);
    }
    result.l2 = std::sqrt(squared / area);
    return result;
}

std::string_view name(const MultipointFluxScheme scheme) {
    switch (scheme) {
    case MultipointFluxScheme::linear:
        return "linear_mpfa";
    case MultipointFluxScheme::nonlinear_conservative:
        return "nmpfa_conservative";
    case MultipointFluxScheme::nonlinear_dmp:
        return "nmpfa_dmp";
    }
    throw std::logic_error("Unknown multipoint scheme");
}

void print_case(const std::string_view test, const double h, const double parameter,
                const NonmatchingColumnGrid& grid, NonmatchingColumnHeatSystem& system,
                const MultipointFluxScheme scheme, const auto& exact, Vector initial) {
    Solution solution = solve(system, std::move(initial), scheme);
    const Metrics error = error_metrics(grid, solution.temperature, exact);
    const NonmatchingColumnEnergyRates rates = system.energy_rates(solution.temperature);
    const NonmatchingColumnFluxDiagnostics diagnostics =
        system.flux_diagnostics(solution.temperature);
    double minimum_subface = std::numeric_limits<double>::infinity();
    for (const NonmatchingVerticalSubface& face : grid.vertical_subfaces) {
        minimum_subface = std::min(minimum_subface, face.length());
    }
    std::cout << test << ',' << name(scheme) << ',' << h << ',' << parameter << ','
              << grid.cells.size() << ',' << grid.vertical_subfaces.size() << ','
              << solution.matrix.nonzero_count() << ',' << solution.nonlinear_iterations << ','
              << diagnostics.maximum_stencil_size << ','
              << diagnostics.internal_conservation_defect_linf << ',' << error.l2 << ','
              << error.linf << ',' << error.minimum << ',' << error.maximum << ','
              << residual_linf(solution.residual) << ','
              << rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow -
                     rates.heat_production
              << ',' << minimum_subface << '\n';
}

void smooth_case(const double h, const MultipointFluxScheme scheme) {
    constexpr double conductivity_value = 2.0;
    constexpr double amplitude = 20.0;
    constexpr double kx = std::numbers::pi / kWidth;
    constexpr double kz = std::numbers::pi / 4.0;
    const BasinGeometry geometry = smooth_geometry();
    const LayeredMeshLayout layout = make_nonmatching_column_layout(geometry, h);
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(geometry, layout);
    const auto exact = [](const double x, const double z, int) {
        return kBaseTemperature + amplitude * std::cos(kx * x) * std::sin(kz * z);
    };
    std::vector<double> conductivity(grid.cells.size(), conductivity_value);
    std::vector<double> source(grid.cells.size());
    for (const NonmatchingColumnCell& cell : grid.cells) {
        const Point2D center = cell.center();
        source[cell.index] = conductivity_value * (kx * kx + kz * kz) *
                             (exact(center.x, center.z, cell.layer_id) - kBaseTemperature);
    }
    NonmatchingColumnHeatSystem system(
        grid, conductivity, source,
        {.surface_temperature = [exact](const Point2D point) { return exact(point.x, point.z, 0); },
         .basal_inward_flux =
             [](const Point2D point) {
                 return conductivity_value * amplitude * kz * std::cos(kx * point.x) *
                        std::cos(kz * point.z);
             }});
    system.set_flux_scheme(scheme);
    Vector initial(static_cast<Vector::Index>(grid.cells.size()));
    initial.set_constant(kBaseTemperature);
    print_case("smooth_mms", h, 0.0, grid, system, scheme, exact, std::move(initial));
}

void tilted_contact_case(const double h, const double contrast, const MultipointFluxScheme scheme) {
    constexpr double slope = 0.3;
    constexpr double flux_scale = 5.0;
    const BasinGeometry geometry = tilted_contact_geometry();
    const LayeredMeshLayout layout = make_nonmatching_column_layout(geometry, h);
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(geometry, layout);
    const auto exact = [contrast](const double x, const double z, const int layer) {
        const double signed_coordinate = z - (0.7 + slope * x);
        const double conductivity = layer == 1 ? 1.0 : contrast;
        return kBaseTemperature + flux_scale * signed_coordinate / conductivity;
    };
    std::vector<double> conductivity(grid.cells.size());
    for (const NonmatchingColumnCell& cell : grid.cells) {
        conductivity[cell.index] = cell.layer_id == 1 ? 1.0 : contrast;
    }
    const std::vector<double> source(grid.cells.size(), 0.0);
    NonmatchingColumnHeatSystem system(
        grid, conductivity, source,
        {.surface_temperature = [exact](const Point2D point) { return exact(point.x, point.z, 1); },
         .basal_inward_flux = [](Point2D) { return flux_scale; },
         .left_inward_flux = [](Point2D) { return slope * flux_scale; },
         .right_inward_flux = [](Point2D) { return -slope * flux_scale; }});
    system.set_flux_scheme(scheme);
    Vector initial(static_cast<Vector::Index>(grid.cells.size()));
    initial.set_constant(kBaseTemperature);
    print_case("tilted_contact", h, contrast, grid, system, scheme, exact, std::move(initial));
}

} // namespace

int main() {
    try {
        std::cout << std::setprecision(17)
                  << "test,scheme,h,parameter,cells,vertical_subfaces,nnz,nonlinear_updates,"
                     "maximum_stencil,internal_flux_mismatch,l2,linf,Tmin,Tmax,residual_linf,"
                     "global_balance,minimum_subface\n";
        for (const MultipointFluxScheme scheme :
             {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative,
              MultipointFluxScheme::nonlinear_dmp}) {
            for (const double h : {0.4, 0.2, 0.1}) {
                smooth_case(h, scheme);
                tilted_contact_case(h, 100.0, scheme);
                tilted_contact_case(h, 1000.0, scheme);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
