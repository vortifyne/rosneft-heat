#include "basin/nonmatching_column_grid.hpp"
#include "discretization/nonmatching_column_heat_system.hpp"
#include "linear/umfpack_linear_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr double kWidth = 4.0;
constexpr double kDepth = 4.0;
constexpr double kBaseTemperature = 300.0;
constexpr double kAmplitude = 20.0;
constexpr double kx = std::numbers::pi / kWidth;
constexpr double kz = std::numbers::pi / kDepth;
constexpr double kNormalizedResidualTolerance = 1.0e-9;

using ExactTemperature = std::function<double(double, double, int)>;

struct Solution {
    Vector temperature;
    Vector residual;
    SparseMatrix matrix;
    std::size_t nonlinear_updates = 0;
    std::size_t residual_assemblies = 0;
    std::size_t matrix_assemblies = 0;
    double elapsed_ms = 0.0;
};

struct Metrics {
    double l2 = 0.0;
    double e2 = 0.0;
    double linf = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    double left_mean = 0.0;
    double right_mean = 0.0;
};

struct VerificationCase {
    std::string_view name;
    double parameter = 0.0;
    BasinGeometry geometry;
    ExactTemperature exact;
    std::function<double(Point2D, int)> conductivity;
    std::function<double(Point2D, int)> source;
    NonmatchingColumnBoundaryData boundary_data;
    double delta_temperature = 1.0;
    double characteristic_rate = 1.0;
    std::optional<int> interface_layer;
    double expected_interface_outflow = std::numeric_limits<double>::quiet_NaN();
};

ScalarProfile profile(const std::vector<double>& x, const std::vector<double>& values) {
    return {.x = x, .values = values};
}

std::vector<double> axis() {
    return {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
}

BasinGeometry two_layer_geometry(const std::function<double(double)>& contact,
                                 const std::function<double(double)>& top,
                                 const std::function<double(double)>& bottom) {
    const std::vector<double> x = axis();
    std::vector<double> top_values;
    std::vector<double> contact_values;
    std::vector<double> bottom_values;
    for (const double coordinate : x) {
        top_values.push_back(top(coordinate));
        contact_values.push_back(contact(coordinate));
        bottom_values.push_back(bottom(coordinate));
    }
    return {.age_ma = 0.0,
            .boundaries = {{profile(x, top_values)},
                           {profile(x, contact_values)},
                           {profile(x, bottom_values)}},
            .layers = {{1, 0, 1, 0.0, kWidth}, {2, 1, 2, 0.0, kWidth}}};
}

BasinGeometry one_layer_geometry(const std::function<double(double)>& top,
                                 const std::function<double(double)>& bottom) {
    const std::vector<double> x = axis();
    std::vector<double> top_values;
    std::vector<double> bottom_values;
    for (const double coordinate : x) {
        top_values.push_back(top(coordinate));
        bottom_values.push_back(bottom(coordinate));
    }
    return {.age_ma = 0.0,
            .boundaries = {{profile(x, top_values)}, {profile(x, bottom_values)}},
            .layers = {{1, 0, 1, 0.0, kWidth}}};
}

double smooth_temperature(const double x, const double z) {
    return kBaseTemperature + kAmplitude * std::cos(kx * x) * std::sin(kz * z);
}

double smooth_dx(const double x, const double z) {
    return -kAmplitude * kx * std::sin(kx * x) * std::sin(kz * z);
}

double smooth_dz(const double x, const double z) {
    return kAmplitude * kz * std::cos(kx * x) * std::cos(kz * z);
}

double smooth_negative_laplacian(const double x, const double z) {
    return (kx * kx + kz * kz) * (smooth_temperature(x, z) - kBaseTemperature);
}

double residual_linf(const Vector& residual) {
    return residual.infinity_norm();
}

Solution solve(NonmatchingColumnHeatSystem& system, Vector temperature,
               const double characteristic_rate) {
    UmfpackLinearSolver solver;
    Solution result{.temperature = std::move(temperature),
                    .residual = {},
                    .matrix = SparseMatrix(system.size(), system.size(), SparseStorageOrder::csc)};
    constexpr std::size_t maximum_iterations = 300;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration <= maximum_iterations; ++iteration) {
        system.assemble_residual(result.temperature, result.residual);
        ++result.residual_assemblies;
        const double current = residual_linf(result.residual);
        if (current / characteristic_rate <= kNormalizedResidualTolerance) {
            const auto end = std::chrono::steady_clock::now();
            result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            return result;
        }
        if (iteration == maximum_iterations) {
            break;
        }
        system.assemble_matrix(result.temperature, result.matrix);
        ++result.matrix_assemblies;
        Vector correction;
        const LinearSolveResult linear = solver.solve(
            result.matrix, -result.residual, correction,
            {.relative_tolerance = 1.0e-13, .absolute_tolerance = 1.0e-13, .max_iterations = 1});
        if (linear.status != LinearSolveStatus::converged) {
            throw std::runtime_error("UMFPACK failed on nonmatching-column verification");
        }
        result.temperature += correction;
        ++result.nonlinear_updates;
    }
    throw std::runtime_error("Nonmatching-column iteration did not reach residual tolerance");
}

double region_mean(const NonmatchingColumnGrid& grid, const Vector& solution, const double x_left,
                   const double x_right) {
    double integral = 0.0;
    double area = 0.0;
    for (const NonmatchingColumnCell& cell : grid.cells) {
        const double overlap =
            std::max(0.0, std::min(cell.x_right, x_right) - std::max(cell.x_left, x_left));
        const double overlap_area = overlap * cell.height();
        integral += overlap_area * solution[static_cast<Vector::Index>(cell.index)];
        area += overlap_area;
    }
    return integral / area;
}

Metrics error_metrics(const NonmatchingColumnGrid& grid, const Vector& solution,
                      const ExactTemperature& exact, const double delta_temperature) {
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
    result.e2 = result.l2 / delta_temperature;
    result.left_mean = region_mean(grid, solution, 0.0, 0.5 * kWidth);
    result.right_mean = region_mean(grid, solution, 0.5 * kWidth, kWidth);
    return result;
}

double material_interface_outflow(const NonmatchingColumnGrid& grid,
                                  const std::span<const double> conductivity,
                                  const Vector& temperature,
                                  const NonmatchingColumnBoundaryData& boundary_data,
                                  const MultipointFluxScheme scheme, const int layer) {
    const auto values = std::span<const double>(temperature.native().data(),
                                                static_cast<std::size_t>(temperature.size()));
    const NonmatchingColumnFluxes fluxes =
        NonmatchingColumnFlux(grid).compute(scheme, conductivity, values, boundary_data);
    double result = 0.0;
    for (const NonmatchingColumnInternalFlux& flux : fluxes.vertical) {
        const int left_layer = grid.cells[flux.left_cell].layer_id;
        const int right_layer = grid.cells[flux.right_cell].layer_id;
        if (left_layer == right_layer) {
            continue;
        }
        const double left_to_right = flux.left_expression.value(values);
        if (left_layer == layer) {
            result += left_to_right;
        } else if (right_layer == layer) {
            result -= left_to_right;
        }
    }
    for (const NonmatchingHorizontalFace& face : grid.horizontal_faces) {
        const int upper_layer = grid.cells[face.upper_cell].layer_id;
        const int lower_layer = grid.cells[face.lower_cell].layer_id;
        if (upper_layer == lower_layer) {
            continue;
        }
        const NonmatchingColumnCell& upper = grid.cells[face.upper_cell];
        const NonmatchingColumnCell& lower = grid.cells[face.lower_cell];
        const double upper_distance = face.z - upper.center().z;
        const double lower_distance = lower.center().z - face.z;
        const double transmissibility =
            face.length / ((upper_distance / conductivity[face.upper_cell]) +
                           (lower_distance / conductivity[face.lower_cell]));
        const double upper_to_lower =
            transmissibility * (temperature[static_cast<Vector::Index>(face.upper_cell)] -
                                temperature[static_cast<Vector::Index>(face.lower_cell)]);
        if (upper_layer == layer) {
            result += upper_to_lower;
        } else if (lower_layer == layer) {
            result -= upper_to_lower;
        }
    }
    return result;
}

std::string_view name(const MultipointFluxScheme scheme) {
    switch (scheme) {
    case MultipointFluxScheme::linear:
        return "linear_harmonic_baseline";
    case MultipointFluxScheme::nonlinear_conservative:
        return "nmpfa_conservative";
    }
    throw std::logic_error("Unknown multipoint scheme");
}

VerificationCase smooth_constant_case() {
    constexpr double conductivity_value = 2.0;
    VerificationCase result;
    result.name = "smooth_constant_lambda";
    result.geometry = two_layer_geometry(
        [](const double x) { return 1.7 + 0.25 * std::sin(std::numbers::pi * x / 2.0); },
        [](double) { return 0.0; }, [](double) { return kDepth; });
    result.exact = [](const double x, const double z, int) { return smooth_temperature(x, z); };
    result.conductivity = [](Point2D, int) { return conductivity_value; };
    result.source = [](const Point2D point, int) {
        return conductivity_value * smooth_negative_laplacian(point.x, point.z);
    };
    result.boundary_data = {
        .surface_temperature =
            [](const Point2D point) { return smooth_temperature(point.x, point.z); },
        .basal_inward_flux =
            [](const Point2D point) { return conductivity_value * smooth_dz(point.x, point.z); }};
    result.delta_temperature = 2.0 * kAmplitude;
    result.characteristic_rate = 1000.0;
    return result;
}

VerificationCase smooth_variable_case() {
    constexpr double grad_k_x = 0.15;
    constexpr double grad_k_z = 0.10;
    const auto conductivity = [](const Point2D point, int) {
        return 1.0 + grad_k_x * point.x + grad_k_z * point.z;
    };
    VerificationCase result;
    result.name = "smooth_variable_lambda";
    result.geometry = one_layer_geometry([](double) { return 0.0; }, [](double) { return kDepth; });
    result.exact = [](const double x, const double z, int) { return smooth_temperature(x, z); };
    result.conductivity = conductivity;
    result.source = [conductivity](const Point2D point, int layer) {
        return conductivity(point, layer) * smooth_negative_laplacian(point.x, point.z) -
               grad_k_x * smooth_dx(point.x, point.z) - grad_k_z * smooth_dz(point.x, point.z);
    };
    result.boundary_data = {
        .surface_temperature =
            [](const Point2D point) { return smooth_temperature(point.x, point.z); },
        .basal_inward_flux =
            [conductivity](const Point2D point) {
                return conductivity(point, 1) * smooth_dz(point.x, point.z);
            }};
    result.delta_temperature = 2.0 * kAmplitude;
    result.characteristic_rate = 1000.0;
    return result;
}

VerificationCase sloped_surface_case() {
    constexpr double conductivity_value = 2.0;
    VerificationCase result;
    result.name = "smooth_sloped_surface";
    result.geometry = one_layer_geometry([](const double x) { return 0.25 + 0.15 * x; },
                                         [](double) { return kDepth; });
    result.exact = [](const double x, const double z, int) { return smooth_temperature(x, z); };
    result.conductivity = [](Point2D, int) { return conductivity_value; };
    result.source = [](const Point2D point, int) {
        return conductivity_value * smooth_negative_laplacian(point.x, point.z);
    };
    result.boundary_data = {
        .surface_temperature =
            [](const Point2D point) { return smooth_temperature(point.x, point.z); },
        .basal_inward_flux =
            [](const Point2D point) { return conductivity_value * smooth_dz(point.x, point.z); }};
    result.delta_temperature = 2.0 * kAmplitude;
    result.characteristic_rate = 1000.0;
    return result;
}

VerificationCase horizontal_contact_case(const double contrast) {
    constexpr double contact = 1.35;
    constexpr double flux_scale = 5.0;
    VerificationCase result;
    result.name = "horizontal_contact";
    result.parameter = contrast;
    result.geometry = two_layer_geometry([](double) { return contact; }, [](double) { return 0.0; },
                                         [](double) { return kDepth; });
    result.exact = [contrast](const double, const double z, const int layer) {
        const double conductivity = layer == 1 ? 1.0 : contrast;
        return kBaseTemperature + flux_scale * (z - contact) / conductivity;
    };
    result.conductivity = [contrast](Point2D, const int layer) {
        return layer == 1 ? 1.0 : contrast;
    };
    result.source = [](Point2D, int) { return 0.0; };
    result.boundary_data = {
        .surface_temperature =
            [exact = result.exact](const Point2D point) { return exact(point.x, point.z, 1); },
        .basal_inward_flux = [](Point2D) { return flux_scale; }};
    result.delta_temperature = 20.0;
    result.characteristic_rate = flux_scale * kWidth;
    result.interface_layer = 1;
    result.expected_interface_outflow = -flux_scale * kWidth;
    return result;
}

VerificationCase tilted_contact_case(const double contrast) {
    constexpr double intercept = 0.7;
    constexpr double slope = 0.3;
    constexpr double flux_scale = 5.0;
    VerificationCase result;
    result.name = "tilted_contact";
    result.parameter = contrast;
    result.geometry = two_layer_geometry([](const double x) { return intercept + slope * x; },
                                         [](double) { return 0.0; }, [](double) { return 2.6; });
    result.exact = [contrast](const double x, const double z, const int layer) {
        const double signed_coordinate = z - (intercept + slope * x);
        const double conductivity = layer == 1 ? 1.0 : contrast;
        return kBaseTemperature + flux_scale * signed_coordinate / conductivity;
    };
    result.conductivity = [contrast](Point2D, const int layer) {
        return layer == 1 ? 1.0 : contrast;
    };
    result.source = [](Point2D, int) { return 0.0; };
    result.boundary_data = {
        .surface_temperature =
            [exact = result.exact](const Point2D point) { return exact(point.x, point.z, 1); },
        .basal_inward_flux = [](Point2D) { return flux_scale; },
        .left_inward_flux = [](Point2D) { return slope * flux_scale; },
        .right_inward_flux = [](Point2D) { return -slope * flux_scale; }};
    result.delta_temperature = 20.0;
    result.characteristic_rate = 2.0 * flux_scale * kWidth;
    result.interface_layer = 1;
    result.expected_interface_outflow = -flux_scale * (1.0 + slope * slope) * kWidth;
    return result;
}

void print_case(const VerificationCase& definition, const double hx, const double hz,
                const double phase, const MultipointFluxScheme scheme,
                const double initial_offset) {
    const LayeredMeshLayout layout =
        make_nonmatching_column_layout(definition.geometry, hx, hz, phase);
    const NonmatchingColumnGrid grid = make_nonmatching_column_grid(definition.geometry, layout);
    std::vector<double> conductivity(grid.cells.size());
    std::vector<double> source(grid.cells.size());
    Vector initial(static_cast<Vector::Index>(grid.cells.size()));
    initial.set_constant(kBaseTemperature + initial_offset);
    for (const NonmatchingColumnCell& cell : grid.cells) {
        conductivity[cell.index] = definition.conductivity(cell.center(), cell.layer_id);
        source[cell.index] = definition.source(cell.center(), cell.layer_id);
    }
    NonmatchingColumnHeatSystem system(grid, conductivity, source, definition.boundary_data);
    system.set_flux_scheme(scheme);
    Solution solution = solve(system, std::move(initial), definition.characteristic_rate);
    const Metrics error =
        error_metrics(grid, solution.temperature, definition.exact, definition.delta_temperature);
    const NonmatchingColumnEnergyRates rates = system.energy_rates(solution.temperature);
    const NonmatchingColumnFluxDiagnostics diagnostics =
        system.flux_diagnostics(solution.temperature);
    const double global_balance =
        rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow - rates.heat_production;
    double minimum_subface = std::numeric_limits<double>::infinity();
    double maximum_dx = 0.0;
    double maximum_dz = 0.0;
    for (const NonmatchingColumnCell& cell : grid.cells) {
        maximum_dx = std::max(maximum_dx, cell.width());
        maximum_dz = std::max(maximum_dz, cell.height());
    }
    for (const NonmatchingVerticalSubface& face : grid.vertical_subfaces) {
        minimum_subface = std::min(minimum_subface, face.length());
    }
    const double interface_outflow =
        definition.interface_layer
            ? material_interface_outflow(grid, system.thermal_conductivity(), solution.temperature,
                                         definition.boundary_data, scheme,
                                         *definition.interface_layer)
            : std::numeric_limits<double>::quiet_NaN();
    std::cout << definition.name << ',' << name(scheme) << ',' << initial_offset << ',' << hx << ','
              << hz << ',' << phase << ',' << definition.parameter << ',' << grid.columns.size()
              << ',' << grid.cells.size() << ',' << grid.vertical_subfaces.size() << ','
              << grid.surface_vertical_faces.size() << ',' << grid.basal_vertical_faces.size()
              << ',' << solution.matrix.nonzero_count() << ',' << solution.nonlinear_updates << ','
              << solution.residual_assemblies << ',' << solution.matrix_assemblies << ','
              << solution.elapsed_ms << ',' << diagnostics.maximum_stencil_size << ','
              << diagnostics.internal_conservation_defect_linf << ',' << error.e2 << ',' << error.l2
              << ',' << error.linf << ',' << error.minimum << ',' << error.maximum << ','
              << error.left_mean << ',' << error.right_mean << ','
              << residual_linf(solution.residual) << ','
              << residual_linf(solution.residual) / definition.characteristic_rate << ','
              << global_balance << ',' << global_balance / definition.characteristic_rate << ','
              << rates.surface_outflow << ',' << rates.basal_inflow << ',' << rates.lateral_inflow
              << ',' << rates.heat_production << ',' << interface_outflow << ','
              << definition.expected_interface_outflow << ',' << minimum_subface << ','
              << maximum_dx << ',' << maximum_dz << '\n';
}

std::vector<VerificationCase> cases() {
    std::vector<VerificationCase> result;
    result.push_back(smooth_constant_case());
    result.push_back(smooth_variable_case());
    result.push_back(sloped_surface_case());
    result.push_back(horizontal_contact_case(100.0));
    result.push_back(horizontal_contact_case(1000.0));
    result.push_back(tilted_contact_case(100.0));
    result.push_back(tilted_contact_case(1000.0));
    return result;
}

void run_setting(const double hx, const double hz, const double phase,
                 const double initial_offset) {
    for (const auto scheme :
         {MultipointFluxScheme::linear, MultipointFluxScheme::nonlinear_conservative}) {
        for (const auto& definition : cases()) {
            print_case(definition, hx, hz, phase, scheme, initial_offset);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(17)
                  << "test,scheme,initial_offset,hx,hz,phase,parameter,columns,cells,"
                     "vertical_subfaces,"
                     "surface_steps,basal_steps,nnz,nonlinear_updates,residual_assemblies,"
                     "matrix_assemblies,elapsed_ms,maximum_stencil,internal_flux_mismatch,E2,l2,"
                     "linf,Tmin,Tmax,left_mean,right_mean,residual_linf,residual_normalized,"
                     "global_balance,balance_normalized,surface_outflow,basal_inflow,"
                     "lateral_inflow,production,material_interface_outflow,"
                     "expected_material_interface_outflow,minimum_subface,max_dx,max_dz\n";
        if (argc == 4 || argc == 5) {
            const double initial_offset = argc == 5 ? std::stod(argv[4]) : 0.0;
            run_setting(std::stod(argv[1]), std::stod(argv[2]), std::stod(argv[3]), initial_offset);
        } else if (argc == 1) {
            for (const double h : {0.4, 0.2, 0.1, 0.05})
                run_setting(h, h, 0.0, 0.0);
        } else {
            throw std::invalid_argument(
                "usage: nonmatching_discretization_verification [HX HZ PHASE [INITIAL_OFFSET]]");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
