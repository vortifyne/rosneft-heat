#include "basin/discretization/nonmatching_column_heat_system.hpp"
#include "basin/geometry/basin_geometry.hpp"
#include "basin/mesh/column_test_utils.hpp"
#include "basin/mesh/nonmatching_column_grid.hpp"
#include "basin/postprocessing/energy_rates.hpp"
#include "numerics/linear/umfpack_linear_solver.hpp"
#include "numerics/nonlinear/nonlinear_solver.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

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
};

struct Metrics {
    double l2 = 0.0;
    double e2 = 0.0;
    double linf = 0.0;
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
    return {.x = x,
            .boundaries = {top_values, contact_values, bottom_values},
            .layers = {{1, 0.0, kWidth}, {2, 0.0, kWidth}}};
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
    return {.x = x, .boundaries = {top_values, bottom_values}, .layers = {{1, 0.0, kWidth}}};
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
    for (std::size_t iteration = 0; iteration <= maximum_iterations; ++iteration) {
        system.assemble_residual(result.temperature, result.residual);
        const double current = residual_linf(result.residual);
        if (current / characteristic_rate <= kNormalizedResidualTolerance) {
            return result;
        }
        if (iteration == maximum_iterations) {
            break;
        }
        system.assemble_matrix(result.temperature, result.matrix);
        Vector correction;
        const LinearSolveResult linear = solver.solve(
            result.matrix, -result.residual, correction,
            {.relative_tolerance = 1.0e-13, .absolute_tolerance = 1.0e-13, .max_iterations = 1});
        if (linear.status != LinearSolveStatus::converged) {
            throw std::runtime_error("UMFPACK failed on nonmatching-column verification");
        }
        result.temperature += correction;
    }
    throw std::runtime_error("Nonmatching-column iteration did not reach residual tolerance");
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
    }
    result.l2 = std::sqrt(squared / area);
    result.e2 = result.l2 / delta_temperature;
    return result;
}

double material_interface_outflow(const NonmatchingColumnGrid& grid,
                                  const std::span<const double> conductivity,
                                  const Vector& temperature,
                                  const NonmatchingColumnBoundaryData& boundary_data,
                                  const MultipointFluxScheme scheme, const int layer) {
    const auto values = std::span<const double>(temperature.native().data(),
                                                static_cast<std::size_t>(temperature.size()));
    NonmatchingColumnFluxStencil stencil;
    double result = 0.0;
    for (const auto& face : grid.vertical_subfaces) {
        const int left_layer = grid.cells[face.left_cell].layer_id;
        const int right_layer = grid.cells[face.right_cell].layer_id;
        if (left_layer == right_layer) {
            continue;
        }
        compute_internal_flux(grid, face, scheme, conductivity, values, boundary_data, stencil);
        const double left_to_right = evaluate_flux(stencil, values);
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

std::vector<VerificationCase> cases() {
    return {smooth_constant_case(),          smooth_variable_case(),
            sloped_surface_case(),           horizontal_contact_case(100.0),
            horizontal_contact_case(1000.0), tilted_contact_case(100.0),
            tilted_contact_case(1000.0)};
}

Metrics check_case(const VerificationCase& definition, double h, MultipointFluxScheme scheme) {
    const auto layout = make_column_axis(definition.geometry.x, h);
    const auto grid = make_nonmatching_column_grid(definition.geometry, layout, h);
    std::vector<double> conductivity(grid.cells.size()), source(grid.cells.size());
    for (const auto& cell : grid.cells) {
        conductivity[cell.index] = definition.conductivity(cell.center(), cell.layer_id);
        source[cell.index] = definition.source(cell.center(), cell.layer_id);
    }
    NonmatchingColumnHeatSystem system(grid, conductivity, source, definition.boundary_data);
    system.set_flux_scheme(scheme);
    Vector initial(static_cast<Vector::Index>(grid.cells.size()));
    initial.set_constant(kBaseTemperature);
    const auto solution = solve(system, std::move(initial), definition.characteristic_rate);
    EXPECT_LE(residual_linf(solution.residual) / definition.characteristic_rate,
              kNormalizedResidualTolerance);
    const auto rates = nonmatching_column_energy_rates(
        grid, {solution.temperature.native().data(), grid.cells.size()}, conductivity, source,
        definition.boundary_data);
    const double balance =
        rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow - rates.heat_production;
    EXPECT_LT(std::abs(balance) / definition.characteristic_rate, 1.0e-9);
    if (definition.interface_layer) {
        const double outflow = material_interface_outflow(grid, conductivity, solution.temperature,
                                                          definition.boundary_data, scheme,
                                                          *definition.interface_layer);
        const double tolerance = definition.name == "horizontal_contact"
                                     ? 1.0e-8
                                     : 0.01 * definition.characteristic_rate;
        EXPECT_NEAR(outflow, definition.expected_interface_outflow, tolerance);
    }
    return error_metrics(grid, solution.temperature, definition.exact,
                         definition.delta_temperature);
}

class NonmatchingColumnAccuracy
    : public testing::TestWithParam<std::tuple<int, MultipointFluxScheme>> {};

TEST_P(NonmatchingColumnAccuracy, RefinesToKnownSolutionAndConservesEnergy) {
    const auto [index, scheme] = GetParam();
    const auto definition = cases()[static_cast<std::size_t>(index)];
    SCOPED_TRACE(std::string(definition.name) +
                 " contrast=" + std::to_string(definition.parameter));
    const auto coarse = check_case(definition, 0.4, scheme);
    const auto fine = check_case(definition, 0.2, scheme);
    if (definition.name == "horizontal_contact") {
        EXPECT_LT(coarse.linf, 1.0e-8);
        EXPECT_LT(fine.linf, 1.0e-8);
    } else {
        EXPECT_LT(fine.l2, coarse.l2);
        EXPECT_LT(fine.e2, 0.01);
    }
}

std::string
accuracy_case_name(const testing::TestParamInfo<NonmatchingColumnAccuracy::ParamType>& info) {
    const auto [index, scheme] = info.param;
    const auto definition = cases()[static_cast<std::size_t>(index)];
    return std::string(definition.name) + "_" +
           std::to_string(static_cast<int>(definition.parameter)) +
           (scheme == MultipointFluxScheme::linear ? "_MPFA" : "_NMPFA");
}

INSTANTIATE_TEST_SUITE_P(
    MpfaAndNmpfa, NonmatchingColumnAccuracy,
    testing::Combine(testing::Range(0, 7),
                     testing::Values(MultipointFluxScheme::linear,
                                     MultipointFluxScheme::nonlinear_conservative)),
    accuracy_case_name);

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
    const NonmatchingColumnBoundaryData boundary_data{
        .surface_temperature = [](Point2D) { return kBaseTemperature; },
        .basal_inward_flux = [](Point2D) { return 0.0; }};
    NonmatchingColumnHeatSystem system(grid, coefficients, source, boundary_data);
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
    const auto rates =
        nonmatching_column_energy_rates(grid, {temperature.native().data(), grid.cells.size()},
                                        coefficients, source, boundary_data);
    EXPECT_NEAR(rates.surface_outflow - rates.basal_inflow - rates.lateral_inflow,
                rates.heat_production, 1e-8);
    return area_weighted_error(grid, {temperature.native().data(), grid.cells.size()}, exact);
}

TEST(NonmatchingColumnAccuracy, HasSecondOrderSpatialConvergence) {
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

} // namespace
