#include "basin/basin_input.hpp"
#include "basin/nonmatching_column_grid.hpp"
#include "discretization/nonmatching_column_heat_system.hpp"
#include "linear/umfpack_linear_solver.hpp"
#include "physics/thermophysical_properties.hpp"

#include <Eigen/QR>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr double kSnapshotAgeMa = 5.0;
constexpr FluidThermophysicalProperties kFluid{
    .thermal_conductivity = 0.7,
    .density = 1040.0,
    .specific_heat = 4184.0,
};

struct SolveResult {
    Vector temperature;
    Vector residual;
    SparseMatrix matrix;
    std::size_t nonlinear_iterations = 0;
    double final_correction_linf = std::numeric_limits<double>::infinity();
    double assembly_seconds = 0.0;
    double linear_solve_seconds = 0.0;
};

const BasinConfiguration& configuration_at(const BasinInput& input, const double age_ma) {
    const auto configurations = input.configurations();
    const auto found = std::find_if(configurations.begin(), configurations.end(),
                                    [age_ma](const auto& value) { return value.age_ma == age_ma; });
    if (found == configurations.end()) {
        throw std::runtime_error("Requested snapshot age is absent from basin input");
    }
    return *found;
}

double integrated_profile_flux(const ScalarProfile& depth, const ScalarProfile& flux,
                               const double left, const double right) {
    std::vector<double> points{left, right};
    for (const auto* profile : {&depth, &flux}) {
        for (const double x : profile->x) {
            if (x > left && x < right) {
                points.push_back(x);
            }
        }
    }
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    double result = 0.0;
    for (std::size_t i = 1; i < points.size(); ++i) {
        const double a = points[i - 1];
        const double b = points[i];
        result += 0.5 * (flux.at(a) + flux.at(b)) * std::hypot(b - a, depth.at(b) - depth.at(a));
    }
    return result;
}

MultipointFluxScheme parse_scheme(const std::string_view value) {
    if (value == "linear_mpfa") {
        return MultipointFluxScheme::linear;
    }
    if (value == "nmpfa_conservative") {
        return MultipointFluxScheme::nonlinear_conservative;
    }
    if (value == "nmpfa_dmp") {
        return MultipointFluxScheme::nonlinear_dmp;
    }
    throw std::invalid_argument("Scheme must be linear_mpfa, nmpfa_conservative, or nmpfa_dmp");
}

SolveResult solve(NonmatchingColumnHeatSystem& system, Vector temperature,
                  const MultipointFluxScheme scheme) {
    UmfpackLinearSolver solver;
    SolveResult result{.temperature = std::move(temperature),
                       .residual = {},
                       .matrix =
                           SparseMatrix(system.size(), system.size(), SparseStorageOrder::csc)};
    constexpr std::size_t maximum_iterations = 300;
    std::vector<Eigen::VectorXd> anderson_fixed_points;
    std::vector<Eigen::VectorXd> anderson_residuals;
    const auto elapsed = [](const auto start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    for (std::size_t iteration = 0; iteration < maximum_iterations; ++iteration) {
        auto start = std::chrono::steady_clock::now();
        system.assemble_residual(result.temperature, result.residual);
        result.assembly_seconds += elapsed(start);
        const double current = result.residual.infinity_norm();
        start = std::chrono::steady_clock::now();
        system.assemble_matrix(result.temperature, result.matrix);
        result.assembly_seconds += elapsed(start);
        Vector correction;
        start = std::chrono::steady_clock::now();
        const LinearSolveResult linear = solver.solve(
            result.matrix, -result.residual, correction,
            {.relative_tolerance = 1.0e-12, .absolute_tolerance = 1.0e-12, .max_iterations = 1});
        result.linear_solve_seconds += elapsed(start);
        if (linear.status != LinearSolveStatus::converged || !correction.all_finite()) {
            throw std::runtime_error("UMFPACK failed on nonmatching fixed epoch");
        }
        result.final_correction_linf = correction.infinity_norm();
        if (iteration < 5 || (iteration + 1) % 25 == 0) {
            std::cerr << "picard iteration=" << iteration << " residual=" << current
                      << " correction=" << result.final_correction_linf << '\n';
        }
        if (current <= 1.0e-8 || result.final_correction_linf <= 1.0e-7) {
            return result;
        }
        if (scheme == MultipointFluxScheme::nonlinear_dmp) {
            constexpr double damping = 0.5;
            const Eigen::VectorXd fixed_point =
                result.temperature.native() + damping * correction.native();
            const Eigen::VectorXd fixed_point_residual = fixed_point - result.temperature.native();
            anderson_fixed_points.push_back(fixed_point);
            anderson_residuals.push_back(fixed_point_residual);
            constexpr std::size_t history_limit = 6;
            if (anderson_fixed_points.size() > history_limit) {
                anderson_fixed_points.erase(anderson_fixed_points.begin());
                anderson_residuals.erase(anderson_residuals.begin());
            }
            Eigen::VectorXd next = fixed_point;
            const std::size_t count = anderson_fixed_points.size();
            if (count >= 2) {
                const Eigen::Index depth = static_cast<Eigen::Index>(count - 1);
                Eigen::MatrixXd delta_residual(system.size(), depth);
                Eigen::MatrixXd delta_fixed_point(system.size(), depth);
                for (Eigen::Index column = 0; column < depth; ++column) {
                    const std::size_t i = static_cast<std::size_t>(column);
                    delta_residual.col(column) = anderson_residuals[i + 1] - anderson_residuals[i];
                    delta_fixed_point.col(column) =
                        anderson_fixed_points[i + 1] - anderson_fixed_points[i];
                }
                const Eigen::VectorXd coefficients =
                    delta_residual.colPivHouseholderQr().solve(anderson_residuals.back());
                const Eigen::VectorXd accelerated = fixed_point - delta_fixed_point * coefficients;
                const double ordinary_step = (fixed_point - result.temperature.native()).norm();
                const double accelerated_step = (accelerated - result.temperature.native()).norm();
                if (accelerated.allFinite() &&
                    accelerated_step <= 10.0 * std::max(1.0, ordinary_step)) {
                    next = accelerated;
                }
            }
            result.temperature.native() = std::move(next);
        } else if (scheme == MultipointFluxScheme::nonlinear_conservative) {
            Vector best = result.temperature;
            double best_residual = current;
            double relaxation = 1.0;
            for (int search = 0; search < 24; ++search) {
                Vector trial = result.temperature + correction * relaxation;
                Vector trial_residual;
                start = std::chrono::steady_clock::now();
                system.assemble_residual(trial, trial_residual);
                result.assembly_seconds += elapsed(start);
                const double trial_norm = trial_residual.infinity_norm();
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
    throw std::runtime_error("Nonmatching fixed-epoch Picard iteration did not converge");
}

std::optional<double> column_value(const NonmatchingColumnGrid& grid, const Vector& temperature,
                                   const std::size_t column, const double z,
                                   const std::function<double(Point2D)>& surface_temperature,
                                   const std::span<const double> conductivity,
                                   const std::function<double(Point2D)>& basal_flux) {
    const auto& indices = grid.columns[column];
    const NonmatchingColumnCell& first = grid.cells[indices.front()];
    const NonmatchingColumnCell& last = grid.cells[indices.back()];
    if (z < first.z_top || z > last.z_bottom) {
        return std::nullopt;
    }
    const double x = first.center().x;
    const auto center_z = [&grid](const std::size_t index) { return grid.cells[index].center().z; };
    if (z <= center_z(indices.front())) {
        const double boundary = surface_temperature({x, first.z_top});
        const double fraction = (z - first.z_top) / (center_z(indices.front()) - first.z_top);
        return std::lerp(boundary, temperature[static_cast<Vector::Index>(indices.front())],
                         fraction);
    }
    for (std::size_t i = 1; i < indices.size(); ++i) {
        const double upper_z = center_z(indices[i - 1]);
        const double lower_z = center_z(indices[i]);
        if (z <= lower_z) {
            const double fraction = (z - upper_z) / (lower_z - upper_z);
            return std::lerp(temperature[static_cast<Vector::Index>(indices[i - 1])],
                             temperature[static_cast<Vector::Index>(indices[i])], fraction);
        }
    }
    const double center = center_z(indices.back());
    const double gradient = basal_flux({x, last.z_bottom}) / conductivity[indices.back()];
    return temperature[static_cast<Vector::Index>(indices.back())] + gradient * (z - center);
}

int material_at(const NonmatchingColumnGrid& grid, const std::size_t column, const double z) {
    for (const std::size_t index : grid.columns[column]) {
        const NonmatchingColumnCell& cell = grid.cells[index];
        if (z >= cell.z_top && z <= cell.z_bottom) {
            return cell.layer_id;
        }
    }
    return -1;
}

void write_samples(const std::filesystem::path& path, const NonmatchingColumnGrid& grid,
                   const Vector& temperature,
                   const std::function<double(Point2D)>& surface_temperature,
                   const std::span<const double> conductivity,
                   const std::function<double(Point2D)>& basal_flux) {
    std::ofstream output(path);
    output << "x,z,temperature,material\n" << std::setprecision(17);
    constexpr double step = 12.5;
    for (double x = 0.0; x <= 1500.0; x += step) {
        const auto upper = std::upper_bound(grid.x.begin(), grid.x.end(), x);
        std::size_t column =
            upper == grid.x.begin() ? 0 : static_cast<std::size_t>(upper - grid.x.begin() - 1);
        column = std::min(column, grid.columns.size() - 1);
        for (double z = 0.0; z <= 4500.0; z += step) {
            const auto value = column_value(grid, temperature, column, z, surface_temperature,
                                            conductivity, basal_flux);
            output << x << ',' << z << ',';
            if (value) {
                output << *value;
            }
            output << ',';
            const int material = material_at(grid, column, z);
            if (material >= 0) {
                output << material;
            }
            output << '\n';
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 8) {
            throw std::invalid_argument(
                "usage: nonmatching_fixed_epoch_verification PHYSICAL_INPUT GEOMETRY_INPUT "
                "GEOMETRY_LABEL SCHEME H geotherm|constant OUTPUT_DIR");
        }
        const std::filesystem::path physical_path = argv[1];
        const std::filesystem::path geometry_path = argv[2];
        const std::string geometry_label = argv[3];
        const MultipointFluxScheme scheme = parse_scheme(argv[4]);
        const double cell_size = std::stod(argv[5]);
        const std::string_view initial_guess = argv[6];
        const std::filesystem::path output_directory = argv[7];
        if ((initial_guess != "geotherm" && initial_guess != "constant") ||
            !std::isfinite(cell_size) || !(cell_size > 0.0)) {
            throw std::invalid_argument("Invalid H or initial guess");
        }

        const auto wall_start = std::chrono::steady_clock::now();
        const BasinInput physical_input = BasinInput::read(physical_path);
        const BasinInput geometry_input = BasinInput::read(geometry_path);
        const BasinConfiguration& physical_configuration =
            configuration_at(physical_input, kSnapshotAgeMa);
        const BasinConfiguration& configuration = configuration_at(geometry_input, kSnapshotAgeMa);
        const BasinGeometry snapshot_geometry = make_basin_geometry(configuration);
        const LayeredMeshLayout layout =
            make_nonmatching_column_layout(snapshot_geometry, cell_size);
        const auto mesh_start = std::chrono::steady_clock::now();
        const NonmatchingColumnGrid grid = make_nonmatching_column_grid(snapshot_geometry, layout);
        const double mesh_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - mesh_start).count();

        std::vector<double> porosity(grid.cells.size());
        std::vector<const LithotypeThermophysicalProperties*> lithotypes(grid.cells.size());
        std::vector<double> conductivity(grid.cells.size());
        std::vector<double> production(grid.cells.size());
        Vector temperature(static_cast<Vector::Index>(grid.cells.size()));
        const BasinLayerProfile& top_layer = configuration.layers.front();
        for (const NonmatchingColumnCell& cell : grid.cells) {
            const Point2D center = cell.center();
            const BasinLayerProfile& layer = configuration.layer(cell.layer_id);
            porosity[cell.index] = layer.porosity_at(center.x);
            lithotypes[cell.index] = &geometry_input.lithotype(layer.lithotype_at(center.x));
            const double surface = configuration.surface_temperature.at(center.x);
            const double initial_conductivity =
                evaluate_thermophysical_properties(surface, porosity[cell.index],
                                                   *lithotypes[cell.index], kFluid)
                    .thermal_conductivity;
            temperature[static_cast<Vector::Index>(cell.index)] =
                initial_guess == "constant"
                    ? 300.0
                    : surface + configuration.basal_heat_flux.at(center.x) / initial_conductivity *
                                    (center.z - top_layer.top_at(center.x));
            const EffectiveThermophysicalProperties properties = evaluate_thermophysical_properties(
                temperature[static_cast<Vector::Index>(cell.index)], porosity[cell.index],
                *lithotypes[cell.index], kFluid);
            conductivity[cell.index] = properties.thermal_conductivity;
            production[cell.index] = properties.heat_production;
        }

        const BasinGeometry physical_geometry = make_basin_geometry(physical_configuration);
        const ScalarProfile& physical_bottom = physical_geometry.boundaries.back().depth;
        std::vector<double> column_basal_flux(grid.columns.size());
        double supplied_heat = 0.0;
        for (std::size_t column = 0; column < grid.columns.size(); ++column) {
            const double heat =
                integrated_profile_flux(physical_bottom, physical_configuration.basal_heat_flux,
                                        grid.x[column], grid.x[column + 1]);
            column_basal_flux[column] = heat / (grid.x[column + 1] - grid.x[column]);
            supplied_heat += heat;
        }
        const auto surface_temperature = [&configuration](const Point2D point) {
            return configuration.surface_temperature.at(point.x);
        };
        const auto basal_flux = [&grid, &column_basal_flux](const Point2D point) {
            const auto upper = std::upper_bound(grid.x.begin(), grid.x.end(), point.x);
            std::size_t column =
                upper == grid.x.begin() ? 0 : static_cast<std::size_t>(upper - grid.x.begin() - 1);
            column = std::min(column, column_basal_flux.size() - 1);
            return column_basal_flux[column];
        };
        auto updater = [&porosity, &lithotypes](const std::span<const double> values,
                                                const std::span<double> updated_conductivity,
                                                const std::span<double> updated_production) {
            for (std::size_t i = 0; i < values.size(); ++i) {
                const EffectiveThermophysicalProperties properties =
                    evaluate_thermophysical_properties(values[i], porosity[i], *lithotypes[i],
                                                       kFluid);
                updated_conductivity[i] = properties.thermal_conductivity;
                updated_production[i] = properties.heat_production;
            }
        };
        NonmatchingColumnHeatSystem system(
            grid, conductivity, production,
            {.surface_temperature = surface_temperature, .basal_inward_flux = basal_flux}, updater);
        system.set_flux_scheme(scheme);
        SolveResult solved = solve(system, std::move(temperature), scheme);
        const NonmatchingColumnEnergyRates rates = system.energy_rates(solved.temperature);
        const NonmatchingColumnFluxDiagnostics diagnostics =
            system.flux_diagnostics(solved.temperature);

        double minimum_subface = std::numeric_limits<double>::infinity();
        for (const NonmatchingVerticalSubface& face : grid.vertical_subfaces) {
            minimum_subface = std::min(minimum_subface, face.length());
        }
        double minimum_height = std::numeric_limits<double>::infinity();
        for (const NonmatchingColumnCell& cell : grid.cells) {
            minimum_height = std::min(minimum_height, cell.height());
        }
        const double global_balance = rates.surface_outflow - rates.basal_inflow -
                                      rates.lateral_inflow - rates.heat_production;
        const double wall_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();

        std::filesystem::create_directories(output_directory);
        write_samples(output_directory / "samples.csv", grid, solved.temperature,
                      surface_temperature, system.thermal_conductivity(), basal_flux);
        std::ofstream cells(output_directory / "cells.csv");
        cells << "index,column,layer,x_left,x_right,z_top,z_bottom,temperature\n"
              << std::setprecision(17);
        for (const NonmatchingColumnCell& cell : grid.cells) {
            cells << cell.index << ',' << cell.column << ',' << cell.layer_id << ',' << cell.x_left
                  << ',' << cell.x_right << ',' << cell.z_top << ',' << cell.z_bottom << ','
                  << solved.temperature[static_cast<Vector::Index>(cell.index)] << '\n';
        }

        std::cout << std::setprecision(17)
                  << "geometry,mesh,scheme,h,initial,cells,vertical_subfaces,nnz,"
                     "nonlinear_updates,maximum_stencil,internal_flux_mismatch,residual_linf,"
                     "correction_linf,global_balance,surface_outflow,basal_inflow,supplied_heat,"
                     "production,Tmin,Tmax,area,minimum_cell_height,minimum_subface,mesh_seconds,"
                     "assembly_seconds,linear_solve_seconds,wall_seconds\n"
                  << geometry_label << ",C," << argv[4] << ',' << cell_size << ',' << argv[6] << ','
                  << grid.cells.size() << ',' << grid.vertical_subfaces.size() << ','
                  << solved.matrix.nonzero_count() << ',' << solved.nonlinear_iterations << ','
                  << diagnostics.maximum_stencil_size << ','
                  << diagnostics.internal_conservation_defect_linf << ','
                  << solved.residual.infinity_norm() << ',' << solved.final_correction_linf << ','
                  << global_balance << ',' << rates.surface_outflow << ',' << rates.basal_inflow
                  << ',' << supplied_heat << ',' << rates.heat_production << ','
                  << solved.temperature.native().minCoeff() << ','
                  << solved.temperature.native().maxCoeff() << ',' << grid.area() << ','
                  << minimum_height << ',' << minimum_subface << ',' << mesh_seconds << ','
                  << solved.assembly_seconds << ',' << solved.linear_solve_seconds << ','
                  << wall_seconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
