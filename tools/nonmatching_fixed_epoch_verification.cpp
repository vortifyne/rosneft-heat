#include "basin/basin_input.hpp"
#include "basin/nonmatching_column_grid.hpp"
#include "discretization/nonmatching_column_heat_system.hpp"
#include "discretization/nonmatching_column_hfv_system.hpp"
#include "linear/petsc_krylov_linear_solver.hpp"
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
#include <memory>
#include <petscsys.h>
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
    std::size_t total_linear_iterations = 0;
    std::size_t maximum_linear_iterations = 0;
    double final_correction_linf = std::numeric_limits<double>::infinity();
    double assembly_seconds = 0.0;
    double linear_solve_seconds = 0.0;
};

enum class AxisMode {
    retained_lines,
    support_only
};
enum class SurfaceStepPolicy {
    dirichlet,
    omit
};
enum class NonlinearStrategy {
    residual_line_search,
    picard
};

struct VerificationScheme {
    bool hfv = false;
    MultipointFluxScheme multipoint = MultipointFluxScheme::linear;
};

class PetscSession {
public:
    PetscSession() {
        PetscBool initialized = PETSC_FALSE;
        if (PetscInitialized(&initialized) != PETSC_SUCCESS) {
            throw std::runtime_error("Failed to query PETSc initialization state");
        }
        if (!initialized) {
            if (PetscInitializeNoArguments() != PETSC_SUCCESS) {
                throw std::runtime_error("Failed to initialize PETSc");
            }
            owns_session_ = true;
        }
    }

    ~PetscSession() {
        if (owns_session_) {
            PetscFinalize();
        }
    }

    PetscSession(const PetscSession&) = delete;
    PetscSession& operator=(const PetscSession&) = delete;

private:
    bool owns_session_ = false;
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

VerificationScheme parse_scheme(const std::string_view value) {
    if (value == "linear_mpfa") {
        return {.hfv = false, .multipoint = MultipointFluxScheme::linear};
    }
    if (value == "linear_equal_weight") {
        return {.hfv = false, .multipoint = MultipointFluxScheme::linear_equal_weight};
    }
    if (value == "nmpfa_conservative") {
        return {.hfv = false, .multipoint = MultipointFluxScheme::nonlinear_conservative};
    }
    if (value == "nmpfa_dmp") {
        return {.hfv = false, .multipoint = MultipointFluxScheme::nonlinear_dmp};
    }
    if (value == "sushi_hfv") {
        return {.hfv = true, .multipoint = MultipointFluxScheme::linear};
    }
    throw std::invalid_argument(
        "Scheme must be linear_mpfa, linear_equal_weight, nmpfa_conservative, nmpfa_dmp, "
        "or sushi_hfv");
}

AxisMode parse_axis_mode(const std::string_view value) {
    if (value == "retained") {
        return AxisMode::retained_lines;
    }
    if (value == "support_only") {
        return AxisMode::support_only;
    }
    throw std::invalid_argument("Axis mode must be retained or support_only");
}

SurfaceStepPolicy parse_surface_policy(const std::string_view value) {
    if (value == "dirichlet_steps") {
        return SurfaceStepPolicy::dirichlet;
    }
    if (value == "omit_steps") {
        return SurfaceStepPolicy::omit;
    }
    throw std::invalid_argument("Surface policy must be dirichlet_steps or omit_steps");
}

NonlinearStrategy parse_nonlinear_strategy(const std::string_view value) {
    if (value == "line_search") {
        return NonlinearStrategy::residual_line_search;
    }
    if (value == "picard") {
        return NonlinearStrategy::picard;
    }
    throw std::invalid_argument("Nonlinear strategy must be line_search or picard");
}

std::unique_ptr<LinearSolver> make_linear_solver(const std::string_view value) {
    if (value == "umfpack") {
        return std::make_unique<UmfpackLinearSolver>();
    }
    if (value == "petsc_gmres_ilu2") {
        return std::make_unique<PetscKrylovLinearSolver>(PetscKrylovMethod::gmres,
                                                         PetscPreconditioner::ilu2, false);
    }
    if (value == "petsc_gmres_gamg") {
        return std::make_unique<PetscKrylovLinearSolver>(PetscKrylovMethod::gmres,
                                                         PetscPreconditioner::gamg, false);
    }
    if (value == "petsc_gmres_gamg_reuse") {
        return std::make_unique<PetscKrylovLinearSolver>(PetscKrylovMethod::gmres,
                                                         PetscPreconditioner::gamg, true);
    }
    if (value == "petsc_gmres_boomeramg") {
        return std::make_unique<PetscKrylovLinearSolver>(PetscKrylovMethod::gmres,
                                                         PetscPreconditioner::boomeramg, false);
    }
    if (value == "petsc_gmres_boomeramg_reuse") {
        return std::make_unique<PetscKrylovLinearSolver>(PetscKrylovMethod::gmres,
                                                         PetscPreconditioner::boomeramg, true);
    }
    throw std::invalid_argument("Unknown linear solver for fixed-epoch verification");
}

LayeredMeshLayout support_only_layout(const BasinGeometry& geometry, const double hx,
                                      const double hz, const double phase) {
    std::vector<double> events;
    for (const BasinLayerRegion& region : geometry.layers) {
        events.push_back(region.x_min);
        events.push_back(region.x_max);
    }
    std::sort(events.begin(), events.end());
    events.erase(std::unique(events.begin(), events.end()), events.end());
    LayeredMeshLayout result{.x = {}, .cell_size = hz};
    for (std::size_t interval = 1; interval < events.size(); ++interval) {
        const double left = events[interval - 1];
        const double right = events[interval];
        const std::size_t count =
            std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil((right - left) / hx)));
        result.x.push_back(left);
        for (std::size_t part = 1; part < count; ++part) {
            result.x.push_back(std::lerp(
                left, right, (static_cast<double>(part) + phase) / static_cast<double>(count)));
        }
    }
    result.x.push_back(events.back());
    return result;
}

SolveResult solve(NonmatchingColumnHeatSystem& system, Vector temperature,
                  const MultipointFluxScheme scheme, const NonlinearStrategy strategy,
                  LinearSolver& solver) {
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
        if (current <= 1.0e-8) {
            return result;
        }
        start = std::chrono::steady_clock::now();
        system.assemble_matrix(result.temperature, result.matrix);
        result.matrix.make_compressed();
        result.matrix.convert_to(solver.required_storage_order());
        result.assembly_seconds += elapsed(start);
        Vector correction;
        start = std::chrono::steady_clock::now();
        const bool direct = solver.type() == LinearSolverType::direct;
        const LinearSolveResult linear =
            solver.solve(result.matrix, -result.residual, correction,
                         {.relative_tolerance = direct ? 1.0e-12 : 1.0e-10,
                          .absolute_tolerance = 1.0e-12,
                          .max_iterations = direct ? 1 : 5000});
        result.linear_solve_seconds += elapsed(start);
        result.total_linear_iterations += static_cast<std::size_t>(std::max(0, linear.iterations));
        result.maximum_linear_iterations =
            std::max(result.maximum_linear_iterations,
                     static_cast<std::size_t>(std::max(0, linear.iterations)));
        if (linear.status != LinearSolveStatus::converged || !correction.all_finite()) {
            throw std::runtime_error("UMFPACK failed on nonmatching fixed epoch");
        }
        result.final_correction_linf = correction.infinity_norm();
        if (iteration < 5 || (iteration + 1) % 25 == 0) {
            std::cerr << "picard iteration=" << iteration << " residual=" << current
                      << " correction=" << result.final_correction_linf << '\n';
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
        } else if (scheme == MultipointFluxScheme::nonlinear_conservative &&
                   strategy == NonlinearStrategy::residual_line_search) {
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

SolveResult solve_hfv(NonmatchingColumnHfvSystem& system, Vector temperature,
                      LinearSolver& solver) {
    SolveResult result{.temperature = std::move(temperature),
                       .residual = {},
                       .matrix =
                           SparseMatrix(system.size(), system.size(), SparseStorageOrder::csc)};
    constexpr std::size_t maximum_iterations = 300;
    const auto elapsed = [](const auto start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    for (std::size_t iteration = 0; iteration < maximum_iterations; ++iteration) {
        auto start = std::chrono::steady_clock::now();
        system.assemble_residual(result.temperature, result.residual);
        result.assembly_seconds += elapsed(start);
        const double current = result.residual.infinity_norm();
        if (current <= 1.0e-8) {
            return result;
        }
        Vector rhs;
        start = std::chrono::steady_clock::now();
        system.assemble_system(result.temperature, result.matrix, rhs);
        result.matrix.make_compressed();
        result.matrix.convert_to(solver.required_storage_order());
        result.assembly_seconds += elapsed(start);
        Vector next;
        start = std::chrono::steady_clock::now();
        const bool direct = solver.type() == LinearSolverType::direct;
        const LinearSolveResult linear =
            solver.solve(result.matrix, rhs, next,
                         {.relative_tolerance = direct ? 1.0e-12 : 1.0e-10,
                          .absolute_tolerance = 1.0e-12,
                          .max_iterations = direct ? 1 : 5000});
        result.linear_solve_seconds += elapsed(start);
        result.total_linear_iterations += static_cast<std::size_t>(std::max(0, linear.iterations));
        result.maximum_linear_iterations =
            std::max(result.maximum_linear_iterations,
                     static_cast<std::size_t>(std::max(0, linear.iterations)));
        if (linear.status != LinearSolveStatus::converged || !next.all_finite()) {
            throw std::runtime_error("UMFPACK failed on nonmatching HFV fixed epoch");
        }
        result.final_correction_linf = (next - result.temperature).infinity_norm();
        if (iteration < 5 || (iteration + 1) % 25 == 0) {
            std::cerr << "hfv picard iteration=" << iteration << " residual=" << current
                      << " correction=" << result.final_correction_linf << '\n';
        }
        result.temperature = std::move(next);
        ++result.nonlinear_iterations;
    }
    throw std::runtime_error("Nonmatching HFV fixed epoch did not converge");
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

double volume_weighted_mean(const NonmatchingColumnGrid& grid, const Vector& temperature,
                            const double x_min, const double x_max) {
    double integral = 0.0;
    double area = 0.0;
    for (const NonmatchingColumnCell& cell : grid.cells) {
        const double width =
            std::max(0.0, std::min(cell.x_right, x_max) - std::max(cell.x_left, x_min));
        const double weight = width * cell.height();
        integral += weight * temperature[static_cast<Vector::Index>(cell.index)];
        area += weight;
    }
    if (!(area > 0.0)) {
        throw std::runtime_error("Requested an empty region for a temperature mean");
    }
    return integral / area;
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
        PetscSession petsc_session;
        if (argc != 8 && (argc < 10 || argc > 14)) {
            throw std::invalid_argument(
                "usage: nonmatching_fixed_epoch_verification PHYSICAL_INPUT GEOMETRY_INPUT "
                "GEOMETRY_LABEL SCHEME H geotherm|constant OUTPUT_DIR\n"
                "   or: nonmatching_fixed_epoch_verification PHYSICAL_INPUT GEOMETRY_INPUT "
                "GEOMETRY_LABEL SCHEME HX HZ PHASE geotherm|constant OUTPUT_DIR "
                "[retained|support_only [dirichlet_steps|omit_steps "
                "[line_search|picard [LINEAR_SOLVER]]]]");
        }
        const std::filesystem::path physical_path = argv[1];
        const std::filesystem::path geometry_path = argv[2];
        const std::string geometry_label = argv[3];
        const VerificationScheme scheme = parse_scheme(argv[4]);
        const double horizontal_cell_size = std::stod(argv[5]);
        const double vertical_cell_size = argc >= 10 ? std::stod(argv[6]) : horizontal_cell_size;
        const double phase = argc >= 10 ? std::stod(argv[7]) : 0.0;
        const std::string_view initial_guess = argc >= 10 ? argv[8] : argv[6];
        const std::filesystem::path output_directory = argc >= 10 ? argv[9] : argv[7];
        const AxisMode axis_mode =
            argc >= 11 ? parse_axis_mode(argv[10]) : AxisMode::retained_lines;
        const SurfaceStepPolicy surface_policy =
            argc >= 12 ? parse_surface_policy(argv[11]) : SurfaceStepPolicy::dirichlet;
        const NonlinearStrategy nonlinear_strategy = argc >= 13
                                                         ? parse_nonlinear_strategy(argv[12])
                                                         : NonlinearStrategy::residual_line_search;
        const std::string_view linear_solver_name = argc >= 14 ? argv[13] : "umfpack";
        std::unique_ptr<LinearSolver> linear_solver = make_linear_solver(linear_solver_name);
        if ((initial_guess != "geotherm" && initial_guess != "constant") ||
            !std::isfinite(horizontal_cell_size) || !(horizontal_cell_size > 0.0) ||
            !std::isfinite(vertical_cell_size) || !(vertical_cell_size > 0.0) ||
            !std::isfinite(phase) || std::abs(phase) > 0.45) {
            throw std::invalid_argument("Invalid mesh step, phase, or initial guess");
        }
        if (scheme.hfv && nonlinear_strategy != NonlinearStrategy::picard) {
            throw std::invalid_argument("sushi_hfv verification currently requires picard");
        }

        const auto wall_start = std::chrono::steady_clock::now();
        const BasinInput physical_input = BasinInput::read(physical_path);
        const BasinInput geometry_input = BasinInput::read(geometry_path);
        const BasinConfiguration& physical_configuration =
            configuration_at(physical_input, kSnapshotAgeMa);
        const BasinConfiguration& configuration = configuration_at(geometry_input, kSnapshotAgeMa);
        const BasinGeometry snapshot_geometry = make_basin_geometry(configuration);
        const LayeredMeshLayout layout =
            axis_mode == AxisMode::retained_lines
                ? make_nonmatching_column_layout(snapshot_geometry, horizontal_cell_size,
                                                 vertical_cell_size, phase)
                : support_only_layout(snapshot_geometry, horizontal_cell_size, vertical_cell_size,
                                      phase);
        const auto mesh_start = std::chrono::steady_clock::now();
        NonmatchingColumnGrid grid = make_nonmatching_column_grid(snapshot_geometry, layout);
        const std::size_t represented_surface_steps = grid.surface_vertical_faces.size();
        if (surface_policy == SurfaceStepPolicy::omit) {
            grid.surface_vertical_faces.clear();
        }
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
        SolveResult solved;
        double surface_outflow = 0.0;
        double basal_inflow = 0.0;
        double lateral_inflow = 0.0;
        double total_production = 0.0;
        std::size_t maximum_stencil = 0;
        double internal_flux_mismatch = 0.0;
        std::vector<double> final_conductivity;
        const NonmatchingColumnBoundaryData boundary_data{
            .surface_temperature = surface_temperature, .basal_inward_flux = basal_flux};
        if (scheme.hfv) {
            NonmatchingColumnHfvSystem system(grid, conductivity, production, boundary_data,
                                              updater);
            Vector full_initial = system.interpolate([](Point2D) { return 300.0; });
            for (const NonmatchingColumnCell& cell : grid.cells) {
                full_initial[static_cast<Vector::Index>(cell.index)] =
                    temperature[static_cast<Vector::Index>(cell.index)];
            }
            solved = solve_hfv(system, std::move(full_initial), *linear_solver);
            const NonmatchingHfvEnergyRates rates = system.energy_rates(solved.temperature);
            const NonmatchingHfvDiagnostics diagnostics = system.diagnostics(solved.temperature);
            surface_outflow = rates.surface_outflow;
            basal_inflow = rates.basal_inflow;
            lateral_inflow = rates.lateral_inflow;
            total_production = rates.heat_production;
            maximum_stencil = diagnostics.maximum_cell_faces;
            internal_flux_mismatch = diagnostics.internal_conservation_defect_linf;
            final_conductivity.assign(system.thermal_conductivity().begin(),
                                      system.thermal_conductivity().end());
        } else {
            NonmatchingColumnHeatSystem system(grid, conductivity, production, boundary_data,
                                               updater);
            system.set_flux_scheme(scheme.multipoint);
            solved = solve(system, std::move(temperature), scheme.multipoint, nonlinear_strategy,
                           *linear_solver);
            const NonmatchingColumnEnergyRates rates = system.energy_rates(solved.temperature);
            const NonmatchingColumnFluxDiagnostics diagnostics =
                system.flux_diagnostics(solved.temperature);
            surface_outflow = rates.surface_outflow;
            basal_inflow = rates.basal_inflow;
            lateral_inflow = rates.lateral_inflow;
            total_production = rates.heat_production;
            maximum_stencil = diagnostics.maximum_stencil_size;
            internal_flux_mismatch = diagnostics.internal_conservation_defect_linf;
            final_conductivity.assign(system.thermal_conductivity().begin(),
                                      system.thermal_conductivity().end());
        }

        double minimum_subface = std::numeric_limits<double>::infinity();
        for (const NonmatchingVerticalSubface& face : grid.vertical_subfaces) {
            minimum_subface = std::min(minimum_subface, face.length());
        }
        double minimum_height = std::numeric_limits<double>::infinity();
        for (const NonmatchingColumnCell& cell : grid.cells) {
            minimum_height = std::min(minimum_height, cell.height());
        }
        const double global_balance =
            surface_outflow - basal_inflow - lateral_inflow - total_production;
        const double wall_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();

        const double domain_left = grid.x.front();
        const double domain_right = grid.x.back();
        const double domain_middle = 0.5 * (domain_left + domain_right);
        const double mean_temperature =
            volume_weighted_mean(grid, solved.temperature, domain_left, domain_right);
        const double left_mean =
            volume_weighted_mean(grid, solved.temperature, domain_left, domain_middle);
        const double right_mean =
            volume_weighted_mean(grid, solved.temperature, domain_middle, domain_right);
        const auto probe_temperature = [&](const Point2D point) {
            const auto upper = std::upper_bound(grid.x.begin(), grid.x.end(), point.x);
            std::size_t column =
                upper == grid.x.begin() ? 0 : static_cast<std::size_t>(upper - grid.x.begin() - 1);
            column = std::min(column, grid.columns.size() - 1);
            return column_value(grid, solved.temperature, column, point.z, surface_temperature,
                                final_conductivity, basal_flux);
        };
        const std::optional<double> probe_750_3000 = probe_temperature({750.0, 3000.0});
        const std::optional<double> probe_750_4000 = probe_temperature({750.0, 4000.0});

        std::filesystem::create_directories(output_directory);
        write_samples(output_directory / "samples.csv", grid, solved.temperature,
                      surface_temperature, final_conductivity, basal_flux);
        std::ofstream cells(output_directory / "cells.csv");
        cells << "index,column,layer,x_left,x_right,z_top,z_bottom,temperature\n"
              << std::setprecision(17);
        for (const NonmatchingColumnCell& cell : grid.cells) {
            cells << cell.index << ',' << cell.column << ',' << cell.layer_id << ',' << cell.x_left
                  << ',' << cell.x_right << ',' << cell.z_top << ',' << cell.z_bottom << ','
                  << solved.temperature[static_cast<Vector::Index>(cell.index)] << '\n';
        }

        std::cout << std::setprecision(17)
                  << "geometry,mesh,scheme,hx,hz,phase,initial,axis_mode,surface_policy,"
                     "nonlinear_strategy,linear_solver,columns,"
                     "cells,algebraic_dofs,face_unknowns,vertical_subfaces,represented_surface_"
                     "vertical_faces,"
                     "assembled_surface_vertical_faces,basal_vertical_faces,nnz,"
                     "nonlinear_updates,total_linear_iterations,maximum_linear_iterations,"
                     "maximum_stencil,internal_flux_mismatch,residual_linf,"
                     "correction_linf,global_balance,surface_outflow,basal_inflow,supplied_heat,"
                     "production,Tmin,Tmax,mean_temperature,left_mean,right_mean,"
                     "probe_750_3000,probe_750_4000,area,minimum_cell_height,minimum_subface,mesh_"
                     "seconds,"
                     "assembly_seconds,linear_solve_seconds,wall_seconds\n"
                  << geometry_label << ",C," << argv[4] << ',' << horizontal_cell_size << ','
                  << vertical_cell_size << ',' << phase << ',' << initial_guess << ','
                  << (axis_mode == AxisMode::retained_lines ? "retained" : "support_only") << ','
                  << (surface_policy == SurfaceStepPolicy::dirichlet ? "dirichlet_steps"
                                                                     : "omit_steps")
                  << ','
                  << (nonlinear_strategy == NonlinearStrategy::residual_line_search
                          ? "residual_line_search"
                          : "picard")
                  << ',' << linear_solver_name << ',' << grid.columns.size() << ','
                  << grid.cells.size() << ',' << solved.temperature.size() << ','
                  << static_cast<std::size_t>(solved.temperature.size()) - grid.cells.size() << ','
                  << grid.vertical_subfaces.size() << ',' << represented_surface_steps << ','
                  << grid.surface_vertical_faces.size() << ',' << grid.basal_vertical_faces.size()
                  << ',' << solved.matrix.nonzero_count() << ',' << solved.nonlinear_iterations
                  << ',' << solved.total_linear_iterations << ','
                  << solved.maximum_linear_iterations << ',' << maximum_stencil << ','
                  << internal_flux_mismatch << ',' << solved.residual.infinity_norm() << ','
                  << solved.final_correction_linf << ',' << global_balance << ',' << surface_outflow
                  << ',' << basal_inflow << ',' << supplied_heat << ',' << total_production << ',';
        double minimum_temperature = std::numeric_limits<double>::infinity();
        double maximum_temperature = -std::numeric_limits<double>::infinity();
        for (const NonmatchingColumnCell& cell : grid.cells) {
            const double value = solved.temperature[static_cast<Vector::Index>(cell.index)];
            minimum_temperature = std::min(minimum_temperature, value);
            maximum_temperature = std::max(maximum_temperature, value);
        }
        std::cout << minimum_temperature << ',' << maximum_temperature << ',' << mean_temperature
                  << ',' << left_mean << ',' << right_mean << ',';
        if (probe_750_3000) {
            std::cout << *probe_750_3000;
        }
        std::cout << ',';
        if (probe_750_4000) {
            std::cout << *probe_750_4000;
        }
        std::cout << ',' << grid.area() << ',' << minimum_height << ',' << minimum_subface << ','
                  << mesh_seconds << ',' << solved.assembly_seconds << ','
                  << solved.linear_solve_seconds << ',' << wall_seconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
