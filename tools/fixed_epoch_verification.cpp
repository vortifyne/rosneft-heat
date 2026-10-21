#include "basin/basin_geometry.hpp"
#include "basin/basin_input.hpp"
#include "basin/basin_state.hpp"
#include "basin/layered_mesh.hpp"
#include "basin/vtu_output.hpp"
#include "linear/umfpack_linear_solver.hpp"
#include "mesh/mesh_transfer.hpp"

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
#include <utility>
#include <vector>

namespace {

constexpr double kSnapshotAgeMa = 5.0;

enum class MeshFamily {
    fixed_rows,
    variable_rows
};
enum class InitialGuess {
    geotherm,
    constant
};

struct MatrixSignMetrics {
    std::size_t positive_offdiagonal_count = 0;
    double maximum_positive_offdiagonal = 0.0;
};

struct SolveResult {
    Vector temperature;
    Vector residual;
    SparseMatrix matrix;
    std::size_t nonlinear_iterations = 0;
    std::size_t matrix_assemblies = 0;
    double final_correction_linf = std::numeric_limits<double>::infinity();
    double assembly_seconds = 0.0;
    double linear_solve_seconds = 0.0;
};

struct BasalFaceFlux {
    double left = 0.0;
    double right = 0.0;
    double density = 0.0;
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

std::pair<std::function<double(Point2D, double)>, double>
make_basal_flux(const Mesh2D& mesh, const BasinConfiguration& physical_configuration) {
    const BasinGeometry physical_geometry = make_basin_geometry(physical_configuration);
    const ScalarProfile& physical_bottom = physical_geometry.boundaries.back().depth;
    std::vector<BasalFaceFlux> faces;
    faces.reserve(mesh.boundary_edges(BoundaryKind::bottom).size());
    double supplied_heat = 0.0;
    for (const TQMesh::Edge* edge : mesh.boundary_edges(BoundaryKind::bottom)) {
        const double left = std::min(edge->v1().xy().x, edge->v2().xy().x);
        const double right = std::max(edge->v1().xy().x, edge->v2().xy().x);
        if (!(right > left)) {
            throw std::runtime_error("Vertical basal face cannot be mapped to z(x)");
        }
        const double heat = integrated_profile_flux(
            physical_bottom, physical_configuration.basal_heat_flux, left, right);
        faces.push_back({left, right, heat / edge->length()});
        supplied_heat += heat;
    }
    std::sort(faces.begin(), faces.end(),
              [](const auto& a, const auto& b) { return a.left < b.left; });
    if (faces.empty()) {
        throw std::runtime_error("Snapshot mesh has no basal boundary");
    }
    const double scale =
        std::max({1.0, std::abs(faces.front().left), std::abs(faces.back().right)});
    const double tolerance = 1.0e-12 * scale;
    if (std::abs(faces.front().left - physical_bottom.x.front()) > tolerance ||
        std::abs(faces.back().right - physical_bottom.x.back()) > tolerance) {
        throw std::runtime_error("Snapshot and physical basal supports differ");
    }
    for (std::size_t i = 1; i < faces.size(); ++i) {
        if (std::abs(faces[i].left - faces[i - 1].right) > tolerance) {
            throw std::runtime_error("Snapshot basal faces have a gap or overlap");
        }
    }
    auto shared_faces = std::make_shared<const std::vector<BasalFaceFlux>>(std::move(faces));
    auto callback = [shared_faces](const Point2D point, double) {
        const auto found = std::lower_bound(
            shared_faces->begin(), shared_faces->end(), point.x,
            [](const BasalFaceFlux& face, const double x) { return face.right < x; });
        if (found == shared_faces->end() || point.x < found->left) {
            throw std::runtime_error("Cannot map basal callback point to a face");
        }
        return found->density;
    };
    return {std::move(callback), supplied_heat};
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

SolveResult solve_stationary(const Mesh2D& mesh, HeatSystem& system, Vector temperature,
                             const HeatFluxScheme scheme) {
    Vector derivative(temperature.size());
    derivative.set_zero();
    SparseMatrix matrix(temperature.size(), temperature.size(), SparseStorageOrder::csc);
    SolveResult result{
        .temperature = std::move(temperature), .residual = {}, .matrix = std::move(matrix)};
    UmfpackLinearSolver solver;
    constexpr std::size_t maximum_iterations = 300;
    const auto elapsed = [](const auto start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    for (std::size_t iteration = 0; iteration < maximum_iterations; ++iteration) {
        auto start = std::chrono::steady_clock::now();
        system.assemble_residual(0.0, result.temperature, derivative, result.residual);
        result.assembly_seconds += elapsed(start);
        const double current_residual = integrated_residual_linf(mesh, result.residual);

        start = std::chrono::steady_clock::now();
        system.assemble_matrix(NonlinearMethod::picard, 0.0, result.temperature, derivative, 0.0,
                               result.matrix);
        result.assembly_seconds += elapsed(start);
        ++result.matrix_assemblies;
        Vector correction;
        start = std::chrono::steady_clock::now();
        const LinearSolveResult linear_result = solver.solve(
            result.matrix, -result.residual, correction,
            {.relative_tolerance = 1.0e-12, .absolute_tolerance = 1.0e-12, .max_iterations = 1});
        result.linear_solve_seconds += elapsed(start);
        if (linear_result.status != LinearSolveStatus::converged || !correction.all_finite()) {
            throw std::runtime_error("UMFPACK failed in fixed-epoch verification");
        }
        result.final_correction_linf = correction.infinity_norm();
        if (iteration < 5 || (iteration + 1) % 25 == 0) {
            std::cerr << "picard iteration=" << iteration << " residual=" << current_residual
                      << " correction=" << result.final_correction_linf << '\n';
        }
        if (current_residual <= 1.0e-8) {
            return result;
        }

        if (scheme == HeatFluxScheme::nonlinear_multipoint_dmp) {
            result.temperature += correction * 0.5;
        } else if (scheme == HeatFluxScheme::nonlinear_multipoint_conservative ||
                   scheme == HeatFluxScheme::linear_multipoint) {
            Vector best_temperature = result.temperature;
            double best_residual = current_residual;
            double relaxation = 1.0;
            for (int search = 0; search < 24; ++search) {
                Vector trial = result.temperature + correction * relaxation;
                Vector trial_residual;
                start = std::chrono::steady_clock::now();
                system.assemble_residual(0.0, trial, derivative, trial_residual);
                result.assembly_seconds += elapsed(start);
                const double trial_norm = integrated_residual_linf(mesh, trial_residual);
                if (trial_norm < best_residual) {
                    best_residual = trial_norm;
                    best_temperature = std::move(trial);
                }
                if (trial_norm <= current_residual * (1.0 - 1.0e-4 * relaxation)) {
                    break;
                }
                relaxation *= 0.5;
            }
            if (!(best_residual < current_residual)) {
                throw std::runtime_error(
                    "Multipoint line search did not reduce residual at " +
                    std::to_string(iteration) + ": residual=" + std::to_string(current_residual) +
                    ", correction=" + std::to_string(result.final_correction_linf));
            }
            result.temperature = std::move(best_temperature);
        } else {
            result.temperature += correction;
        }
        ++result.nonlinear_iterations;
    }
    throw std::runtime_error("Fixed-epoch Picard iteration did not converge");
}

MeshFamily parse_mesh(const std::string_view value) {
    if (value == "A") {
        return MeshFamily::fixed_rows;
    }
    if (value == "B") {
        return MeshFamily::variable_rows;
    }
    throw std::invalid_argument("Mesh must be A or B");
}

HeatFluxScheme parse_scheme(const std::string_view value) {
    if (value == "tpfa") {
        return HeatFluxScheme::tpfa_nonorthogonal;
    }
    if (value == "linear_mpfa") {
        return HeatFluxScheme::linear_multipoint;
    }
    if (value == "nmpfa_conservative") {
        return HeatFluxScheme::nonlinear_multipoint_conservative;
    }
    if (value == "nmpfa_dmp") {
        return HeatFluxScheme::nonlinear_multipoint_dmp;
    }
    throw std::invalid_argument(
        "Scheme must be tpfa, linear_mpfa, nmpfa_conservative, or nmpfa_dmp");
}

InitialGuess parse_initial_guess(const std::string_view value) {
    if (value == "geotherm") {
        return InitialGuess::geotherm;
    }
    if (value == "constant") {
        return InitialGuess::constant;
    }
    throw std::invalid_argument("Initial guess must be geotherm or constant");
}

void write_samples(const std::filesystem::path& path, const BasinState& state) {
    const CellFieldSampler sampler =
        make_cell_field_sampler(state.mesh, vector_values(state.temperature));
    const CellIndexLocator locator = make_cell_index_locator(state.mesh);
    std::ofstream output(path);
    output << "x,z,temperature,material\n" << std::setprecision(17);
    constexpr double step = 12.5;
    for (double x = 0.0; x <= 1500.0; x += step) {
        for (double z = 0.0; z <= 4500.0; z += step) {
            const auto temperature = sampler({x, z});
            const auto cell = locator({x, z});
            output << x << ',' << z << ',';
            if (temperature) {
                output << *temperature;
            }
            output << ',';
            if (cell) {
                output << state.mesh.cells()[*cell]->color();
            }
            output << '\n';
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 9) {
            throw std::invalid_argument(
                "usage: fixed_epoch_verification PHYSICAL_INPUT GEOMETRY_INPUT GEOMETRY_LABEL "
                "A|B SCHEME H geotherm|constant OUTPUT_DIR");
        }
        const std::filesystem::path physical_path = argv[1];
        const std::filesystem::path geometry_path = argv[2];
        const std::string geometry_label = argv[3];
        const MeshFamily family = parse_mesh(argv[4]);
        const HeatFluxScheme scheme = parse_scheme(argv[5]);
        const double cell_size = std::stod(argv[6]);
        const InitialGuess initial_guess = parse_initial_guess(argv[7]);
        const std::filesystem::path output_directory = argv[8];
        if (!std::isfinite(cell_size) || !(cell_size > 0.0)) {
            throw std::invalid_argument("H must be finite and positive");
        }

        const BasinInput physical_input = BasinInput::read(physical_path);
        const BasinInput geometry_input = BasinInput::read(geometry_path);
        const BasinConfiguration& physical_configuration =
            configuration_at(physical_input, kSnapshotAgeMa);
        const BasinConfiguration& configuration = configuration_at(geometry_input, kSnapshotAgeMa);
        const LayeredMeshLayout layout =
            make_layered_mesh_layout(geometry_input.configurations(), cell_size);

        const auto wall_start = std::chrono::steady_clock::now();
        const auto mesh_start = std::chrono::steady_clock::now();
        Mesh2D mesh = family == MeshFamily::fixed_rows
                          ? make_layered_mesh(configuration, layout)
                          : make_variable_layered_mesh(configuration, layout);
        const double mesh_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - mesh_start).count();
        double top_length = 0.0;
        double bottom_length = 0.0;
        for (const TQMesh::Edge* edge : mesh.boundary_edges(BoundaryKind::top)) {
            top_length += edge->length();
        }
        for (const TQMesh::Edge* edge : mesh.boundary_edges(BoundaryKind::bottom)) {
            bottom_length += edge->length();
        }
        const auto [basal_flux, supplied_heat] = make_basal_flux(mesh, physical_configuration);

        const EasyRoInput& parameters = geometry_input.easy_ro();
        EasyRoModel easy_ro({parameters.preexponential, parameters.gas_constant,
                             parameters.activation_energies, parameters.weights});
        BasinState state = make_initial_basin_state(configuration, geometry_input, std::move(mesh),
                                                    easy_ro, configuration.surface_temperature,
                                                    configuration.basal_heat_flux);
        if (initial_guess == InitialGuess::constant) {
            state.temperature.set_constant(300.0);
            update_thermophysical_properties(vector_values(state.temperature), state.fields);
        }
        auto heat = make_heat_system(state.mesh, state.fields,
                                     {.surface_temperature =
                                          [&configuration](const Point2D point, double) {
                                              return configuration.surface_temperature.at(point.x);
                                          },
                                      .basal_heat_flux = basal_flux});
        heat->set_flux_scheme(scheme);

        SolveResult solved =
            solve_stationary(state.mesh, *heat, std::move(state.temperature), scheme);
        state.temperature = std::move(solved.temperature);
        const double wall_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
        const HeatEnergyRates rates = heat->energy_rates(0.0, state.temperature);
        const HeatFluxDiagnostics flux_diagnostics = heat->flux_diagnostics(0.0, state.temperature);
        const MatrixSignMetrics matrix_sign = matrix_sign_metrics(solved.matrix);

        double minimum_thickness = std::numeric_limits<double>::infinity();
        double maximum_elongation = 0.0;
        for (const TQMesh::Facet* cell : state.mesh.cells()) {
            const double length = cell->max_edge_length();
            minimum_thickness = std::min(minimum_thickness, cell->area() / length);
            maximum_elongation = std::max(maximum_elongation, length * length / cell->area());
        }
        const double global_balance = rates.surface_outflow - rates.basal_inflow -
                                      rates.lateral_inflow - rates.heat_production;

        std::filesystem::create_directories(output_directory);
        write_vtu(state.mesh, output_directory / "temperature.vtu",
                  std::vector<VtuCellField>{{"temperature", vector_values(state.temperature)},
                                            {"layer_id", state.fields.layer_id}});
        write_samples(output_directory / "samples.csv", state);

        std::cout
            << std::setprecision(17)
            << "geometry,mesh,scheme,h,initial,cells,nnz,matrix_assemblies,nonlinear_updates,"
               "positive_offdiagonal_count,maximum_positive_offdiagonal,maximum_stencil,"
               "internal_flux_mismatch,integrated_residual_linf,correction_linf,global_balance,"
               "surface_outflow,basal_inflow,supplied_heat,production,Tmin,Tmax,area,top_length,"
               "bottom_length,min_thickness,max_elongation,mesh_seconds,assembly_seconds,"
               "linear_solve_seconds,wall_seconds\n"
            << geometry_label << ',' << argv[4] << ',' << argv[5] << ',' << cell_size << ','
            << argv[7] << ',' << state.mesh.cells().size() << ',' << solved.matrix.nonzero_count()
            << ',' << solved.matrix_assemblies << ',' << solved.nonlinear_iterations << ','
            << matrix_sign.positive_offdiagonal_count << ','
            << matrix_sign.maximum_positive_offdiagonal << ','
            << flux_diagnostics.maximum_stencil_size << ','
            << flux_diagnostics.internal_conservation_defect_linf << ','
            << integrated_residual_linf(state.mesh, solved.residual) << ','
            << solved.final_correction_linf << ',' << global_balance << ',' << rates.surface_outflow
            << ',' << rates.basal_inflow << ',' << supplied_heat << ',' << rates.heat_production
            << ',' << state.temperature.native().minCoeff() << ','
            << state.temperature.native().maxCoeff() << ',' << state.mesh.area() << ','
            << top_length << ',' << bottom_length << ',' << minimum_thickness << ','
            << maximum_elongation << ',' << mesh_seconds << ',' << solved.assembly_seconds << ','
            << solved.linear_solve_seconds << ',' << wall_seconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
