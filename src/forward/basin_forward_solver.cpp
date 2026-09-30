#include "forward/basin_forward_solver.hpp"

#include "discretization/heat_system.hpp"
#include "forward/basin_input.hpp"
#include "forward/mesh_transfer.hpp"
#include "forward/vtu_output.hpp"
#include "physics/easy_ro.hpp"
#include "physics/thermophysical_properties.hpp"
#include "time/time_integrator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kSecondsPerYear = 365.25 * 24.0 * 3600.0;
constexpr double kSecondsPerMa = 1.0e6 * kSecondsPerYear;
constexpr FluidThermophysicalProperties kFluid{
    .thermal_conductivity = 0.7,
    .density = 1040.0,
    .specific_heat = 4184.0,
};

std::span<const double> values(const Vector& vector) {
    return {vector.native().data(), static_cast<std::size_t>(vector.size())};
}

struct CellFields {
    std::vector<double> layer_id;
    std::vector<double> lithotype_code;
    std::vector<double> porosity;
    std::vector<double> material_eta;
    std::vector<double> cell_area;
    std::vector<const LithotypeThermophysicalProperties*> lithotypes;
    std::vector<double> conductivity;
    std::vector<double> heat_capacity;
    std::vector<double> heat_production;
    std::vector<double> velocity_z;
};

Mesh2D make_mesh(const BasinConfiguration& configuration, const double cell_size) {
    return Mesh2D::generate(configuration.regions,
                            {.cell_size = [cell_size](Point2D) { return cell_size; },
                             .smoothing_iterations = 2,
                             .make_quadrilateral = false,
                             .diagnostic_vtu = std::nullopt});
}

CellFields make_cell_fields(const Mesh2D& mesh, const BasinConfiguration& configuration,
                            const BasinInput& input) {
    const std::size_t count = mesh.cells().size();
    CellFields fields{
        .layer_id = std::vector<double>(count),
        .lithotype_code = std::vector<double>(count),
        .porosity = std::vector<double>(count),
        .material_eta = std::vector<double>(count),
        .cell_area = std::vector<double>(count),
        .lithotypes = std::vector<const LithotypeThermophysicalProperties*>(count),
        .conductivity = std::vector<double>(count),
        .heat_capacity = std::vector<double>(count),
        .heat_production = std::vector<double>(count),
        .velocity_z = std::vector<double>(count, 0.0),
    };
    for (const TQMesh::Facet* cell : mesh.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        const int layer_id = cell->color();
        const auto& layer = configuration.layer(layer_id);
        const int lithotype_code = layer.lithotype_at(cell->xy().x);
        fields.layer_id[index] = static_cast<double>(layer_id);
        fields.lithotype_code[index] = static_cast<double>(lithotype_code);
        fields.porosity[index] = layer.porosity_at(cell->xy().x);
        fields.material_eta[index] =
            std::clamp((cell->xy().y - layer.top_at(cell->xy().x)) /
                           (layer.bottom_at(cell->xy().x) - layer.top_at(cell->xy().x)),
                       0.0, 1.0);
        fields.cell_area[index] = cell->area();
        fields.lithotypes[index] = &input.lithotype(lithotype_code);
    }
    return fields;
}

void update_properties(const std::span<const double> temperature, CellFields& fields) {
    evaluate_thermophysical_property_fields(temperature, fields.porosity, fields.lithotypes, kFluid,
                                            fields.conductivity, fields.heat_capacity,
                                            fields.heat_production);
}

class MaterialMotion {
public:
    MaterialMotion(Mesh2D& mesh, const BasinConfiguration& first, const BasinConfiguration& second,
                   const double duration)
        : mesh_(mesh), initial_(mesh.vertex_coordinates()), final_(initial_),
          porosity_first_(mesh.cells().size()), porosity_second_(mesh.cells().size()),
          velocity_z_(mesh.cells().size()) {
        for (const auto& vertex : mesh.native().vertices()) {
            if (vertex->facets().empty()) {
                throw std::runtime_error("Mesh vertex has no material cell");
            }
            const int layer_id = vertex->facets().front()->color();
            const auto& layer_first = first.layer(layer_id);
            const auto& layer_second = second.layer(layer_id);
            const std::size_t index = vertex->index();
            const double x = initial_[index].x;
            const double top = layer_first.top_at(x);
            const double bottom = layer_first.bottom_at(x);
            const double eta = std::clamp((initial_[index].z - top) / (bottom - top), 0.0, 1.0);
            final_[index].z =
                ((1.0 - eta) * layer_second.top_at(x)) + (eta * layer_second.bottom_at(x));
        }
        for (const TQMesh::Facet* cell : mesh.cells()) {
            const std::size_t index = static_cast<std::size_t>(cell->index());
            const auto& layer_first = first.layer(cell->color());
            const auto& layer_second = second.layer(cell->color());
            const double x = cell->xy().x;
            const double top = layer_first.top_at(x);
            const double bottom = layer_first.bottom_at(x);
            const double eta = std::clamp((cell->xy().y - top) / (bottom - top), 0.0, 1.0);
            const double final_z =
                ((1.0 - eta) * layer_second.top_at(x)) + (eta * layer_second.bottom_at(x));
            porosity_first_[index] = layer_first.porosity_at(x);
            porosity_second_[index] = layer_second.porosity_at(x);
            velocity_z_[index] = (final_z - cell->xy().y) / duration;
        }
    }

    void set_position(const double fraction) {
        std::vector<Point2D> coordinates(initial_.size());
        for (std::size_t index = 0; index < coordinates.size(); ++index) {
            coordinates[index] = {
                initial_[index].x,
                ((1.0 - fraction) * initial_[index].z) + (fraction * final_[index].z),
            };
        }
        mesh_.set_vertex_coordinates(coordinates);
    }

    void set_porosity(const double fraction, const std::span<double> porosity) const {
        for (std::size_t index = 0; index < porosity.size(); ++index) {
            porosity[index] =
                ((1.0 - fraction) * porosity_first_[index]) + (fraction * porosity_second_[index]);
        }
    }

    [[nodiscard]] std::span<const double> velocity_z() const noexcept {
        return velocity_z_;
    }

private:
    Mesh2D& mesh_;
    std::vector<Point2D> initial_;
    std::vector<Point2D> final_;
    std::vector<double> porosity_first_;
    std::vector<double> porosity_second_;
    std::vector<double> velocity_z_;
};

HeatBoundaryConditions boundary_conditions(const BasinConfiguration& first,
                                           const BasinConfiguration& second,
                                           const double start_time, const double duration) {
    const auto fraction = [start_time, duration](const double time) {
        return std::clamp((time - start_time) / duration, 0.0, 1.0);
    };
    return {
        .surface_temperature =
            [&first, &second, fraction](const Point2D point, const double time) {
                const double alpha = fraction(time);
                return ((1.0 - alpha) * first.surface_temperature.at(point.x)) +
                       (alpha * second.surface_temperature.at(point.x));
            },
        .basal_heat_flux =
            [&first, &second, fraction](const Point2D point, const double time) {
                const double alpha = fraction(time);
                return ((1.0 - alpha) * first.basal_heat_flux.at(point.x)) +
                       (alpha * second.basal_heat_flux.at(point.x));
            },
    };
}

Vector initial_temperature(const Mesh2D& mesh, const BasinConfiguration& configuration,
                           const CellFields& fields) {
    Vector result(static_cast<Vector::Index>(mesh.cells().size()));
    const auto& top_layer = configuration.layers.front();
    for (const TQMesh::Facet* cell : mesh.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        const double surface_temperature = configuration.surface_temperature.at(cell->xy().x);
        const double conductivity =
            evaluate_thermophysical_properties(surface_temperature, fields.porosity[index],
                                               *fields.lithotypes[index], kFluid)
                .thermal_conductivity;
        const double gradient = configuration.basal_heat_flux.at(cell->xy().x) / conductivity;
        result[cell->index()] =
            surface_temperature + (gradient * (cell->xy().y - top_layer.top_at(cell->xy().x)));
    }
    return result;
}

std::unique_ptr<HeatSystem> make_system(const Mesh2D& mesh, CellFields& fields,
                                        HeatBoundaryConditions conditions) {
    auto updater =
        [&fields](const std::span<const double> temperature, const std::span<double> conductivity,
                  const std::span<double> heat_capacity, const std::span<double> heat_production) {
            evaluate_thermophysical_property_fields(temperature, fields.porosity, fields.lithotypes,
                                                    kFluid, conductivity, heat_capacity,
                                                    heat_production);
        };
    auto system =
        std::make_unique<HeatSystem>(mesh, fields.conductivity, fields.heat_capacity,
                                     fields.heat_production, std::move(conditions), updater);
    system->set_implicit_nonorthogonal_correction(true);
    return system;
}

double minimum(const std::span<const double> field) {
    return *std::min_element(field.begin(), field.end());
}

double maximum(const std::span<const double> field) {
    return *std::max_element(field.begin(), field.end());
}

struct OutputState {
    std::filesystem::path directory;
    std::ofstream statistics;
    std::vector<PvdEntry> series;
    int index = 0;
};

void save_state(OutputState& output, const Mesh2D& mesh, const Vector& temperature,
                CellFields& fields, const EasyRoModel& easy_ro,
                const std::span<const double> maturity, const double elapsed_ma,
                const double age_ma, const std::string& kind, const int nonlinear_iterations) {
    update_properties(values(temperature), fields);
    for (const TQMesh::Facet* cell : mesh.cells()) {
        fields.cell_area[static_cast<std::size_t>(cell->index())] = cell->area();
    }
    std::vector<double> reflectance(mesh.cells().size());
    easy_ro.reflectance(maturity, reflectance);
    const std::string filename = "state_" + [&output] {
        std::ostringstream stream;
        stream << std::setw(4) << std::setfill('0') << output.index++;
        return stream.str();
    }() + "_" + kind + ".vtu";
    const std::vector<VtuCellField> vtu_fields = {
        {"temperature", values(temperature)},
        {"porosity", fields.porosity},
        {"material_eta", fields.material_eta},
        {"cell_area", fields.cell_area},
        {"layer_id", fields.layer_id},
        {"lithotype", fields.lithotype_code},
        {"thermal_conductivity", fields.conductivity},
        {"volumetric_heat_capacity", fields.heat_capacity},
        {"heat_production", fields.heat_production},
        {"velocity_z", fields.velocity_z},
        {"Ro", reflectance},
    };
    write_vtu(mesh, output.directory / filename, vtu_fields);
    output.series.push_back({elapsed_ma, filename});
    output.statistics << std::setprecision(15) << elapsed_ma << ',' << age_ma << ',' << kind << ','
                      << mesh.cells().size() << ',' << nonlinear_iterations << ','
                      << minimum(fields.cell_area) << ',' << minimum(values(temperature)) << ','
                      << maximum(values(temperature)) << ',' << minimum(reflectance) << ','
                      << maximum(reflectance) << '\n';
}

double cell_enthalpy(const double temperature, const double porosity,
                     const LithotypeThermophysicalProperties& lithotype) {
    constexpr double reference = 273.15;
    constexpr int intervals = 8;
    const double width = (temperature - reference) / intervals;
    double result = 0.0;
    for (int index = 0; index < intervals; ++index) {
        const double point = reference + ((static_cast<double>(index) + 0.5) * width);
        result += evaluate_thermophysical_properties(point, porosity, lithotype, kFluid)
                      .volumetric_heat_capacity;
    }
    return result * width;
}

double transfer_energy_error(const Mesh2D& old_mesh, const Vector& old_temperature,
                             const CellFields& old_fields, const Mesh2D& new_mesh,
                             const Vector& new_temperature, const CellFields& new_fields,
                             const CellTransferMap& transfer_map) {
    double before = 0.0;
    for (const TQMesh::Facet* cell : old_mesh.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        before +=
            cell->area() * cell_enthalpy(old_temperature[cell->index()], old_fields.porosity[index],
                                         *old_fields.lithotypes[index]);
    }
    double after = 0.0;
    for (const TQMesh::Facet* cell : new_mesh.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        if (!transfer_map.donor_cells[index]) {
            continue;
        }
        after +=
            cell->area() * cell_enthalpy(new_temperature[cell->index()], new_fields.porosity[index],
                                         *new_fields.lithotypes[index]);
    }
    return std::abs(after - before) / std::max(std::abs(before), 1.0);
}

NonlinearSolveRequest nonlinear_request() {
    return {
        .nonlinear_method = NonlinearMethod::picard,
        .relative_tolerance = 1.0e-6,
        .absolute_tolerance = 0.0,
        .step_relative_tolerance = 1.0e-8,
        .max_iterations = 200,
        .use_backtracking = true,
        .max_backtracking_steps = 20,
        .backtracking_reduction = 0.5,
    };
}

LinearSolveRequest linear_request() {
    return {.relative_tolerance = 1.0e-12, .absolute_tolerance = 1.0e-12, .max_iterations = 1};
}

void accumulate_result(BasinForwardResult& total, const TimeIntegrationResult& step) {
    total.accepted_steps += step.accepted_steps;
    total.nonlinear_iterations += step.nonlinear_iterations;
    total.linear_iterations += step.linear_iterations;
    if (!step.completed()) {
        throw std::runtime_error(
            "Nonlinear solution did not converge: nonlinear_status=" +
            std::to_string(
                step.last_nonlinear_status ? static_cast<int>(*step.last_nonlinear_status) : -1) +
            " nonlinear_iterations=" + std::to_string(step.nonlinear_iterations) +
            " residual=" + std::to_string(step.last_residual_norm));
    }
}

} // namespace

BasinForwardResult run_basin_forward(const std::filesystem::path& input_directory,
                                     const std::filesystem::path& output_directory,
                                     const BasinForwardOptions& options) {
    const auto wall_start = std::chrono::steady_clock::now();
    if (!(options.cell_size > 0.0) || !(options.timestep_ma > 0.0) || options.save_every <= 0 ||
        options.steps_after_transition < 0) {
        throw std::invalid_argument("Invalid forward-solver options");
    }
    const BasinInput input = BasinInput::read(input_directory);
    const BasinConfiguration& oldest = input.configuration(200.0);
    const BasinConfiguration& transition = input.configuration(187.5);
    const BasinConfiguration& next = input.configuration(175.0);
    const double first_duration = (oldest.age_ma - transition.age_ma) * kSecondsPerMa;
    const double next_duration = (transition.age_ma - next.age_ma) * kSecondsPerMa;
    const double timestep = options.timestep_ma * kSecondsPerMa;

    std::filesystem::create_directories(output_directory);
    OutputState output{.directory = output_directory,
                       .statistics = std::ofstream(output_directory / "statistics.csv"),
                       .series = {},
                       .index = 0};
    if (!output.statistics) {
        throw std::runtime_error("Cannot open forward-solver statistics output");
    }
    output.statistics << "elapsed_ma,age_ma,kind,cells,nonlinear_iterations,cell_area_min,"
                         "temperature_min,"
                         "temperature_max,Ro_min,Ro_max\n";

    const EasyRoInput& easy_input = input.easy_ro();
    const EasyRoModel easy_ro({.preexponential = easy_input.preexponential,
                               .gas_constant = easy_input.gas_constant,
                               .activation_energies = easy_input.activation_energies,
                               .weights = easy_input.weights});

    Mesh2D mesh = make_mesh(oldest, options.cell_size);
    CellFields fields = make_cell_fields(mesh, oldest, input);
    Vector temperature = initial_temperature(mesh, oldest, fields);
    update_properties(values(temperature), fields);
    std::vector<double> maturity = easy_ro.initial_state(mesh.cells().size());
    MaterialMotion first_motion(mesh, oldest, transition, first_duration);
    if (!options.fixed_mesh) {
        fields.velocity_z.assign(first_motion.velocity_z().begin(),
                                 first_motion.velocity_z().end());
    }
    auto system =
        make_system(mesh, fields, boundary_conditions(oldest, transition, 0.0, first_duration));
    TimeIntegrator integrator;
    integrator.set_initial_solution(0.0, temperature);
    integrator.set_timestep(timestep);

    BasinForwardResult result;
    save_state(output, mesh, temperature, fields, easy_ro, maturity, 0.0, oldest.age_ma, "initial",
               0);
    int step_index = 0;
    while (integrator.current_snapshot().time < first_duration) {
        const double old_time = integrator.current_snapshot().time;
        const Vector old_temperature = integrator.current_snapshot().solution;
        const double target = std::min(first_duration, old_time + timestep);
        if (!options.fixed_mesh) {
            const double alpha = target / first_duration;
            first_motion.set_position(alpha);
            first_motion.set_porosity(alpha, fields.porosity);
            system->update_geometry(mesh);
        }
        const TimeIntegrationResult step =
            integrator.advance_to(*system, target, nonlinear_request(), linear_request());
        accumulate_result(result, step);
        temperature = integrator.current_snapshot().solution;
        easy_ro.advance(values(old_temperature), values(temperature), target - old_time, maturity);
        ++step_index;
        if (step_index % options.save_every == 0 || target == first_duration) {
            save_state(output, mesh, temperature, fields, easy_ro, maturity, target / kSecondsPerMa,
                       oldest.age_ma - (target / kSecondsPerMa), "epoch",
                       step.nonlinear_iterations);
        }
    }

    if (options.fixed_mesh) {
        result.final_age_ma = transition.age_ma;
        result.wall_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
        write_pvd(output_directory / "temperature.pvd", output.series);
        std::ofstream summary(output_directory / "summary.csv");
        summary << "accepted_steps,nonlinear_iterations,linear_iterations,final_age_ma,"
                   "wall_seconds\n"
                << result.accepted_steps << ',' << result.nonlinear_iterations << ','
                << result.linear_iterations << ',' << result.final_age_ma << ','
                << result.wall_seconds << '\n';
        return result;
    }

    save_state(output, mesh, temperature, fields, easy_ro, maturity, first_duration / kSecondsPerMa,
               transition.age_ma, "before_transition", 0);

    Mesh2D new_mesh = make_mesh(transition, options.cell_size);
    CellFields new_fields = make_cell_fields(new_mesh, transition, input);
    const CellTransferMap transfer_map = make_cell_transfer_map(mesh, new_mesh);
    const auto initialize_temperature = [&](const std::size_t index, const Point2D point) {
        const auto& layer = transition.layer(new_mesh.cells()[index]->color());
        const double top = layer.top_at(point.x);
        const double bottom = layer.bottom_at(point.x);
        const double eta = std::clamp((point.z - top) / (bottom - top), 0.0, 1.0);
        const double surface = transition.surface_temperature.at(point.x);
        std::optional<double> contact;
        for (int attempt = 0; attempt <= 10 && !contact; ++attempt) {
            const double inward_offset = static_cast<double>(attempt) * 0.1 * options.cell_size;
            contact =
                sample_cell_field(mesh, values(temperature), {point.x, bottom + inward_offset});
        }
        if (!contact) {
            throw std::runtime_error(
                "Cannot initialize material at contact: layer=" + std::to_string(layer.id) +
                " x=" + std::to_string(point.x) + " z=" + std::to_string(point.z) +
                " bottom=" + std::to_string(bottom));
        }
        return ((1.0 - eta) * surface) + (eta * *contact);
    };
    const std::vector<double> transferred_temperature =
        transfer_cell_field(mesh, values(temperature), transfer_map, initialize_temperature);
    Vector new_temperature(static_cast<Vector::Index>(new_mesh.cells().size()));
    for (std::size_t index = 0; index < transferred_temperature.size(); ++index) {
        new_temperature[static_cast<Vector::Index>(index)] = transferred_temperature[index];
    }

    std::vector<double> new_maturity = easy_ro.initial_state(new_mesh.cells().size());
    for (std::size_t reaction = 0; reaction < easy_ro.reaction_count(); ++reaction) {
        std::vector<double> old_reaction(mesh.cells().size());
        for (std::size_t cell = 0; cell < old_reaction.size(); ++cell) {
            old_reaction[cell] = maturity[(cell * easy_ro.reaction_count()) + reaction];
        }
        const std::vector<double> transferred = transfer_cell_field(
            mesh, old_reaction, transfer_map, [](std::size_t, Point2D) { return 0.0; });
        for (std::size_t cell = 0; cell < transferred.size(); ++cell) {
            new_maturity[(cell * easy_ro.reaction_count()) + reaction] =
                std::max(0.0, transferred[cell]);
        }
    }
    result.transfer_energy_error = transfer_energy_error(mesh, temperature, fields, new_mesh,
                                                         new_temperature, new_fields, transfer_map);
    {
        std::ofstream transition_output(output_directory / "transition.csv");
        transition_output << std::setprecision(17)
                          << "age_ma,energy_relative_error,old_cells,new_cells\n"
                          << transition.age_ma << ',' << result.transfer_energy_error << ','
                          << mesh.cells().size() << ',' << new_mesh.cells().size() << '\n';
    }

    mesh = std::move(new_mesh);
    fields = std::move(new_fields);
    temperature = std::move(new_temperature);
    maturity = std::move(new_maturity);
    update_properties(values(temperature), fields);
    MaterialMotion second_motion(mesh, transition, next, next_duration);
    fields.velocity_z.assign(second_motion.velocity_z().begin(), second_motion.velocity_z().end());
    system = make_system(mesh, fields,
                         boundary_conditions(transition, next, first_duration, next_duration));
    integrator.set_initial_solution(first_duration, temperature);
    save_state(output, mesh, temperature, fields, easy_ro, maturity, first_duration / kSecondsPerMa,
               transition.age_ma, "after_transition", 0);

    for (int next_step = 0; next_step < options.steps_after_transition; ++next_step) {
        const double old_time = integrator.current_snapshot().time;
        const Vector old_temperature = integrator.current_snapshot().solution;
        const double target = old_time + timestep;
        const double alpha = (target - first_duration) / next_duration;
        second_motion.set_position(alpha);
        second_motion.set_porosity(alpha, fields.porosity);
        system->update_geometry(mesh);
        const TimeIntegrationResult step =
            integrator.advance_to(*system, target, nonlinear_request(), linear_request());
        accumulate_result(result, step);
        temperature = integrator.current_snapshot().solution;
        easy_ro.advance(values(old_temperature), values(temperature), target - old_time, maturity);
        save_state(output, mesh, temperature, fields, easy_ro, maturity, target / kSecondsPerMa,
                   transition.age_ma - ((target - first_duration) / kSecondsPerMa), "next_epoch",
                   step.nonlinear_iterations);
    }

    result.final_age_ma = transition.age_ma - (static_cast<double>(options.steps_after_transition) *
                                               options.timestep_ma);
    result.wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    write_pvd(output_directory / "temperature.pvd", output.series);
    std::ofstream summary(output_directory / "summary.csv");
    summary << "accepted_steps,nonlinear_iterations,linear_iterations,final_age_ma,"
               "transfer_energy_error,wall_seconds\n"
            << result.accepted_steps << ',' << result.nonlinear_iterations << ','
            << result.linear_iterations << ',' << result.final_age_ma << ','
            << result.transfer_energy_error << ',' << result.wall_seconds << '\n';
    return result;
}
