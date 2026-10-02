#include "forward/basin_forward_solver.hpp"

#include "discretization/heat_system.hpp"
#include "forward/basin_input.hpp"
#include "forward/layered_mesh.hpp"
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
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr double kSecondsPerYear = 365.25 * 24.0 * 3600.0;
constexpr double kSecondsPerMa = 1.0e6 * kSecondsPerYear;
constexpr double kMinimumTopologyThickness = 0.5;
constexpr double kMinimumTopologyThicknessFraction = 0.001;
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

Mesh2D make_mesh(const BasinConfiguration& configuration, const double cell_size,
                 const double thin_layer_cell_fraction, const bool quad_dominant,
                 const int smoothing_iterations = 2) {
    return Mesh2D::generate(
        configuration.regions,
        {.cell_size = [cell_size](Point2D) { return cell_size; },
         .region_cell_size =
             [&configuration, cell_size, thin_layer_cell_fraction](const int layer_id,
                                                                   const Point2D point) {
                 const auto& layer = configuration.layer(layer_id);
                 if (!(thin_layer_cell_fraction > 0.0)) {
                     return cell_size;
                 }
                 const double thickness =
                     std::abs(layer.bottom_at(point.x) - layer.top_at(point.x));
                 return std::min(cell_size,
                                 std::max(0.1 * cell_size, thin_layer_cell_fraction * thickness));
             },
         .smoothing_iterations = smoothing_iterations,
         .make_quadrilateral = quad_dominant,
         .diagnostic_vtu = std::nullopt});
}

Mesh2D make_preferred_mesh(const BasinConfiguration& configuration, const double cell_size,
                           const double thin_layer_cell_fraction, const bool quad_dominant) {
    if (!quad_dominant) {
        return make_mesh(configuration, cell_size, thin_layer_cell_fraction, false);
    }
    try {
        return make_mesh(configuration, cell_size, thin_layer_cell_fraction, true);
    } catch (const std::exception&) {
        return make_mesh(configuration, cell_size, thin_layer_cell_fraction, false);
    }
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
                   const double duration, const bool preserve_topology = false)
        : mesh_(mesh), initial_(mesh.vertex_coordinates()), final_(initial_),
          porosity_first_(mesh.cells().size()), porosity_second_(mesh.cells().size()),
          velocity_z_(mesh.cells().size()) {
        const auto target_interval = [&first, &second, preserve_topology](const int layer_id,
                                                                          const double x) {
            const auto& requested = second.layer(layer_id);
            if (!preserve_topology) {
                return std::pair{requested.top_at(x), requested.bottom_at(x)};
            }
            const auto& old_requested = first.layer(layer_id);
            if (old_requested.bottom_at(x) - old_requested.top_at(x) <= 1.0e-10) {
                return std::pair{requested.top_at(x), requested.bottom_at(x)};
            }

            bool found_first_material = false;
            double boundary = 0.0;
            for (const BasinLayerProfile& old_layer : first.layers) {
                if (x < old_layer.x.front() || x > old_layer.x.back()) {
                    continue;
                }
                const double old_thickness = old_layer.bottom_at(x) - old_layer.top_at(x);
                if (!(old_thickness > 1.0e-10)) {
                    continue;
                }
                const auto& new_layer = second.layer(old_layer.id);
                if (!found_first_material) {
                    boundary = new_layer.top_at(x);
                    found_first_material = true;
                }
                const double new_thickness = new_layer.bottom_at(x) - new_layer.top_at(x);
                const double retained_thickness =
                    new_thickness > 1.0e-10
                        ? new_thickness
                        : std::min(old_thickness,
                                   std::max(kMinimumTopologyThickness,
                                            kMinimumTopologyThicknessFraction * old_thickness));
                const double top = boundary;
                boundary += retained_thickness;
                if (old_layer.id == layer_id) {
                    return std::pair{top, boundary};
                }
            }
            throw std::runtime_error("Cannot map layer while preserving mesh topology");
        };
        for (const auto& vertex : mesh.native().vertices()) {
            if (vertex->facets().empty()) {
                throw std::runtime_error("Mesh vertex has no material cell");
            }
            const std::size_t index = vertex->index();
            const double x = initial_[index].x;
            double final_z = 0.0;
            std::unordered_set<int> adjacent_layers;
            for (const TQMesh::Facet* facet : vertex->facets()) {
                if (!adjacent_layers.insert(facet->color()).second) {
                    continue;
                }
                const auto& layer_first = first.layer(facet->color());
                const double top = layer_first.top_at(x);
                const double bottom = layer_first.bottom_at(x);
                const double thickness = bottom - top;
                const double eta = std::abs(thickness) > 1.0e-12
                                       ? std::clamp((initial_[index].z - top) / thickness, 0.0, 1.0)
                                       : 0.5;
                const auto [target_top, target_bottom] = target_interval(facet->color(), x);
                final_z += ((1.0 - eta) * target_top) + (eta * target_bottom);
            }
            final_[index].z = final_z / static_cast<double>(adjacent_layers.size());
        }

        const auto first_non_positive_cell =
            [&mesh](const std::span<const Point2D> coordinates) -> std::optional<std::size_t> {
            for (const TQMesh::Facet* cell : mesh.cells()) {
                double twice_area = 0.0;
                for (std::size_t vertex = 0; vertex < cell->n_vertices(); ++vertex) {
                    const Point2D& current = coordinates[cell->vertex(vertex).index()];
                    const Point2D& next =
                        coordinates[cell->vertex((vertex + 1) % cell->n_vertices()).index()];
                    twice_area += (current.x * next.z) - (next.x * current.z);
                }
                if (!(twice_area > 0.0) || !std::isfinite(twice_area)) {
                    return static_cast<std::size_t>(cell->index());
                }
            }
            return std::nullopt;
        };
        if (const auto cell_index = first_non_positive_cell(final_)) {
            const TQMesh::Facet* cell = mesh.cells()[*cell_index];
            std::ostringstream message;
            message << (preserve_topology ? "Topology-preserving" : "Exact")
                    << " material motion produces a non-positive cell " << *cell_index
                    << " in layer " << cell->color() << ":";
            for (std::size_t vertex = 0; vertex < cell->n_vertices(); ++vertex) {
                const std::size_t index = cell->vertex(vertex).index();
                message << " (" << initial_[index].x << ',' << initial_[index].z << " -> "
                        << final_[index].x << ',' << final_[index].z << ')';
            }
            throw std::runtime_error(message.str());
        }
        for (const TQMesh::Facet* cell : mesh.cells()) {
            const std::size_t index = static_cast<std::size_t>(cell->index());
            const auto& layer_first = first.layer(cell->color());
            const auto& layer_second = second.layer(cell->color());
            const double x = cell->xy().x;
            porosity_first_[index] = layer_first.porosity_at(x);
            porosity_second_[index] = layer_second.porosity_at(x);
            double final_z = 0.0;
            for (std::size_t vertex = 0; vertex < cell->n_vertices(); ++vertex) {
                final_z += final_[cell->vertex(vertex).index()].z;
            }
            final_z /= static_cast<double>(cell->n_vertices());
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
    std::ofstream epochs;
    std::ofstream transitions;
    std::ofstream conditioning;
    std::ofstream energy_balance;
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

struct TransferEnergy {
    double imbalance = 0.0;
};

double total_energy(const Mesh2D& mesh, const Vector& temperature, const CellFields& fields) {
    double result = 0.0;
    for (const TQMesh::Facet* cell : mesh.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        result += cell->area() * cell_enthalpy(temperature[cell->index()], fields.porosity[index],
                                               *fields.lithotypes[index]);
    }
    return result;
}

TransferEnergy transfer_energy(const Mesh2D& old_mesh, const Vector& old_temperature,
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
    const double imbalance = after - before;
    return {.imbalance = imbalance};
}

struct TransferredState {
    Mesh2D mesh;
    CellFields fields;
    Vector temperature;
    std::vector<double> maturity;
    double energy_imbalance = 0.0;
};

CellTransferMap make_material_transfer_map(const Mesh2D& source, const Mesh2D& target,
                                           const BasinConfiguration& source_configuration,
                                           const BasinConfiguration& target_configuration) {
    CellTransferMap result;
    const CellIndexLocator locate = make_cell_index_locator(source);
    result.target_points.resize(target.cells().size());
    result.donor_cells.resize(target.cells().size());
    for (const TQMesh::Facet* cell : target.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        const auto& target_layer = target_configuration.layer(cell->color());
        const auto& source_layer = source_configuration.layer(cell->color());
        const double x = cell->xy().x;
        const double target_top = target_layer.top_at(x);
        const double target_bottom = target_layer.bottom_at(x);
        const double eta =
            std::clamp((cell->xy().y - target_top) / (target_bottom - target_top), 0.0, 1.0);
        result.target_points[index] = {x, ((1.0 - eta) * source_layer.top_at(x)) +
                                              (eta * source_layer.bottom_at(x))};
        result.donor_cells[index] = locate(result.target_points[index]);
    }
    return result;
}

TransferredState transfer_state(const Mesh2D& old_mesh, const Vector& old_temperature,
                                const CellFields& old_fields,
                                const std::span<const double> old_maturity, Mesh2D new_mesh,
                                const BasinConfiguration& old_configuration,
                                const BasinConfiguration& new_configuration,
                                const BasinInput& input, const EasyRoModel& easy_ro,
                                const double cell_size, const bool initialize_new_material,
                                const bool follow_material = false) {
    CellFields new_fields = make_cell_fields(new_mesh, new_configuration, input);
    CellTransferMap transfer_map =
        follow_material
            ? make_material_transfer_map(old_mesh, new_mesh, old_configuration, new_configuration)
            : make_cell_transfer_map(old_mesh, new_mesh);
    for (const TQMesh::Facet* cell : new_mesh.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        if (transfer_map.donor_cells[index]) {
            continue;
        }
        const auto& layer = new_configuration.layer(cell->color());
        const bool existed = std::any_of(
            old_configuration.layers.begin(), old_configuration.layers.end(),
            [&layer](const BasinLayerProfile& old_layer) { return old_layer.id == layer.id; });
        if (!existed) {
            if (!initialize_new_material) {
                throw std::runtime_error("Remeshing created material outside the old domain");
            }
            continue;
        }
        const Point2D target_point = transfer_map.target_points[index];
        const double target_eta = new_fields.material_eta[index];
        double nearest_distance = std::numeric_limits<double>::max();
        std::optional<std::size_t> nearest;
        for (const TQMesh::Facet* source_cell : old_mesh.cells()) {
            const std::size_t source_index = static_cast<std::size_t>(source_cell->index());
            if (static_cast<int>(old_fields.layer_id[source_index]) != layer.id) {
                continue;
            }
            const double horizontal = (source_cell->xy().x - target_point.x) / cell_size;
            const double vertical = old_fields.material_eta[source_index] - target_eta;
            const double distance = (horizontal * horizontal) + (vertical * vertical);
            if (distance < nearest_distance) {
                nearest_distance = distance;
                nearest = source_index;
            }
        }
        if (!nearest) {
            throw std::runtime_error("Cannot locate existing material during state transfer");
        }
        transfer_map.donor_cells[index] = nearest;
        transfer_map.target_points[index] = {old_mesh.cells()[*nearest]->xy().x,
                                             old_mesh.cells()[*nearest]->xy().y};
    }

    const CellFieldSampler temperature_sampler =
        make_cell_field_sampler(old_mesh, values(old_temperature));
    const auto initialize_temperature = [&](const std::size_t index, const Point2D point) {
        const auto& layer = new_configuration.layer(new_mesh.cells()[index]->color());
        const double top = layer.top_at(point.x);
        const double bottom = layer.bottom_at(point.x);
        const double eta = std::clamp((point.z - top) / (bottom - top), 0.0, 1.0);
        const double surface = new_configuration.surface_temperature.at(point.x);
        std::optional<double> contact;
        for (int attempt = 0; attempt <= 10 && !contact; ++attempt) {
            const double inward_offset = static_cast<double>(attempt) * 0.1 * cell_size;
            contact = temperature_sampler({point.x, bottom + inward_offset});
        }
        if (!contact) {
            throw std::runtime_error("Cannot initialize new material at its lower contact");
        }
        return ((1.0 - eta) * surface) + (eta * *contact);
    };
    const std::vector<double> transferred_temperature = transfer_cell_field(
        old_mesh, values(old_temperature), transfer_map, initialize_temperature);
    Vector new_temperature(static_cast<Vector::Index>(new_mesh.cells().size()));
    for (std::size_t index = 0; index < transferred_temperature.size(); ++index) {
        new_temperature[static_cast<Vector::Index>(index)] = transferred_temperature[index];
    }

    std::vector<double> new_maturity = easy_ro.initial_state(new_mesh.cells().size());
    for (std::size_t reaction = 0; reaction < easy_ro.reaction_count(); ++reaction) {
        std::vector<double> old_reaction(old_mesh.cells().size());
        for (std::size_t cell = 0; cell < old_reaction.size(); ++cell) {
            old_reaction[cell] = old_maturity[(cell * easy_ro.reaction_count()) + reaction];
        }
        const std::vector<double> transferred = transfer_cell_field(
            old_mesh, old_reaction, transfer_map, [](std::size_t, Point2D) { return 0.0; });
        for (std::size_t cell = 0; cell < transferred.size(); ++cell) {
            new_maturity[(cell * easy_ro.reaction_count()) + reaction] =
                std::max(0.0, transferred[cell]);
        }
    }
    const TransferEnergy energy = transfer_energy(old_mesh, old_temperature, old_fields, new_mesh,
                                                  new_temperature, new_fields, transfer_map);
    return {.mesh = std::move(new_mesh),
            .fields = std::move(new_fields),
            .temperature = std::move(new_temperature),
            .maturity = std::move(new_maturity),
            .energy_imbalance = energy.imbalance};
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

LinearSolveRequest linear_request(const bool estimate_condition) {
    return {.relative_tolerance = 1.0e-12,
            .absolute_tolerance = 1.0e-12,
            .max_iterations = 1,
            .estimate_condition = estimate_condition};
}

void accumulate_result(BasinForwardResult& total, const TimeIntegrationResult& step) {
    total.accepted_steps += step.accepted_steps;
    total.nonlinear_iterations += step.nonlinear_iterations;
    total.maximum_nonlinear_iterations_per_step =
        std::max(total.maximum_nonlinear_iterations_per_step, step.nonlinear_iterations);
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

double maximum_cell_diameter(const Mesh2D& mesh) {
    double result = 0.0;
    for (const TQMesh::Facet* cell : mesh.cells()) {
        for (std::size_t first = 0; first < cell->n_vertices(); ++first) {
            for (std::size_t second = first + 1; second < cell->n_vertices(); ++second) {
                const auto difference = cell->vertex(first).xy() - cell->vertex(second).xy();
                result = std::max(result, difference.norm());
            }
        }
    }
    return result;
}

double minimum_cell_area(const Mesh2D& mesh) {
    double result = std::numeric_limits<double>::infinity();
    for (const TQMesh::Facet* cell : mesh.cells()) {
        result = std::min(result, cell->area());
    }
    return result;
}

void update_mesh_statistics(BasinForwardResult& result, const Mesh2D& mesh) {
    const std::size_t cells = mesh.cells().size();
    if (result.minimum_cells == 0) {
        result.minimum_cells = cells;
    } else {
        result.minimum_cells = std::min(result.minimum_cells, cells);
    }
    result.maximum_cells = std::max(result.maximum_cells, cells);
    result.maximum_cell_diameter =
        std::max(result.maximum_cell_diameter, maximum_cell_diameter(mesh));
}

struct ComparisonPoint {
    std::string identifier;
    Point2D point;
};

std::vector<ComparisonPoint> read_comparison_points(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open comparison points: " + path.string());
    }
    std::string line;
    if (!std::getline(input, line) || line != "id,x,z") {
        throw std::runtime_error("Comparison points must start with id,x,z");
    }
    std::vector<ComparisonPoint> result;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const std::size_t first = line.find(',');
        const std::size_t second = line.find(',', first == std::string::npos ? first : first + 1);
        if (first == std::string::npos || second == std::string::npos ||
            line.find(',', second + 1) != std::string::npos) {
            throw std::runtime_error("Invalid comparison point row: " + line);
        }
        result.push_back({.identifier = line.substr(0, first),
                          .point = {.x = std::stod(line.substr(first + 1, second - first - 1)),
                                    .z = std::stod(line.substr(second + 1))}});
    }
    if (result.empty()) {
        throw std::runtime_error("Comparison point list is empty");
    }
    return result;
}

void write_comparison_temperatures(const std::filesystem::path& path, const Mesh2D& mesh,
                                   const Vector& temperature,
                                   const std::span<const ComparisonPoint> points) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot write comparison temperatures: " + path.string());
    }
    output << std::setprecision(17) << "id,x,z,temperature\n";
    for (const auto& point : points) {
        const std::optional<double> value =
            sample_cell_field(mesh, values(temperature), point.point);
        if (!value) {
            throw std::runtime_error("Comparison point is outside the final mesh: " +
                                     point.identifier);
        }
        output << point.identifier << ',' << point.point.x << ',' << point.point.z << ',' << *value
               << '\n';
    }
}

std::uintmax_t directory_size(const std::filesystem::path& directory) {
    std::uintmax_t result = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            result += entry.file_size();
        }
    }
    return result;
}

} // namespace

BasinForwardResult run_basin_forward(const std::filesystem::path& input_directory,
                                     const std::filesystem::path& output_directory,
                                     const BasinForwardOptions& options) {
    const auto wall_start = std::chrono::steady_clock::now();
    if (!(options.cell_size > 0.0) || !(options.timestep_ma > 0.0) ||
        options.saved_states_per_epoch <= 0 || options.thin_layer_cell_fraction < 0.0 ||
        (options.max_configurations && *options.max_configurations < 2)) {
        throw std::invalid_argument("Invalid forward-solver options");
    }
    const BasinInput input = BasinInput::read(input_directory);
    const std::span<const BasinConfiguration> configurations = input.configurations();
    std::size_t configuration_count = configurations.size();
    if (options.max_configurations) {
        configuration_count = std::min(configuration_count, *options.max_configurations);
    }
    if (configuration_count < 2) {
        throw std::runtime_error("At least two basin configurations are required");
    }
    if (options.fixed_mesh) {
        configuration_count = 2;
    }
    for (std::size_t index = 1; index < configuration_count; ++index) {
        if (!(configurations[index - 1].age_ma > configurations[index].age_ma)) {
            throw std::runtime_error("Basin configurations must be ordered from oldest to newest");
        }
    }
    const double timestep = options.timestep_ma * kSecondsPerMa;

    std::filesystem::create_directories(output_directory);
    OutputState output{.directory = output_directory,
                       .statistics = std::ofstream(output_directory / "statistics.csv"),
                       .epochs = std::ofstream(output_directory / "epochs.csv"),
                       .transitions = std::ofstream(output_directory / "transitions.csv"),
                       .conditioning = {},
                       .energy_balance = std::ofstream(output_directory / "energy-balance.csv"),
                       .series = {},
                       .index = 0};
    if (!output.statistics || !output.epochs || !output.transitions || !output.energy_balance) {
        throw std::runtime_error("Cannot open forward-solver tabular output");
    }
    output.statistics << "elapsed_ma,age_ma,kind,cells,nonlinear_iterations,cell_area_min,"
                         "temperature_min,"
                         "temperature_max,Ro_min,Ro_max\n";
    output.epochs << "start_age_ma,end_age_ma,steps,nonlinear_iterations,linear_iterations,"
                     "start_cells,end_cells,topology_regularized,wall_seconds\n";
    output.transitions << "age_ma,old_cells,new_cells\n";
    output.energy_balance << "elapsed_ma,age_ma,kind,relative_error\n";
    if (options.estimate_condition) {
        output.conditioning.open(output_directory / "conditioning.csv");
        if (!output.conditioning) {
            throw std::runtime_error("Cannot open condition-estimate output");
        }
        output.conditioning << "elapsed_ma,age_ma,cells,cell_area_min,condition_estimate\n";
    }

    const EasyRoInput& easy_input = input.easy_ro();
    const EasyRoModel easy_ro({.preexponential = easy_input.preexponential,
                               .gas_constant = easy_input.gas_constant,
                               .activation_energies = easy_input.activation_energies,
                               .weights = easy_input.weights});

    const std::span<const BasinConfiguration> used_configurations =
        configurations.first(configuration_count);
    std::optional<LayeredMeshLayout> layered_layout;
    std::vector<LayeredCellId> layered_cell_ids;
    const BasinConfiguration& oldest = configurations.front();
    Mesh2D mesh = [&]() {
        if (options.layered_mesh) {
            layered_layout = make_layered_mesh_layout(used_configurations, options.cell_size);
            LayeredMesh layered = make_layered_mesh(oldest, *layered_layout);
            layered_cell_ids = std::move(layered.cell_ids);
            return std::move(layered.mesh);
        }
        return make_preferred_mesh(oldest, options.cell_size, options.thin_layer_cell_fraction,
                                   options.quad_dominant);
    }();
    CellFields fields = make_cell_fields(mesh, oldest, input);
    Vector temperature = initial_temperature(mesh, oldest, fields);
    update_properties(values(temperature), fields);
    std::vector<double> maturity = easy_ro.initial_state(mesh.cells().size());

    BasinForwardResult result;
    result.configurations = static_cast<int>(configuration_count);
    update_mesh_statistics(result, mesh);
    save_state(output, mesh, temperature, fields, easy_ro, maturity, 0.0, oldest.age_ma, "initial",
               0);
    double elapsed = 0.0;
    double accumulated_energy_imbalance = 0.0;
    double accumulated_basal_energy = 0.0;
    for (std::size_t epoch = 0; epoch + 1 < configuration_count; ++epoch) {
        const auto epoch_wall_start = std::chrono::steady_clock::now();
        const BasinConfiguration& first = configurations[epoch];
        const BasinConfiguration& second = configurations[epoch + 1];
        const double duration = (first.age_ma - second.age_ma) * kSecondsPerMa;
        const double epoch_end = elapsed + duration;
        const int steps_before = result.accepted_steps;
        const int nonlinear_before = result.nonlinear_iterations;
        const int linear_before = result.linear_iterations;
        const std::size_t old_cell_count = mesh.cells().size();

        int saved_states = 0;
        bool topology_regularized = false;
        std::unique_ptr<MaterialMotion> motion;
        if (!options.fixed_mesh) {
            try {
                motion = std::make_unique<MaterialMotion>(mesh, first, second, duration);
            } catch (const std::exception&) {
                motion = std::make_unique<MaterialMotion>(mesh, first, second, duration, true);
                topology_regularized = true;
                ++result.topology_regularized_epochs;
            }
            fields.velocity_z.assign(motion->velocity_z().begin(), motion->velocity_z().end());
        }
        std::unique_ptr<HeatSystem> system =
            make_system(mesh, fields, boundary_conditions(first, second, elapsed, duration));

        double current_time = elapsed;
        while (current_time < epoch_end) {
            const double old_time = current_time;
            const double target = std::min(epoch_end, old_time + timestep);
            Vector old_temperature = temperature;
            const double energy_before = total_energy(mesh, old_temperature, fields);
            if (!options.fixed_mesh) {
                const double fraction = (target - elapsed) / duration;
                motion->set_position(fraction);
                motion->set_porosity(fraction, fields.porosity);
                system->update_geometry(mesh);
            }

            TimeIntegrator integrator;
            integrator.set_initial_solution(old_time, temperature);
            integrator.set_timestep(target - old_time);
            const TimeIntegrationResult step = integrator.advance_to(
                *system, target, nonlinear_request(), linear_request(options.estimate_condition));
            try {
                accumulate_result(result, step);
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    "Epoch " + std::to_string(first.age_ma) + " -> " +
                    std::to_string(second.age_ma) + " Ma at age " +
                    std::to_string(first.age_ma - ((target - elapsed) / kSecondsPerMa)) +
                    " Ma: " + error.what());
            }
            temperature = integrator.current_snapshot().solution;
            const HeatEnergyRates rates = system->energy_rates(target, temperature);
            const double energy_after = total_energy(mesh, temperature, fields);
            const double step_duration = target - old_time;
            accumulated_energy_imbalance +=
                energy_after - energy_before +
                (step_duration *
                 (rates.surface_outflow - rates.basal_inflow - rates.heat_production));
            accumulated_basal_energy += step_duration * rates.basal_inflow;
            result.global_energy_balance = std::abs(accumulated_energy_imbalance) /
                                           std::max(std::abs(accumulated_basal_energy), 1.0);
            output.energy_balance << std::setprecision(17) << target / kSecondsPerMa << ','
                                  << first.age_ma - ((target - elapsed) / kSecondsPerMa) << ",step,"
                                  << result.global_energy_balance << '\n';
            if (step.minimum_reciprocal_condition_estimate) {
                const double reciprocal = *step.minimum_reciprocal_condition_estimate;
                const double condition =
                    reciprocal > 0.0 ? 1.0 / reciprocal : std::numeric_limits<double>::infinity();
                result.maximum_condition_estimate =
                    std::max(result.maximum_condition_estimate, condition);
                output.conditioning << std::setprecision(17) << target / kSecondsPerMa << ','
                                    << first.age_ma - ((target - elapsed) / kSecondsPerMa) << ','
                                    << mesh.cells().size() << ',' << minimum_cell_area(mesh) << ','
                                    << condition << '\n';
            }
            easy_ro.advance(values(old_temperature), values(temperature), target - old_time,
                            maturity);
            current_time = target;

            const double fraction = (target - elapsed) / duration;
            const int expected_saved =
                std::min(options.saved_states_per_epoch,
                         static_cast<int>(
                             std::floor((fraction * options.saved_states_per_epoch) + 1.0e-10)));
            if (expected_saved > saved_states || target == epoch_end) {
                saved_states = std::max(saved_states, expected_saved);
                save_state(output, mesh, temperature, fields, easy_ro, maturity,
                           target / kSecondsPerMa,
                           first.age_ma - ((target - elapsed) / kSecondsPerMa), "epoch",
                           step.nonlinear_iterations);
            }
        }

        elapsed = epoch_end;
        result.final_age_ma = second.age_ma;
        if (options.fixed_mesh) {
            const double epoch_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_wall_start)
                    .count();
            output.epochs << std::setprecision(17) << first.age_ma << ',' << second.age_ma << ','
                          << result.accepted_steps - steps_before << ','
                          << result.nonlinear_iterations - nonlinear_before << ','
                          << result.linear_iterations - linear_before << ',' << old_cell_count
                          << ',' << old_cell_count << ',' << topology_regularized << ','
                          << epoch_seconds << '\n';
            break;
        }

        save_state(output, mesh, temperature, fields, easy_ro, maturity, elapsed / kSecondsPerMa,
                   second.age_ma, "before_transition", 0);

        std::vector<LayeredCellId> new_layered_cell_ids;
        Mesh2D new_mesh = [&]() {
            if (options.layered_mesh) {
                LayeredMesh layered = make_layered_mesh(second, *layered_layout);
                new_layered_cell_ids = std::move(layered.cell_ids);
                return std::move(layered.mesh);
            }
            return make_preferred_mesh(second, options.cell_size, options.thin_layer_cell_fraction,
                                       options.quad_dominant);
        }();
        TransferredState transferred =
            transfer_state(mesh, temperature, fields, maturity, std::move(new_mesh), first, second,
                           input, easy_ro, options.cell_size, true);
        accumulated_energy_imbalance += transferred.energy_imbalance;
        result.global_energy_balance = std::abs(accumulated_energy_imbalance) /
                                       std::max(std::abs(accumulated_basal_energy), 1.0);
        output.transitions << std::setprecision(17) << second.age_ma << ',' << mesh.cells().size()
                           << ',' << transferred.mesh.cells().size() << '\n';
        output.energy_balance << std::setprecision(17) << elapsed / kSecondsPerMa << ','
                              << second.age_ma << ",transition," << result.global_energy_balance
                              << '\n';

        mesh = std::move(transferred.mesh);
        if (options.layered_mesh) {
            layered_cell_ids = std::move(new_layered_cell_ids);
        }
        fields = std::move(transferred.fields);
        temperature = std::move(transferred.temperature);
        maturity = std::move(transferred.maturity);
        update_properties(values(temperature), fields);
        update_mesh_statistics(result, mesh);
        save_state(output, mesh, temperature, fields, easy_ro, maturity, elapsed / kSecondsPerMa,
                   second.age_ma, "after_transition", 0);

        const double epoch_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_wall_start)
                .count();
        output.epochs << std::setprecision(17) << first.age_ma << ',' << second.age_ma << ','
                      << result.accepted_steps - steps_before << ','
                      << result.nonlinear_iterations - nonlinear_before << ','
                      << result.linear_iterations - linear_before << ',' << old_cell_count << ','
                      << mesh.cells().size() << ',' << topology_regularized << ',' << epoch_seconds
                      << '\n';
    }

    if (options.comparison_points) {
        const std::vector<ComparisonPoint> points =
            read_comparison_points(*options.comparison_points);
        write_comparison_temperatures(output_directory / "comparison-temperatures.csv", mesh,
                                      temperature, points);
    }

    result.wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    write_pvd(output_directory / "temperature.pvd", output.series);
    output.statistics.flush();
    output.epochs.flush();
    output.transitions.flush();
    output.energy_balance.flush();
    if (output.conditioning) {
        output.conditioning.flush();
    }
    const std::uintmax_t output_bytes = directory_size(output_directory);
    std::ofstream summary(output_directory / "summary.csv");
    summary << "configurations,accepted_steps,nonlinear_iterations,"
               "maximum_nonlinear_iterations_per_step,linear_iterations,final_age_ma,"
               "topology_regularized_epochs,"
               "global_energy_balance,cells_min,cells_max,"
               "cell_diameter_max,wall_seconds,condition_estimate_max,output_bytes\n"
            << result.configurations << ',' << result.accepted_steps << ','
            << result.nonlinear_iterations << ',' << result.maximum_nonlinear_iterations_per_step
            << ',' << result.linear_iterations << ',' << result.final_age_ma << ','
            << result.topology_regularized_epochs << ',' << result.global_energy_balance << ','
            << result.minimum_cells << ',' << result.maximum_cells << ','
            << result.maximum_cell_diameter << ',' << result.wall_seconds << ','
            << result.maximum_condition_estimate << ',' << output_bytes << '\n';
    return result;
}
