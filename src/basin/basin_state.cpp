#include "basin/basin_state.hpp"

#include "mesh/mesh_transfer.hpp"
#include "physics/thermophysical_properties.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace {

constexpr FluidThermophysicalProperties kFluid{
    .thermal_conductivity = 0.7,
    .density = 1040.0,
    .specific_heat = 4184.0,
};

BasinCellFields make_cell_fields(const Mesh2D& mesh, const BasinConfiguration& configuration,
                                 const BasinInput& input) {
    const std::size_t count = mesh.cells().size();
    BasinCellFields fields{
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

Vector initial_temperature(const Mesh2D& mesh, const BasinConfiguration& configuration,
                           const BasinCellFields& fields, const ScalarProfile& surface_profile,
                           const ScalarProfile& basal_profile) {
    Vector result(static_cast<Vector::Index>(mesh.cells().size()));
    const auto& top_layer = configuration.layers.front();
    for (const TQMesh::Facet* cell : mesh.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        const double surface_temperature = surface_profile.at(cell->xy().x);
        const double conductivity =
            evaluate_thermophysical_properties(surface_temperature, fields.porosity[index],
                                               *fields.lithotypes[index], kFluid)
                .thermal_conductivity;
        const double gradient = basal_profile.at(cell->xy().x) / conductivity;
        result[cell->index()] =
            surface_temperature + (gradient * (cell->xy().y - top_layer.top_at(cell->xy().x)));
    }
    return result;
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

double transferred_energy(const BasinState& old_state, const BasinState& new_state,
                          const CellTransferMap& transfer_map) {
    const double before = total_energy(old_state);
    double after = 0.0;
    for (const TQMesh::Facet* cell : new_state.mesh.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        if (!transfer_map.donor_cells[index]) {
            continue;
        }
        after += cell->area() * cell_enthalpy(new_state.temperature[cell->index()],
                                              new_state.fields.porosity[index],
                                              *new_state.fields.lithotypes[index]);
    }
    return after - before;
}

} // namespace

std::span<const double> vector_values(const Vector& vector) {
    return {vector.native().data(), static_cast<std::size_t>(vector.size())};
}

void update_thermophysical_properties(const std::span<const double> temperature,
                                      BasinCellFields& fields) {
    evaluate_thermophysical_property_fields(temperature, fields.porosity, fields.lithotypes, kFluid,
                                            fields.conductivity, fields.heat_capacity,
                                            fields.heat_production);
}

BasinState make_initial_basin_state(const BasinConfiguration& configuration,
                                    const BasinInput& input, Mesh2D mesh,
                                    const EasyRoModel& easy_ro,
                                    const ScalarProfile& surface_temperature,
                                    const ScalarProfile& basal_heat_flux) {
    BasinCellFields fields = make_cell_fields(mesh, configuration, input);
    Vector temperature =
        initial_temperature(mesh, configuration, fields, surface_temperature, basal_heat_flux);
    update_thermophysical_properties(vector_values(temperature), fields);
    std::vector<double> maturity = easy_ro.initial_state(mesh.cells().size());
    return {.mesh = std::move(mesh),
            .fields = std::move(fields),
            .temperature = std::move(temperature),
            .maturity = std::move(maturity)};
}

std::unique_ptr<HeatSystem> make_heat_system(const Mesh2D& mesh, BasinCellFields& fields,
                                             HeatBoundaryConditions boundary_conditions) {
    auto updater =
        [&fields](const std::span<const double> temperature, const std::span<double> conductivity,
                  const std::span<double> heat_capacity, const std::span<double> heat_production) {
            evaluate_thermophysical_property_fields(temperature, fields.porosity, fields.lithotypes,
                                                    kFluid, conductivity, heat_capacity,
                                                    heat_production);
        };
    auto system = std::make_unique<HeatSystem>(mesh, fields.conductivity, fields.heat_capacity,
                                               fields.heat_production,
                                               std::move(boundary_conditions), updater);
    system->set_implicit_nonorthogonal_correction(true);
    return system;
}

double total_energy(const BasinState& state) {
    double result = 0.0;
    for (const TQMesh::Facet* cell : state.mesh.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        result += cell->area() * cell_enthalpy(state.temperature[cell->index()],
                                               state.fields.porosity[index],
                                               *state.fields.lithotypes[index]);
    }
    return result;
}

BasinStateTransfer transfer_basin_state(const BasinState& old_state, Mesh2D new_mesh,
                                        const BasinConfiguration& old_configuration,
                                        const BasinConfiguration& new_configuration,
                                        const BasinInput& input, const EasyRoModel& easy_ro,
                                        const ScalarProfile& surface_temperature,
                                        const double cell_size) {
    BasinCellFields new_fields = make_cell_fields(new_mesh, new_configuration, input);
    CellTransferMap transfer_map = make_cell_transfer_map(old_state.mesh, new_mesh);
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
            continue;
        }
        const Point2D target_point = transfer_map.target_points[index];
        const double target_eta = new_fields.material_eta[index];
        double nearest_distance = std::numeric_limits<double>::max();
        std::optional<std::size_t> nearest;
        for (const TQMesh::Facet* source_cell : old_state.mesh.cells()) {
            const std::size_t source_index = static_cast<std::size_t>(source_cell->index());
            if (static_cast<int>(old_state.fields.layer_id[source_index]) != layer.id) {
                continue;
            }
            const double horizontal = (source_cell->xy().x - target_point.x) / cell_size;
            const double vertical = old_state.fields.material_eta[source_index] - target_eta;
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
        transfer_map.target_points[index] = {old_state.mesh.cells()[*nearest]->xy().x,
                                             old_state.mesh.cells()[*nearest]->xy().y};
    }

    const CellFieldSampler temperature_sampler =
        make_cell_field_sampler(old_state.mesh, vector_values(old_state.temperature));
    const auto initialize_temperature = [&](const std::size_t index, const Point2D point) {
        const auto& layer = new_configuration.layer(new_mesh.cells()[index]->color());
        const double top = layer.top_at(point.x);
        const double bottom = layer.bottom_at(point.x);
        const double eta = std::clamp((point.z - top) / (bottom - top), 0.0, 1.0);
        const double surface = surface_temperature.at(point.x);
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
        old_state.mesh, vector_values(old_state.temperature), transfer_map, initialize_temperature);
    Vector new_temperature(static_cast<Vector::Index>(new_mesh.cells().size()));
    for (std::size_t index = 0; index < transferred_temperature.size(); ++index) {
        new_temperature[static_cast<Vector::Index>(index)] = transferred_temperature[index];
    }

    std::vector<double> new_maturity = easy_ro.initial_state(new_mesh.cells().size());
    for (std::size_t reaction = 0; reaction < easy_ro.reaction_count(); ++reaction) {
        std::vector<double> old_reaction(old_state.mesh.cells().size());
        for (std::size_t cell = 0; cell < old_reaction.size(); ++cell) {
            old_reaction[cell] = old_state.maturity[(cell * easy_ro.reaction_count()) + reaction];
        }
        const std::vector<double> transferred = transfer_cell_field(
            old_state.mesh, old_reaction, transfer_map, [](std::size_t, Point2D) { return 0.0; });
        for (std::size_t cell = 0; cell < transferred.size(); ++cell) {
            new_maturity[(cell * easy_ro.reaction_count()) + reaction] =
                std::max(0.0, transferred[cell]);
        }
    }
    BasinState new_state{.mesh = std::move(new_mesh),
                         .fields = std::move(new_fields),
                         .temperature = std::move(new_temperature),
                         .maturity = std::move(new_maturity)};
    const double energy_imbalance = transferred_energy(old_state, new_state, transfer_map);
    return {.state = std::move(new_state), .energy_imbalance = energy_imbalance};
}
