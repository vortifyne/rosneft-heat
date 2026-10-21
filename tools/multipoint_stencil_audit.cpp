#include "basin/basin_input.hpp"
#include "basin/basin_state.hpp"
#include "basin/layered_mesh.hpp"
#include "discretization/multipoint_flux.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr double kSnapshotAgeMa = 5.0;

const BasinConfiguration& configuration_at(const BasinInput& input, const double age_ma) {
    const auto configurations = input.configurations();
    const auto found = std::find_if(configurations.begin(), configurations.end(),
                                    [age_ma](const auto& value) { return value.age_ma == age_ma; });
    if (found == configurations.end()) {
        throw std::runtime_error("Requested snapshot age is absent from basin input");
    }
    return *found;
}

void write_entries(const std::filesystem::path& path, const Mesh2D& mesh,
                   const MultipointStencilAudit& audit) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open multipoint stencil audit output");
    }
    output << "cell,cell_x,cell_z,vertices,material,area,max_edge,elongation,"
              "target_face,target_is_internal,restricted_pair_found,"
              "general_pair_found,general_pair_uses_target,target_harmonic_point_on_face,"
              "general_harmonic_points_on_faces,general_coefficient_sum,"
              "general_qtkq_condition,general_relative_reconstruction_defect,"
              "closest_normalized_negative_coefficient\n"
           << std::setprecision(17);
    for (const MultipointStencilAuditEntry& entry : audit.entries) {
        const TQMesh::Facet* cell = mesh.cells()[entry.cell];
        const auto center = cell->xy();
        const double max_edge = cell->max_edge_length();
        output << entry.cell << ',' << center.x << ',' << center.y << ',' << cell->n_vertices()
               << ',' << cell->color() << ',' << cell->area() << ',' << max_edge << ','
               << max_edge * max_edge / cell->area() << ',' << entry.target_face << ','
               << entry.target_is_internal << ',' << entry.restricted_pair_found << ','
               << entry.general_pair_found << ',' << entry.general_pair_uses_target << ','
               << entry.target_harmonic_point_on_face << ','
               << entry.general_harmonic_points_on_faces << ',' << entry.general_coefficient_sum
               << ',' << entry.general_qtkq_condition << ','
               << entry.general_relative_reconstruction_defect << ','
               << entry.closest_normalized_negative_coefficient << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            throw std::invalid_argument(
                "usage: multipoint_stencil_audit GEOMETRY_INPUT GEOMETRY_LABEL H "
                "initial|constant OUTPUT_CSV");
        }
        const std::filesystem::path geometry_path = argv[1];
        const std::string geometry_label = argv[2];
        const double cell_size = std::stod(argv[3]);
        const std::string_view conductivity_mode = argv[4];
        const std::filesystem::path output_path = argv[5];
        if (!std::isfinite(cell_size) || !(cell_size > 0.0)) {
            throw std::invalid_argument("H must be finite and positive");
        }

        const BasinInput input = BasinInput::read(geometry_path);
        const BasinConfiguration& configuration = configuration_at(input, kSnapshotAgeMa);
        const LayeredMeshLayout layout =
            make_layered_mesh_layout(input.configurations(), cell_size);
        Mesh2D mesh = make_layered_mesh(configuration, layout);

        const EasyRoInput& parameters = input.easy_ro();
        EasyRoModel easy_ro({parameters.preexponential, parameters.gas_constant,
                             parameters.activation_energies, parameters.weights});
        BasinState state = make_initial_basin_state(configuration, input, std::move(mesh), easy_ro,
                                                    configuration.surface_temperature,
                                                    configuration.basal_heat_flux);
        if (conductivity_mode == "constant") {
            std::ranges::fill(state.fields.conductivity, 1.0);
        } else if (conductivity_mode != "initial") {
            throw std::invalid_argument("Conductivity mode must be initial or constant");
        }
        MultipointFlux flux(state.mesh);
        const auto zero_flux = [](Point2D) { return 0.0; };
        const MultipointStencilAudit audit = flux.audit_stencils(
            state.fields.conductivity,
            {.surface_temperature = [&configuration](const Point2D point) {
                 return configuration.surface_temperature.at(point.x);
             },
             .basal_inward_flux = zero_flux,
             .left_inward_flux = zero_flux,
             .right_inward_flux = zero_flux});

        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        write_entries(output_path, state.mesh, audit);
        std::cout << std::setprecision(17)
                  << "geometry,h,conductivity,cells,audited_sides,restricted_failures,general_failures,"
                     "general_pairs_without_target,target_harmonic_points_outside_faces,"
                     "selected_harmonic_points_outside_faces,maximum_qtkq_condition,"
                     "maximum_relative_reconstruction_defect\n"
                  << geometry_label << ',' << cell_size << ',' << conductivity_mode << ','
                  << state.mesh.cells().size() << ','
                  << audit.entries.size() << ',' << audit.restricted_failures << ','
                  << audit.general_failures << ',' << audit.general_pairs_without_target << ','
                  << audit.target_harmonic_points_outside_faces << ','
                  << audit.selected_harmonic_points_outside_faces << ','
                  << audit.maximum_qtkq_condition << ','
                  << audit.maximum_relative_reconstruction_defect << '\n';
        return audit.general_failures == 0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
