#include "basin/mesh_motion.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace {

constexpr double kMinimumTopologyThickness = 0.5;
constexpr double kMinimumTopologyThicknessFraction = 0.001;

} // namespace

BasinMeshMotion::BasinMeshMotion(Mesh2D& mesh, const BasinConfiguration& first,
                                 const BasinConfiguration& second, const double duration,
                                 const bool preserve_topology)
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
                << " material motion produces a non-positive cell " << *cell_index << " in layer "
                << cell->color() << ':';
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

void BasinMeshMotion::set_position(const double fraction) {
    std::vector<Point2D> coordinates(initial_.size());
    for (std::size_t index = 0; index < coordinates.size(); ++index) {
        coordinates[index] = {
            initial_[index].x,
            ((1.0 - fraction) * initial_[index].z) + (fraction * final_[index].z),
        };
    }
    mesh_.set_vertex_coordinates(coordinates);
}

void BasinMeshMotion::set_porosity(const double fraction, const std::span<double> porosity) const {
    for (std::size_t index = 0; index < porosity.size(); ++index) {
        porosity[index] =
            ((1.0 - fraction) * porosity_first_[index]) + (fraction * porosity_second_[index]);
    }
}

std::span<const double> BasinMeshMotion::velocity_z() const noexcept {
    return velocity_z_;
}
