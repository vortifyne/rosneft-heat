#include "basin/layered_mesh.hpp"
#include "mesh/strip_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

MeshRegion layer_region(const BasinLayerGeometry& layer, const std::size_t begin,
                        const std::size_t end) {
    MeshRegion region{.id = layer.id, .boundary = {}, .edge_kinds = {}};
    const auto append = [&](const Point2D point) {
        if (region.boundary.empty() || region.boundary.back().x != point.x ||
            region.boundary.back().z != point.z) {
            region.boundary.push_back(point);
        }
    };
    for (std::size_t i = begin; i <= end; ++i) {
        append({layer.x[i], layer.top[i]});
    }
    for (std::size_t i = end + 1; i-- > begin;) {
        append({layer.x[i], layer.bottom[i]});
    }
    if (region.boundary.back().x == region.boundary.front().x &&
        region.boundary.back().z == region.boundary.front().z) {
        region.boundary.pop_back();
    }
    for (std::size_t i = 0; i < region.boundary.size(); ++i) {
        const auto a = region.boundary[i], b = region.boundary[(i + 1) % region.boundary.size()];
        if (a.x != b.x) {
            region.edge_kinds.push_back(a.x < b.x ? BoundaryKind::top : BoundaryKind::bottom);
        } else {
            region.edge_kinds.push_back(a.x == layer.x[begin] ? BoundaryKind::left
                                                              : BoundaryKind::right);
        }
    }
    return region;
}

} // namespace

Mesh2D make_layer_strip_mesh(const BasinGeometry& geometry, const double cell_size) {
    if (!std::isfinite(cell_size) || !(cell_size > 0.)) {
        throw std::invalid_argument("Layer strip mesh requires a positive cell size");
    }
    std::vector<MeshRegion> regions;
    for (const auto& region : geometry.layers) {
        const auto layer = geometry.layer(region.id);
        if (layer.x.size() < 2 || layer.x.size() != layer.top.size() ||
            layer.x.size() != layer.bottom.size()) {
            throw std::invalid_argument("Invalid layer strip geometry");
        }
        std::size_t edge = 0;
        while (edge + 1 < layer.x.size()) {
            if (layer.bottom[edge] == layer.top[edge] &&
                layer.bottom[edge + 1] == layer.top[edge + 1]) {
                ++edge;
                continue;
            }
            const std::size_t begin = edge;
            std::size_t end = edge + 1;
            while (end + 1 < layer.x.size() && layer.bottom[end] > layer.top[end]) {
                ++end;
            }
            regions.push_back(layer_region(layer, begin, end));
            edge = end;
        }
    }
    Mesh2D mesh = make_strip_mesh(regions, cell_size);
    validate_layered_mesh_coverage(mesh, geometry);
    return mesh;
}
