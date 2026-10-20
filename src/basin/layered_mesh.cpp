#include "basin/layered_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

constexpr double kRelativeThicknessTolerance = 1.0e-10;

double maximum_thickness(const BasinLayerGeometry& layer) {
    double result = 0.0;
    for (std::size_t index = 0; index < layer.x.size(); ++index) {
        result = std::max(result, layer.bottom[index] - layer.top[index]);
    }
    return result;
}

void append_geometry_events(std::vector<double>& points, const BasinLayerGeometry& layer) {
    // Geometry simplification belongs before meshing: dropping a profile endpoint can
    // remove a whole strip of a layer and turn its contacts into external boundaries.
    points.insert(points.end(), layer.x.begin(), layer.x.end());
}

std::vector<double> make_horizontal_axis(std::vector<double> events, const double cell_size) {
    std::sort(events.begin(), events.end());
    events.erase(std::unique(events.begin(), events.end()), events.end());

    std::vector<double> result;
    for (std::size_t interval = 0; interval + 1 < events.size(); ++interval) {
        const double left = events[interval];
        const double right = events[interval + 1];
        const std::size_t count = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil((right - left) / cell_size)));
        for (std::size_t part = 0; part < count; ++part) {
            const double fraction = static_cast<double>(part) / static_cast<double>(count);
            result.push_back(((1.0 - fraction) * left) + (fraction * right));
        }
    }
    result.push_back(events.back());
    return result;
}

Point2D point(const double x, const double top, const double thickness, const double eta) {
    return {.x = x, .z = top + eta * thickness};
}

void append_cell(std::vector<Mesh2D::Cell>& quads, std::vector<Mesh2D::Cell>& triangles,
                 const BasinLayerGeometry& layer, const std::size_t row, const double left_x,
                 const double right_x, const double left_top, const double right_top,
                 const double left_thickness, const double right_thickness, const std::size_t rows,
                 const double tolerance) {
    const double eta_top = static_cast<double>(row) / static_cast<double>(rows);
    const double eta_bottom = static_cast<double>(row + 1) / static_cast<double>(rows);
    const Point2D top_left = point(left_x, left_top, left_thickness, eta_top);
    const Point2D top_right = point(right_x, right_top, right_thickness, eta_top);
    const Point2D bottom_right = point(right_x, right_top, right_thickness, eta_bottom);
    const Point2D bottom_left = point(left_x, left_top, left_thickness, eta_bottom);
    if (left_thickness <= tolerance) {
        triangles.push_back(
            {.vertices = {top_left, top_right, bottom_right},
             .edge_kinds = {BoundaryKind::top, BoundaryKind::right, BoundaryKind::bottom},
             .id = layer.id});
    } else if (right_thickness <= tolerance) {
        triangles.push_back(
            {.vertices = {top_left, top_right, bottom_left},
             .edge_kinds = {BoundaryKind::top, BoundaryKind::bottom, BoundaryKind::left},
             .id = layer.id});
    } else {
        quads.push_back({.vertices = {top_left, top_right, bottom_right, bottom_left},
                         .edge_kinds = {BoundaryKind::top, BoundaryKind::right,
                                        BoundaryKind::bottom, BoundaryKind::left},
                         .id = layer.id});
    }
}

} // namespace

LayeredMeshLayout make_layered_mesh_layout(const std::span<const BasinGeometry> configurations,
                                           const double cell_size) {
    if (configurations.empty()) {
        throw std::invalid_argument("Layered mesh layout requires basin configurations");
    }
    if (!std::isfinite(cell_size) || !(cell_size > 0.0)) {
        throw std::invalid_argument("Layered mesh cell size must be finite and positive");
    }

    std::vector<double> events;
    for (const BasinGeometry& configuration : configurations) {
        for (const auto& region : configuration.layers) {
            const auto layer = configuration.layer(region.id);
            if (layer.x.size() < 2 || layer.x.size() != layer.top.size() ||
                layer.x.size() != layer.bottom.size()) {
                throw std::invalid_argument("Invalid basin layer profile for layered mesh");
            }
            append_geometry_events(events, layer);
        }
    }
    if (events.size() < 2) {
        throw std::invalid_argument("Layered mesh layout requires non-empty geometry");
    }
    return {.x = make_horizontal_axis(std::move(events), cell_size), .cell_size = cell_size};
}

Mesh2D make_layered_mesh(const BasinGeometry& configuration, const LayeredMeshLayout& layout) {
    if (layout.x.size() < 2) {
        throw std::invalid_argument("Layered mesh layout requires at least two x coordinates");
    }
    std::vector<Mesh2D::Cell> quads;
    std::vector<Mesh2D::Cell> triangles;

    for (const auto& region : configuration.layers) {
        const auto layer = configuration.layer(region.id);
        const std::size_t row_count = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil(maximum_thickness(layer) / layout.cell_size)));
        const double tolerance =
            kRelativeThicknessTolerance * std::max(1.0, maximum_thickness(layer));
        for (std::size_t column = 0; column + 1 < layout.x.size(); ++column) {
            const double left_x = layout.x[column];
            const double right_x = layout.x[column + 1];
            if (left_x < layer.x.front() || right_x > layer.x.back()) {
                continue;
            }
            const double left_top = layer.top_at(left_x);
            const double right_top = layer.top_at(right_x);
            const double left_thickness = std::max(0.0, layer.bottom_at(left_x) - left_top);
            const double right_thickness = std::max(0.0, layer.bottom_at(right_x) - right_top);
            if (left_thickness + right_thickness <= tolerance) {
                continue;
            }
            for (std::size_t row = 0; row < row_count; ++row) {
                append_cell(quads, triangles, layer, row, left_x, right_x, left_top, right_top,
                            left_thickness, right_thickness, row_count, tolerance);
            }
        }
    }

    std::vector<Mesh2D::Cell> cells;
    cells.reserve(quads.size() + triangles.size());
    cells.insert(cells.end(), std::make_move_iterator(quads.begin()),
                 std::make_move_iterator(quads.end()));
    cells.insert(cells.end(), std::make_move_iterator(triangles.begin()),
                 std::make_move_iterator(triangles.end()));
    Mesh2D mesh = Mesh2D::from_cells(cells);
    validate_layered_mesh_coverage(mesh, configuration);
    return mesh;
}

LayeredMeshLayout make_layered_mesh_layout(const std::span<const BasinConfiguration> configurations,
                                           const double cell_size) {
    std::vector<BasinGeometry> geometries;
    geometries.reserve(configurations.size());
    for (const auto& configuration : configurations) {
        geometries.push_back(make_basin_geometry(configuration));
    }
    return make_layered_mesh_layout(geometries, cell_size);
}

Mesh2D make_layered_mesh(const BasinConfiguration& configuration, const LayeredMeshLayout& layout) {
    return make_layered_mesh(make_basin_geometry(configuration), layout);
}
