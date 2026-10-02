#include "forward/layered_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

constexpr double kRelativeThicknessTolerance = 1.0e-10;

double maximum_thickness(const BasinLayerProfile& layer) {
    double result = 0.0;
    for (std::size_t index = 0; index < layer.x.size(); ++index) {
        result = std::max(result, layer.bottom[index] - layer.top[index]);
    }
    return result;
}

void append_geometry_events(std::vector<double>& points, const BasinLayerProfile& layer) {
    const double tolerance = kRelativeThicknessTolerance * std::max(1.0, maximum_thickness(layer));
    points.push_back(layer.x.front());
    points.push_back(layer.x.back());
    for (std::size_t index = 0; index < layer.x.size(); ++index) {
        if (layer.bottom[index] - layer.top[index] > tolerance) {
            continue;
        }
        const std::size_t first = index == 0 ? 0 : index - 1;
        const std::size_t last = std::min(index + 1, layer.x.size() - 1);
        for (std::size_t neighbor = first; neighbor <= last; ++neighbor) {
            points.push_back(layer.x[neighbor]);
        }
    }
}

Point2D point(const double x, const double top, const double thickness, const double eta) {
    return {.x = x, .z = top + eta * thickness};
}

void append_cell(std::vector<Mesh2D::Cell>& quads, std::vector<LayeredCellId>& quad_ids,
                 std::vector<Mesh2D::Cell>& triangles, std::vector<LayeredCellId>& triangle_ids,
                 const BasinLayerProfile& layer, const std::size_t column, const std::size_t row,
                 const double left_x, const double right_x, const double left_top,
                 const double right_top, const double left_thickness, const double right_thickness,
                 const std::size_t rows, const double tolerance) {
    const double eta_top = static_cast<double>(row) / static_cast<double>(rows);
    const double eta_bottom = static_cast<double>(row + 1) / static_cast<double>(rows);
    const Point2D top_left = point(left_x, left_top, left_thickness, eta_top);
    const Point2D top_right = point(right_x, right_top, right_thickness, eta_top);
    const Point2D bottom_right = point(right_x, right_top, right_thickness, eta_bottom);
    const Point2D bottom_left = point(left_x, left_top, left_thickness, eta_bottom);
    const LayeredCellId identifier{.layer_id = layer.id, .column = column, .row = row};

    if (left_thickness <= tolerance) {
        triangles.push_back(
            {.vertices = {top_left, top_right, bottom_right},
             .edge_kinds = {BoundaryKind::top, BoundaryKind::right, BoundaryKind::bottom},
             .id = layer.id});
        triangle_ids.push_back(identifier);
    } else if (right_thickness <= tolerance) {
        triangles.push_back(
            {.vertices = {top_left, top_right, bottom_left},
             .edge_kinds = {BoundaryKind::top, BoundaryKind::bottom, BoundaryKind::left},
             .id = layer.id});
        triangle_ids.push_back(identifier);
    } else {
        quads.push_back({.vertices = {top_left, top_right, bottom_right, bottom_left},
                         .edge_kinds = {BoundaryKind::top, BoundaryKind::right,
                                        BoundaryKind::bottom, BoundaryKind::left},
                         .id = layer.id});
        quad_ids.push_back(identifier);
    }
}

} // namespace

LayeredMeshLayout make_layered_mesh_layout(const std::span<const BasinConfiguration> configurations,
                                           const double cell_size) {
    if (configurations.empty()) {
        throw std::invalid_argument("Layered mesh layout requires basin configurations");
    }
    if (!std::isfinite(cell_size) || !(cell_size > 0.0)) {
        throw std::invalid_argument("Layered mesh cell size must be finite and positive");
    }

    double minimum_x = std::numeric_limits<double>::max();
    double maximum_x = std::numeric_limits<double>::lowest();
    std::unordered_map<int, double> layer_thickness;
    LayeredMeshLayout result;
    for (const BasinConfiguration& configuration : configurations) {
        for (const BasinLayerProfile& layer : configuration.layers) {
            if (layer.x.size() < 2 || layer.x.size() != layer.top.size() ||
                layer.x.size() != layer.bottom.size()) {
                throw std::invalid_argument("Invalid basin layer profile for layered mesh");
            }
            minimum_x = std::min(minimum_x, layer.x.front());
            maximum_x = std::max(maximum_x, layer.x.back());
            layer_thickness[layer.id] =
                std::max(layer_thickness[layer.id], maximum_thickness(layer));
            append_geometry_events(result.x, layer);
        }
    }
    const std::size_t regular_intervals = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil((maximum_x - minimum_x) / cell_size)));
    for (std::size_t interval = 0; interval <= regular_intervals; ++interval) {
        const double fraction =
            static_cast<double>(interval) / static_cast<double>(regular_intervals);
        result.x.push_back(((1.0 - fraction) * minimum_x) + (fraction * maximum_x));
    }
    std::sort(result.x.begin(), result.x.end());
    result.x.erase(std::unique(result.x.begin(), result.x.end()), result.x.end());
    for (const auto& [layer_id, thickness] : layer_thickness) {
        result.layer_rows[layer_id] =
            std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(thickness / cell_size)));
    }
    return result;
}

LayeredMesh make_layered_mesh(const BasinConfiguration& configuration,
                              const LayeredMeshLayout& layout) {
    if (layout.x.size() < 2) {
        throw std::invalid_argument("Layered mesh layout requires at least two x coordinates");
    }
    std::vector<Mesh2D::Cell> quads;
    std::vector<Mesh2D::Cell> triangles;
    std::vector<LayeredCellId> quad_ids;
    std::vector<LayeredCellId> triangle_ids;

    for (const BasinLayerProfile& layer : configuration.layers) {
        const auto row_count = layout.layer_rows.find(layer.id);
        if (row_count == layout.layer_rows.end()) {
            throw std::invalid_argument("Layer is absent from layered mesh layout");
        }
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
            for (std::size_t row = 0; row < row_count->second; ++row) {
                append_cell(quads, quad_ids, triangles, triangle_ids, layer, column, row, left_x,
                            right_x, left_top, right_top, left_thickness, right_thickness,
                            row_count->second, tolerance);
            }
        }
    }

    std::vector<Mesh2D::Cell> cells;
    cells.reserve(quads.size() + triangles.size());
    cells.insert(cells.end(), std::make_move_iterator(quads.begin()),
                 std::make_move_iterator(quads.end()));
    cells.insert(cells.end(), std::make_move_iterator(triangles.begin()),
                 std::make_move_iterator(triangles.end()));
    std::vector<LayeredCellId> identifiers;
    identifiers.reserve(quad_ids.size() + triangle_ids.size());
    identifiers.insert(identifiers.end(), quad_ids.begin(), quad_ids.end());
    identifiers.insert(identifiers.end(), triangle_ids.begin(), triangle_ids.end());

    Mesh2D mesh = Mesh2D::from_cells(cells);
    if (mesh.cells().size() != identifiers.size()) {
        throw std::logic_error("Layered mesh cell identifiers do not match mesh cells");
    }
    return {.mesh = std::move(mesh), .cell_ids = std::move(identifiers)};
}
