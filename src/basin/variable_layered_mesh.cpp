#include "basin/layered_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace {

double distance_to_curve(const Point2D point, const BasinLayerGeometry& layer,
                         const std::vector<double>& depth) {
    double distance = std::numeric_limits<double>::infinity();
    for (std::size_t i = 1; i < layer.x.size(); ++i) {
        const double dx = layer.x[i] - layer.x[i - 1];
        const double dz = depth[i] - depth[i - 1];
        const double fraction = std::clamp(
            ((point.x - layer.x[i - 1]) * dx + (point.z - depth[i - 1]) * dz) / (dx * dx + dz * dz),
            0.0, 1.0);
        distance = std::min(distance, std::hypot(point.x - (layer.x[i - 1] + fraction * dx),
                                                 point.z - (depth[i - 1] + fraction * dz)));
    }
    return distance;
}

std::size_t choose_count(const double width, const double size,
                         const std::optional<std::size_t> previous) {
    // The interval is about transverse resolution, not vertical edge length.
    const double lower = std::max(1.0, std::ceil(width / (1.5 * size)));
    const double upper = std::max(lower, std::floor(width / (0.5 * size)));
    if (upper >= static_cast<double>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("Requested layer resolution produces too many edges");
    }
    const auto minimum = static_cast<std::size_t>(lower);
    const auto maximum = static_cast<std::size_t>(upper);
    const auto desired =
        previous.value_or(static_cast<std::size_t>(std::max(1.0, std::floor(width / size + 0.5))));
    return std::clamp(desired, minimum, maximum);
}

std::vector<Point2D> vertical_points(const BasinLayerGeometry& layer, const double x,
                                     const double size, const double tolerance,
                                     const std::optional<std::size_t> previous) {
    const double top = layer.top_at(x);
    const double height = std::max(0.0, layer.bottom_at(x) - top);
    if (height <= tolerance) {
        return {{x, top}};
    }
    const Point2D middle{x, top + 0.5 * height};
    // Diameter of a disk centred on this section's midpoint and bounded by the
    // roof and floor. For parallel lines of slope s this is height/sqrt(1+s*s).
    // Lateral domain edges do not shorten the estimate.
    const double width =
        std::min(height, 2.0 * std::min(distance_to_curve(middle, layer, layer.top),
                                        distance_to_curve(middle, layer, layer.bottom)));
    const std::size_t count = choose_count(width, size, previous);
    std::vector<Point2D> points;
    points.reserve(count + 1);
    for (std::size_t i = 0; i <= count; ++i) {
        const double eta = static_cast<double>(i) / static_cast<double>(count);
        points.push_back({x, top + eta * height});
    }
    return points;
}

void connect(std::span<const Point2D> left, std::span<const Point2D> right, const int layer_id,
             std::vector<Mesh2D::Cell>& quads, std::vector<Mesh2D::Cell>& triangles) {
    const auto nl = left.size() - 1;
    const auto nr = right.size() - 1;
    std::size_t l = 0, r = 0;
    while (l < nl || r < nr) {
        bool advance_left = l < nl;
        bool advance_right = r < nr;
        if (advance_left && advance_right) {
            const double current_left = static_cast<double>(l) / static_cast<double>(nl);
            const double current_right = static_cast<double>(r) / static_cast<double>(nr);
            const double next_left = static_cast<double>(l + 1) / static_cast<double>(nl);
            const double next_right = static_cast<double>(r + 1) / static_cast<double>(nr);
            const double together = std::abs(next_left - next_right);
            const double left_only = std::abs(next_left - current_right);
            const double right_only = std::abs(current_left - next_right);
            if (left_only < together && left_only <= right_only) {
                advance_right = false;
            } else if (right_only < together) {
                advance_left = false;
            }
        }
        const auto top = l == 0 && r == 0 ? BoundaryKind::top : BoundaryKind::interface;
        const auto bottom = l + static_cast<std::size_t>(advance_left) == nl &&
                                    r + static_cast<std::size_t>(advance_right) == nr
                                ? BoundaryKind::bottom
                                : BoundaryKind::interface;
        if (advance_left && advance_right) {
            quads.push_back({.vertices = {left[l], right[r], right[r + 1], left[l + 1]},
                             .edge_kinds = {top, BoundaryKind::right, bottom, BoundaryKind::left},
                             .id = layer_id});
        } else if (advance_left) {
            triangles.push_back({.vertices = {left[l], right[r], left[l + 1]},
                                 .edge_kinds = {top, bottom, BoundaryKind::left},
                                 .id = layer_id});
        } else {
            triangles.push_back({.vertices = {left[l], right[r], right[r + 1]},
                                 .edge_kinds = {top, BoundaryKind::right, bottom},
                                 .id = layer_id});
        }
        l += static_cast<std::size_t>(advance_left);
        r += static_cast<std::size_t>(advance_right);
    }
}

} // namespace

Mesh2D make_variable_layered_mesh(const BasinGeometry& geometry, const LayeredMeshLayout& layout) {
    if (layout.x.size() < 2 || !std::isfinite(layout.cell_size) || !(layout.cell_size > 0.0)) {
        throw std::invalid_argument("Variable layered mesh requires an axis and a positive step");
    }
    for (std::size_t i = 0; i < layout.x.size(); ++i) {
        if (!std::isfinite(layout.x[i]) || (i > 0 && !(layout.x[i] > layout.x[i - 1]))) {
            throw std::invalid_argument("Variable layered mesh requires an increasing finite axis");
        }
    }
    std::vector<Mesh2D::Cell> quads, triangles;
    for (const auto& region : geometry.layers) {
        const auto layer = geometry.layer(region.id);
        double maximum_height = 0.0;
        for (std::size_t i = 0; i < layer.x.size(); ++i) {
            maximum_height = std::max(maximum_height, layer.bottom[i] - layer.top[i]);
        }
        const double tolerance = 1.0e-10 * std::max(1.0, maximum_height);
        std::vector<Point2D> left;
        for (const double x : layout.x) {
            if (x < region.x_min || x > region.x_max) {
                continue;
            }
            const auto previous =
                left.empty() ? std::nullopt : std::optional<std::size_t>(left.size() - 1);
            auto right = vertical_points(layer, x, layout.cell_size, tolerance, previous);
            if (!left.empty()) {
                connect(left, right, region.id, quads, triangles);
            }
            left = std::move(right);
        }
    }
    quads.insert(quads.end(), std::make_move_iterator(triangles.begin()),
                 std::make_move_iterator(triangles.end()));
    auto mesh = Mesh2D::from_cells(quads);
    validate_layered_mesh_coverage(mesh, geometry);
    return mesh;
}

Mesh2D make_variable_layered_mesh(const BasinConfiguration& configuration,
                                  const LayeredMeshLayout& layout) {
    return make_variable_layered_mesh(make_basin_geometry(configuration), layout);
}
