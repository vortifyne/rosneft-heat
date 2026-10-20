#include "basin/nonmatching_column_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

double profile_mean(const ScalarProfile& profile, const double left, const double right) {
    if (!(right > left)) {
        throw std::invalid_argument("Column interval must have positive width");
    }
    std::vector<double> points{left, right};
    for (const double x : profile.x) {
        if (x > left && x < right) {
            points.push_back(x);
        }
    }
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    double integral = 0.0;
    for (std::size_t i = 1; i < points.size(); ++i) {
        const double a = points[i - 1];
        const double b = points[i];
        integral += 0.5 * (profile.at(a) + profile.at(b)) * (b - a);
    }
    return integral / (right - left);
}

} // namespace

LayeredMeshLayout make_nonmatching_column_layout(const BasinGeometry& geometry,
                                                 const double cell_size) {
    if (!std::isfinite(cell_size) || !(cell_size > 0.0) || geometry.layers.empty()) {
        throw std::invalid_argument("Nonmatching-column layout requires geometry and a step");
    }
    std::vector<double> events;
    events.reserve(2 * geometry.layers.size());
    for (const BasinLayerRegion& region : geometry.layers) {
        events.push_back(region.x_min);
        events.push_back(region.x_max);
    }
    std::sort(events.begin(), events.end());
    events.erase(std::unique(events.begin(), events.end()), events.end());
    LayeredMeshLayout result{.x = {}, .cell_size = cell_size};
    for (std::size_t interval = 1; interval < events.size(); ++interval) {
        const double left = events[interval - 1];
        const double right = events[interval];
        const std::size_t count = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil((right - left) / cell_size)));
        for (std::size_t part = 0; part < count; ++part) {
            const double fraction = static_cast<double>(part) / static_cast<double>(count);
            result.x.push_back(std::lerp(left, right, fraction));
        }
    }
    result.x.push_back(events.back());
    return result;
}

double NonmatchingColumnCell::width() const noexcept {
    return x_right - x_left;
}

double NonmatchingColumnCell::height() const noexcept {
    return z_bottom - z_top;
}

double NonmatchingColumnCell::area() const noexcept {
    return width() * height();
}

Point2D NonmatchingColumnCell::center() const noexcept {
    return {.x = 0.5 * (x_left + x_right), .z = 0.5 * (z_top + z_bottom)};
}

double NonmatchingVerticalSubface::length() const noexcept {
    return z_bottom - z_top;
}

double NonmatchingVerticalSubface::center_z() const noexcept {
    return 0.5 * (z_top + z_bottom);
}

double NonmatchingColumnGrid::area() const noexcept {
    double result = 0.0;
    for (const NonmatchingColumnCell& cell : cells) {
        result += cell.area();
    }
    return result;
}

NonmatchingColumnGrid make_nonmatching_column_grid(const BasinGeometry& geometry,
                                                   const LayeredMeshLayout& layout) {
    if (layout.x.size() < 2 || !std::isfinite(layout.cell_size) || !(layout.cell_size > 0.0)) {
        throw std::invalid_argument("Nonmatching columns require an axis and a positive step");
    }
    NonmatchingColumnGrid result;
    result.x = layout.x;
    result.columns.resize(layout.x.size() - 1);
    result.top_cells.resize(layout.x.size() - 1);
    result.bottom_cells.resize(layout.x.size() - 1);

    for (std::size_t column = 0; column + 1 < layout.x.size(); ++column) {
        const double left = layout.x[column];
        const double right = layout.x[column + 1];
        if (!std::isfinite(left) || !std::isfinite(right) || !(right > left)) {
            throw std::invalid_argument("Nonmatching-column axis must be increasing and finite");
        }
        auto& column_cells = result.columns[column];
        for (const BasinLayerRegion& region : geometry.layers) {
            if (left < region.x_min || right > region.x_max) {
                continue;
            }
            const ScalarProfile& top = geometry.boundaries[region.top_boundary].depth;
            const ScalarProfile& bottom = geometry.boundaries[region.bottom_boundary].depth;
            const double z_top = profile_mean(top, left, right);
            const double z_bottom = profile_mean(bottom, left, right);
            const double thickness = z_bottom - z_top;
            const double tolerance = 1.0e-12 * std::max({1.0, std::abs(z_top), std::abs(z_bottom)});
            if (thickness < -tolerance) {
                throw std::runtime_error("Area-preserving column has a negative layer thickness");
            }
            if (thickness <= tolerance) {
                continue;
            }
            const std::size_t rows = std::max<std::size_t>(
                1, static_cast<std::size_t>(std::ceil(thickness / layout.cell_size)));
            for (std::size_t row = 0; row < rows; ++row) {
                const double eta_top = static_cast<double>(row) / static_cast<double>(rows);
                const double eta_bottom = static_cast<double>(row + 1) / static_cast<double>(rows);
                const std::size_t index = result.cells.size();
                result.cells.push_back({.index = index,
                                        .column = column,
                                        .layer_id = region.id,
                                        .x_left = left,
                                        .x_right = right,
                                        .z_top = std::lerp(z_top, z_bottom, eta_top),
                                        .z_bottom = std::lerp(z_top, z_bottom, eta_bottom),
                                        .upward = std::nullopt,
                                        .downward = std::nullopt});
                column_cells.push_back(index);
            }
        }
        if (column_cells.empty()) {
            throw std::runtime_error("Nonmatching column contains no material");
        }
        std::sort(column_cells.begin(), column_cells.end(), [&result](const auto a, const auto b) {
            return result.cells[a].z_top < result.cells[b].z_top;
        });
        const double scale = std::max(1.0, result.cells[column_cells.back()].z_bottom -
                                               result.cells[column_cells.front()].z_top);
        const double tolerance = 1.0e-10 * scale;
        for (std::size_t row = 1; row < column_cells.size(); ++row) {
            NonmatchingColumnCell& upper = result.cells[column_cells[row - 1]];
            NonmatchingColumnCell& lower = result.cells[column_cells[row]];
            if (std::abs(upper.z_bottom - lower.z_top) > tolerance) {
                throw std::runtime_error("Nonmatching column has a vertical gap or overlap");
            }
            const double interface = 0.5 * (upper.z_bottom + lower.z_top);
            upper.z_bottom = interface;
            lower.z_top = interface;
            upper.downward = lower.index;
            lower.upward = upper.index;
            result.horizontal_faces.push_back({upper.index, lower.index, interface, right - left});
        }
        result.top_cells[column] = column_cells.front();
        result.bottom_cells[column] = column_cells.back();
    }

    for (std::size_t interface = 1; interface < result.columns.size(); ++interface) {
        const auto& left_cells = result.columns[interface - 1];
        const auto& right_cells = result.columns[interface];
        const double depth_scale = std::max({1.0, result.cells[left_cells.back()].z_bottom,
                                             result.cells[right_cells.back()].z_bottom});
        const double tolerance = 1.0e-10 * depth_scale;
        std::size_t left_row = 0;
        std::size_t right_row = 0;
        while (left_row < left_cells.size() && right_row < right_cells.size()) {
            const NonmatchingColumnCell& left_cell = result.cells[left_cells[left_row]];
            const NonmatchingColumnCell& right_cell = result.cells[right_cells[right_row]];
            const double top = std::max(left_cell.z_top, right_cell.z_top);
            const double bottom = std::min(left_cell.z_bottom, right_cell.z_bottom);
            if (bottom - top > tolerance) {
                result.vertical_subfaces.push_back(
                    {left_cell.index, right_cell.index, layout.x[interface], top, bottom});
            }
            if (left_cell.z_bottom < right_cell.z_bottom - tolerance) {
                ++left_row;
            } else if (right_cell.z_bottom < left_cell.z_bottom - tolerance) {
                ++right_row;
            } else {
                ++left_row;
                ++right_row;
            }
        }
    }
    return result;
}
