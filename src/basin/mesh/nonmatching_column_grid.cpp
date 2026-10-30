#include "basin/mesh/nonmatching_column_grid.hpp"

#include "basin/geometry/basin_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

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

double NonmatchingVerticalBoundaryFace::length() const noexcept {
    return z_bottom - z_top;
}

double NonmatchingVerticalBoundaryFace::center_z() const noexcept {
    return 0.5 * (z_top + z_bottom);
}

double NonmatchingColumnGrid::area() const noexcept {
    double result = 0.0;
    for (const NonmatchingColumnCell& cell : cells) {
        result += cell.area();
    }
    return result;
}

void rebuild_nonmatching_column_faces(NonmatchingColumnGrid& result) {
    result.horizontal_faces.clear();
    result.vertical_subfaces.clear();
    result.surface_vertical_faces.clear();
    result.basal_vertical_faces.clear();
    result.top_cells.resize(result.columns.size());
    result.bottom_cells.resize(result.columns.size());
    for (NonmatchingColumnCell& cell : result.cells) {
        cell.upward.reset();
        cell.downward.reset();
    }
    for (std::size_t column = 0; column < result.columns.size(); ++column) {
        auto& column_cells = result.columns[column];
        const double right = result.x[column + 1];
        if (column_cells.empty()) {
            throw std::runtime_error("Nonmatching column contains no material");
        }
        std::sort(column_cells.begin(), column_cells.end(), [&result](const auto a, const auto b) {
            if (result.cells[a].z_top != result.cells[b].z_top) {
                return result.cells[a].z_top < result.cells[b].z_top;
            }
            return result.cells[a].z_bottom < result.cells[b].z_bottom;
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
            result.horizontal_faces.push_back(
                {upper.index, lower.index, interface, right - result.x[column]});
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
        const double left_top = result.cells[left_cells.front()].z_top;
        const double right_top = result.cells[right_cells.front()].z_top;
        const double left_bottom = result.cells[left_cells.back()].z_bottom;
        const double right_bottom = result.cells[right_cells.back()].z_bottom;
        const double common_top = std::max(left_top, right_top);
        const double common_bottom = std::min(left_bottom, right_bottom);
        if (common_bottom < common_top - tolerance) {
            throw std::runtime_error("Neighbouring nonmatching columns do not overlap");
        }

        const auto append_boundary_faces =
            [&result, tolerance](const std::vector<std::size_t>& column, const double z_top,
                                 const double z_bottom, const double x, const double outward_x,
                                 auto& output) {
                for (const std::size_t cell_index : column) {
                    const NonmatchingColumnCell& cell = result.cells[cell_index];
                    const double top = std::max(z_top, cell.z_top);
                    const double bottom = std::min(z_bottom, cell.z_bottom);
                    if (bottom - top > tolerance) {
                        output.push_back({cell_index, x, top, bottom, outward_x});
                    }
                }
            };
        const double interface_x = result.x[interface];
        if (left_top < right_top - tolerance) {
            append_boundary_faces(left_cells, left_top, right_top, interface_x, 1.0,
                                  result.surface_vertical_faces);
        } else if (right_top < left_top - tolerance) {
            append_boundary_faces(right_cells, right_top, left_top, interface_x, -1.0,
                                  result.surface_vertical_faces);
        }
        if (left_bottom > right_bottom + tolerance) {
            append_boundary_faces(left_cells, right_bottom, left_bottom, interface_x, 1.0,
                                  result.basal_vertical_faces);
        } else if (right_bottom > left_bottom + tolerance) {
            append_boundary_faces(right_cells, left_bottom, right_bottom, interface_x, -1.0,
                                  result.basal_vertical_faces);
        }

        std::size_t left_row = 0;
        std::size_t right_row = 0;
        while (left_row < left_cells.size() && right_row < right_cells.size()) {
            const NonmatchingColumnCell& left_cell = result.cells[left_cells[left_row]];
            const NonmatchingColumnCell& right_cell = result.cells[right_cells[right_row]];
            const double top = std::max(left_cell.z_top, right_cell.z_top);
            const double bottom = std::min(left_cell.z_bottom, right_cell.z_bottom);
            if (bottom - top > tolerance) {
                result.vertical_subfaces.push_back(
                    {left_cell.index, right_cell.index, interface_x, top, bottom});
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
}

void basin_detail::append_column_cells(NonmatchingColumnGrid& grid, std::size_t column,
                                       std::span<const int> layer_ids,
                                       std::span<const double> boundaries,
                                       std::span<const std::size_t> rows,
                                       double minimum_cell_height) {
    std::vector<double> thickness(rows.size(), 0.0);
    double required = 0.0, available = 0.0;
    for (std::size_t layer = 0; layer < rows.size(); ++layer) {
        if (rows[layer] == 0)
            continue;
        thickness[layer] = std::max(0.0, boundaries[layer + 1] - boundaries[layer]);
        const double floor = static_cast<double>(rows[layer]) * minimum_cell_height;
        required += std::max(0.0, floor - thickness[layer]);
        available += std::max(0.0, thickness[layer] - floor);
    }
    if (required > available + 1.0e-9)
        throw std::runtime_error("Minimum cell height exceeds column height");
    if (required > 0.0) {
        for (std::size_t layer = 0; layer < rows.size(); ++layer) {
            if (rows[layer] == 0)
                continue;
            const double floor = static_cast<double>(rows[layer]) * minimum_cell_height;
            if (thickness[layer] < floor)
                thickness[layer] = floor;
            else if (available > 0.0)
                thickness[layer] -= required * (thickness[layer] - floor) / available;
        }
    }
    double cursor = boundaries.front();
    for (std::size_t layer = 0; layer < rows.size(); ++layer) {
        const std::size_t count = rows[layer];
        for (std::size_t row = 0; row < count; ++row) {
            const double top =
                cursor + thickness[layer] * static_cast<double>(row) / static_cast<double>(count);
            const double bottom = cursor + thickness[layer] * static_cast<double>(row + 1) /
                                               static_cast<double>(count);
            const std::size_t index = grid.cells.size();
            grid.cells.push_back({.index = index,
                                  .column = column,
                                  .layer_id = layer_ids[layer],
                                  .x_left = grid.x[column],
                                  .x_right = grid.x[column + 1],
                                  .z_top = top,
                                  .z_bottom = bottom,
                                  .upward = std::nullopt,
                                  .downward = std::nullopt});
            grid.columns[column].push_back(index);
        }
        cursor += thickness[layer];
    }
}

NonmatchingColumnGrid make_nonmatching_column_grid(const BasinGeometry& geometry,
                                                   std::vector<double> x, double vertical_step) {
    if (x.size() < 2 || !std::isfinite(vertical_step) || !(vertical_step > 0.0))
        throw std::invalid_argument("Nonmatching columns require an axis and a positive step");
    NonmatchingColumnGrid result;
    result.x = std::move(x);
    result.columns.resize(result.x.size() - 1);
    std::vector<int> layer_ids;
    for (const auto& layer : geometry.layers)
        layer_ids.push_back(layer.id);
    for (std::size_t column = 0; column + 1 < result.x.size(); ++column) {
        const double left = result.x[column], right = result.x[column + 1];
        if (!std::isfinite(left) || !std::isfinite(right) || !(right > left))
            throw std::invalid_argument("Nonmatching-column axis must be increasing and finite");
        const auto boundaries = geometry.mean_boundaries(left, right);
        std::vector<std::size_t> rows(geometry.layers.size(), 0);
        for (std::size_t layer = 0; layer < rows.size(); ++layer) {
            const auto& region = geometry.layers[layer];
            if (left < region.x_min || right > region.x_max)
                continue;
            const double thickness = boundaries[layer + 1] - boundaries[layer];
            const double tolerance = 1.0e-12 * std::max({1.0, std::abs(boundaries[layer]),
                                                         std::abs(boundaries[layer + 1])});
            if (thickness < -tolerance)
                throw std::runtime_error("Area-preserving column has a negative layer thickness");
            if (thickness > tolerance)
                rows[layer] = std::max<std::size_t>(
                    1, static_cast<std::size_t>(std::ceil(thickness / vertical_step)));
        }
        basin_detail::append_column_cells(result, column, layer_ids, boundaries, rows, 0.0);
        if (result.columns[column].empty())
            throw std::runtime_error("Nonmatching column contains no material");
    }
    rebuild_nonmatching_column_faces(result);
    return result;
}
