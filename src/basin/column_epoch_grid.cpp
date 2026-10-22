#include "basin/column_epoch_grid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

ColumnEpochGrid make_column_epoch_grid(ColumnEpochGeometry geometry,
                                       double horizontal_step, double vertical_step,
                                       double minimum_cell_height) {
    if (!std::isfinite(vertical_step) || !(vertical_step > 0.0) ||
        !std::isfinite(minimum_cell_height) || minimum_cell_height < 0.0) {
        throw std::invalid_argument("Epoch grid requires a positive vertical step and nonnegative minimum");
    }
    ColumnEpochGrid result{.geometry = std::move(geometry),
                           .x = {}, .rows = {},
                           .minimum_cell_height = minimum_cell_height};
    result.x = make_column_epoch_axis(result.geometry, horizontal_step);
    result.rows.resize(result.x.size() - 1);
    for (std::size_t column = 0; column < result.rows.size(); ++column) {
        const auto start = sample_column_epoch(result.geometry, result.x[column],
                                                result.x[column + 1], 0.0);
        const auto end = sample_column_epoch(result.geometry, result.x[column],
                                              result.x[column + 1], 1.0);
        for (std::size_t layer = 0; layer < start.layer_ids.size(); ++layer) {
            const double thickness = std::max(start.boundaries[layer + 1] - start.boundaries[layer],
                                              end.boundaries[layer + 1] - end.boundaries[layer]);
            result.rows[column].push_back(thickness > 1.0e-12 ?
                std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(thickness / vertical_step))) : 0);
        }
    }
    return result;
}

NonmatchingColumnGrid ColumnEpochGrid::at(double fraction) const {
    NonmatchingColumnGrid result;
    result.x = x;
    result.columns.resize(rows.size());
    for (std::size_t column = 0; column < rows.size(); ++column) {
        const auto section = sample_column_epoch(geometry, x[column], x[column + 1], fraction);
        std::vector<double> thickness(rows[column].size(), 0.0);
        double required = 0.0;
        double available = 0.0;
        for (std::size_t layer = 0; layer < thickness.size(); ++layer) {
            thickness[layer] = std::max(0.0, section.boundaries[layer + 1] -
                                             section.boundaries[layer]);
            if (rows[column][layer] == 0) {
                if (thickness[layer] > 1.0e-9) {
                    throw std::runtime_error("Epoch cell topology changed");
                }
                continue;
            }
            const double floor = static_cast<double>(rows[column][layer]) * minimum_cell_height;
            required += std::max(0.0, floor - thickness[layer]);
            available += std::max(0.0, thickness[layer] - floor);
        }
        if (required > available + 1.0e-9) {
            throw std::runtime_error("Minimum cell height exceeds column height");
        }
        if (required > 0.0) {
            for (std::size_t layer = 0; layer < thickness.size(); ++layer) {
                if (rows[column][layer] == 0) continue;
                const double floor = static_cast<double>(rows[column][layer]) * minimum_cell_height;
                if (thickness[layer] < floor) {
                    thickness[layer] = floor;
                } else if (available > 0.0) {
                    thickness[layer] -= required * (thickness[layer] - floor) / available;
                }
            }
        }
        double cursor = section.boundaries.front();
        for (std::size_t layer = 0; layer < thickness.size(); ++layer) {
            const std::size_t count = rows[column][layer];
            for (std::size_t row = 0; row < count; ++row) {
                const double top = cursor + thickness[layer] * static_cast<double>(row) /
                                              static_cast<double>(count);
                const double bottom = cursor + thickness[layer] * static_cast<double>(row + 1) /
                                                 static_cast<double>(count);
                const std::size_t index = result.cells.size();
                result.cells.push_back({.index = index, .column = column,
                                        .layer_id = section.layer_ids[layer],
                                        .x_left = x[column], .x_right = x[column + 1],
                                        .z_top = top, .z_bottom = bottom,
                                        .upward = std::nullopt, .downward = std::nullopt});
                result.columns[column].push_back(index);
            }
            cursor += thickness[layer];
        }
        if (std::abs(cursor - section.boundaries.back()) > 1.0e-8) {
            throw std::runtime_error("Epoch column does not preserve outer boundary");
        }
    }
    if (minimum_cell_height == 0.0 &&
        std::any_of(result.cells.begin(), result.cells.end(),
                    [](const NonmatchingColumnCell& cell) { return cell.height() <= 1.0e-10; })) {
        // This full grid is only a carrier for cell-indexed state at t=0 or
        // before active-row elimination. No flux is assembled on zero cells.
        result.top_cells.reserve(result.columns.size());
        result.bottom_cells.reserve(result.columns.size());
        for (const auto& column : result.columns) {
            if (column.empty()) throw std::runtime_error("Epoch column has no material");
            result.top_cells.push_back(column.front());
            result.bottom_cells.push_back(column.back());
        }
    } else {
        rebuild_nonmatching_column_faces(result);
    }
    return result;
}

ColumnEpochGrid::Active ColumnEpochGrid::at_active(double fraction) const {
    NonmatchingColumnGrid full = at(fraction);
    Active result{.grid = {}, .full_indices = {}};
    result.grid.x = full.x;
    result.grid.columns.resize(full.columns.size());
    for (const NonmatchingColumnCell& cell : full.cells) {
        if (cell.height() <= 1.0e-10) continue;
        const std::size_t index = result.grid.cells.size();
        result.full_indices.push_back(cell.index);
        NonmatchingColumnCell active = cell;
        active.index = index;
        active.upward.reset();
        active.downward.reset();
        result.grid.cells.push_back(active);
        result.grid.columns[cell.column].push_back(index);
    }
    rebuild_nonmatching_column_faces(result.grid);
    return result;
}
