#include "basin/mesh/column_epoch_grid.hpp"

#include "basin/geometry/basin_geometry_rdp.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

ColumnEpochGrid make_column_epoch_grid(const BasinGeometry& first, const BasinGeometry& second,
                                       double horizontal_step, double vertical_step,
                                       double minimum_cell_height,
                                       std::optional<double> axis_rdp_tolerance,
                                       std::span<const double> required_x) {
    if (!std::isfinite(vertical_step) || !(vertical_step > 0.0) ||
        !std::isfinite(minimum_cell_height) || !(minimum_cell_height > 0.0)) {
        throw std::invalid_argument(
            "Epoch grid requires positive vertical step and minimum height");
    }
    const std::size_t old_count = first.layers.size();
    if (old_count == 0 || second.layers.size() < old_count || second.layers.size() > old_count + 1)
        throw std::invalid_argument("Epoch must retain old layers and birth at most one layer");
    const auto close = [](double a, double b) {
        return std::abs(a - b) <= 1.0e-9 * std::max({1.0, std::abs(a), std::abs(b)});
    };
    if (first.x.empty() || second.x.empty() || !close(first.x.front(), second.x.front()) ||
        !close(first.x.back(), second.x.back()))
        throw std::invalid_argument("Epoch geometries must have the same horizontal domain");
    const bool birth = second.layers.size() == old_count + 1;
    const std::size_t offset = birth ? 1 : 0;
    ColumnEpochGrid result;
    result.minimum_cell_height = minimum_cell_height;
    if (birth)
        result.born_layer_id = second.layers.front().id;
    for (std::size_t layer = 0; layer < old_count; ++layer) {
        if (first.layers[layer].id != second.layers[layer + offset].id)
            throw std::invalid_argument("Old layer order or identity changed across the epoch");
        result.layer_ids.push_back(first.layers[layer].id);
    }
    auto knots =
        axis_rdp_tolerance ? simplify_basin_geometry_rdp(first, *axis_rdp_tolerance).x : first.x;
    const auto end_knots =
        axis_rdp_tolerance ? simplify_basin_geometry_rdp(second, *axis_rdp_tolerance).x : second.x;
    knots.insert(knots.end(), end_knots.begin(), end_knots.end());
    result.x = make_column_axis(knots, horizontal_step, required_x);
    for (std::size_t column = 0; column + 1 < result.x.size(); ++column) {
        const double left = result.x[column], right = result.x[column + 1];
        auto start = first.mean_boundaries(left, right);
        const auto after = second.mean_boundaries(left, right);
        std::vector<double> end(old_count + 1);
        if (!birth) {
            end = after;
        } else {
            const double middle = 0.5 * (left + right);
            std::size_t owner = old_count;
            for (std::size_t layer = 0; layer < old_count; ++layer) {
                const auto& region = second.layers[layer + 1];
                if (middle >= region.x_min && middle <= region.x_max &&
                    after[layer + 2] - after[layer + 1] > 1.0e-12) {
                    owner = layer;
                    break;
                }
            }
            if (owner == old_count)
                throw std::invalid_argument("New layer has no old material below it");
            end.front() = after.front();
            for (std::size_t layer = 0; layer < old_count; ++layer)
                end[layer + 1] = layer < owner ? after.front() : after[layer + 2];
        }
        std::vector<std::size_t> rows;
        for (std::size_t layer = 0; layer < old_count; ++layer) {
            const double thickness =
                std::max(start[layer + 1] - start[layer], end[layer + 1] - end[layer]);
            rows.push_back(thickness > 1.0e-12
                               ? std::max<std::size_t>(1, static_cast<std::size_t>(
                                                              std::ceil(thickness / vertical_step)))
                               : 0);
        }
        result.start_boundaries.push_back(std::move(start));
        result.end_boundaries.push_back(std::move(end));
        result.rows.push_back(std::move(rows));
    }
    return result;
}

NonmatchingColumnGrid ColumnEpochGrid::at(double fraction) const {
    if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0)
        throw std::invalid_argument("Invalid epoch fraction");
    NonmatchingColumnGrid result;
    result.x = x;
    result.columns.resize(rows.size());
    for (std::size_t column = 0; column < rows.size(); ++column) {
        std::vector<double> boundaries(layer_ids.size() + 1);
        for (std::size_t i = 0; i < boundaries.size(); ++i)
            boundaries[i] =
                std::lerp(start_boundaries[column][i], end_boundaries[column][i], fraction);
        basin_detail::append_column_cells(result, column, layer_ids, boundaries, rows[column],
                                          minimum_cell_height);
    }
    rebuild_nonmatching_column_faces(result);
    return result;
}
