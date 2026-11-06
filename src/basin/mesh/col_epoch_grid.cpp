#include "basin/mesh/col_epoch_grid.hpp"

#include "basin/geometry/basin_geometry_rdp.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

ColEpochGrid make_col_epoch_grid(const BasinGeometry& first, const BasinGeometry& second, double dx,
                                 double dz, double dz_min, std::optional<double> axis_rdp_tol,
                                 std::span<const double> required_x) {
    if (!std::isfinite(dz) || !(dz > 0.0) || !std::isfinite(dz_min) || !(dz_min > 0.0)) {
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
    ColEpochGrid result;
    result.dz_min = dz_min;
    if (birth) {
        result.born_layer_id = second.layers.front().id;
        result.layer_ids.push_back(*result.born_layer_id);
    }
    for (std::size_t layer = 0; layer < old_count; ++layer) {
        if (first.layers[layer].id != second.layers[layer + offset].id)
            throw std::invalid_argument("Old layer order or identity changed across the epoch");
        result.layer_ids.push_back(first.layers[layer].id);
    }
    auto knots = axis_rdp_tol ? simplify_basin_geometry_rdp(first, *axis_rdp_tol).x : first.x;
    const auto end_knots =
        axis_rdp_tol ? simplify_basin_geometry_rdp(second, *axis_rdp_tol).x : second.x;
    knots.insert(knots.end(), end_knots.begin(), end_knots.end());
    result.x = make_column_axis(knots, dx, required_x);
    for (std::size_t column = 0; column + 1 < result.x.size(); ++column) {
        const double left = result.x[column], right = result.x[column + 1];
        auto start = first.mean_boundaries(left, right);
        auto end = second.mean_boundaries(left, right);
        if (birth)
            start.insert(start.begin(), start.front());
        std::vector<std::size_t> rows;
        for (std::size_t layer = 0; layer < result.layer_ids.size(); ++layer) {
            const double thickness =
                std::max(start[layer + 1] - start[layer], end[layer + 1] - end[layer]);
            rows.push_back(
                thickness > 1.0e-12
                    ? std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(thickness / dz)))
                    : 0);
        }
        result.z0.push_back(std::move(start));
        result.z1.push_back(std::move(end));
        result.rows.push_back(std::move(rows));
    }
    return result;
}

ColGrid ColEpochGrid::at(double theta) const {
    if (!std::isfinite(theta) || theta < 0.0 || theta > 1.0)
        throw std::invalid_argument("Invalid epoch fraction");
    ColGrid result;
    result.x = x;
    result.columns.resize(rows.size());
    for (std::size_t column = 0; column < rows.size(); ++column) {
        std::vector<double> boundaries(layer_ids.size() + 1);
        for (std::size_t i = 0; i < boundaries.size(); ++i)
            boundaries[i] = std::lerp(z0[column][i], z1[column][i], theta);
        if (born_layer_id && rows[column].front() > 0) {
            // Unfold the new layer from bottom to top. A node reaches the surface
            // at theta = 1 - a, where a is its final relative depth in the layer.
            const auto count = rows[column].front();
            const double height = z1[column][1] - z1[column][0];
            std::vector<double> depths{boundaries.front()};
            std::vector<int> ids(count, *born_layer_id);
            std::vector<std::size_t> counts(count, 1);
            for (std::size_t row = 1; row <= count; ++row) {
                const double a = static_cast<double>(row) / static_cast<double>(count);
                depths.push_back(boundaries.front() + height * std::max(0.0, theta + a - 1.0));
            }
            ids.insert(ids.end(), layer_ids.begin() + 1, layer_ids.end());
            counts.insert(counts.end(), rows[column].begin() + 1, rows[column].end());
            depths.insert(depths.end(), boundaries.begin() + 2, boundaries.end());
            basin_detail::append_column_cells(result, column, ids, depths, counts, dz_min);
        } else {
            basin_detail::append_column_cells(result, column, layer_ids, boundaries, rows[column],
                                              dz_min);
        }
    }
    rebuild_col_faces(result);
    return result;
}
