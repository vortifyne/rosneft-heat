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
    if (birth)
        result.born_layer_id = second.layers.front().id;
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
        basin_detail::append_column_cells(result, column, layer_ids, boundaries, rows[column],
                                          dz_min);
    }
    rebuild_col_faces(result);
    return result;
}
