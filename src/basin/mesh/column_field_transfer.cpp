#include "basin/mesh/column_field_transfer.hpp"

#include "basin/mesh/nonmatching_column_grid.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace {

struct ColumnSample {
    double x = 0.0;
    ColumnFieldStencil vertical;
};

ColumnFieldStencil vertical_stencil(const NonmatchingColumnGrid& grid, std::size_t column, double z,
                                    int layer_id, bool same_layer) {
    std::vector<std::size_t> cells;
    for (const std::size_t index : grid.columns[column]) {
        if (!same_layer || grid.cells[index].layer_id == layer_id) {
            cells.push_back(index);
        }
    }
    if (cells.empty())
        return {};
    if (cells.size() == 1)
        return {{cells.front(), 1.0}};
    const auto center = [&](std::size_t index) { return grid.cells[index].center().z; };
    if (same_layer && z <= center(cells.front()))
        return {{cells.front(), 1.0}};
    if (same_layer && z >= center(cells.back()))
        return {{cells.back(), 1.0}};
    std::size_t left;
    std::size_t right;
    if (z <= center(cells.front())) {
        left = cells[0];
        right = cells[1];
    } else if (z >= center(cells.back())) {
        left = cells[cells.size() - 2];
        right = cells.back();
    } else {
        const auto upper =
            std::upper_bound(cells.begin(), cells.end(), z, [&](double value, std::size_t index) {
                return value < center(index);
            });
        right = *upper;
        left = *(upper - 1);
    }
    const double ratio = (z - center(left)) / (center(right) - center(left));
    return {{left, 1.0 - ratio}, {right, ratio}};
}

ColumnFieldStencil spatial_stencil(const NonmatchingColumnGrid& old_grid,
                                   const NonmatchingColumnCell& target, bool same_layer) {
    std::vector<ColumnSample> available;
    const Point2D point = target.center();
    for (std::size_t column = 0; column < old_grid.columns.size(); ++column) {
        auto vertical = vertical_stencil(old_grid, column, point.z, target.layer_id, same_layer);
        if (!vertical.empty()) {
            available.push_back({.x = 0.5 * (old_grid.x[column] + old_grid.x[column + 1]),
                                 .vertical = std::move(vertical)});
        }
    }
    if (available.empty())
        return {};
    const auto upper =
        std::upper_bound(available.begin(), available.end(), point.x,
                         [](double value, const ColumnSample& sample) { return value < sample.x; });
    if (available.size() == 1)
        return available.front().vertical;
    if (same_layer && upper == available.begin())
        return upper->vertical;
    if (same_layer && upper == available.end())
        return available.back().vertical;
    const ColumnSample& right = upper == available.begin() ? available[1]
                                : upper == available.end() ? available.back()
                                                           : *upper;
    const ColumnSample& left = upper == available.begin() ? available[0]
                               : upper == available.end() ? available[available.size() - 2]
                                                          : *(upper - 1);
    const double ratio = (point.x - left.x) / (right.x - left.x);
    ColumnFieldStencil result;
    for (const auto& donor : left.vertical) {
        result.push_back({donor.source_cell, (1.0 - ratio) * donor.weight});
    }
    for (const auto& donor : right.vertical) {
        result.push_back({donor.source_cell, ratio * donor.weight});
    }
    return result;
}

std::vector<double> apply_stencils(std::span<const ColumnFieldStencil> stencils,
                                   std::span<const double> old_values) {
    std::vector<double> result(stencils.size(), 0.0);
    for (std::size_t cell = 0; cell < stencils.size(); ++cell) {
        for (const ColumnFieldWeight donor : stencils[cell]) {
            if (donor.source_cell >= old_values.size()) {
                throw std::invalid_argument("Column transfer source size does not match");
            }
            result[cell] += donor.weight * old_values[donor.source_cell];
        }
    }
    return result;
}

} // namespace

ColumnFieldTransfer make_column_field_transfer(const NonmatchingColumnGrid& old_grid,
                                               const NonmatchingColumnGrid& new_grid) {
    ColumnFieldTransfer result;
    result.temperature.reserve(new_grid.cells.size());
    result.same_layer.reserve(new_grid.cells.size());
    for (const NonmatchingColumnCell& cell : new_grid.cells) {
        auto temperature = spatial_stencil(old_grid, cell, false);
        if (temperature.empty()) {
            throw std::runtime_error("Cannot interpolate temperature outside old grid");
        }
        result.temperature.push_back(std::move(temperature));
        result.same_layer.push_back(spatial_stencil(old_grid, cell, true));
    }
    return result;
}

std::vector<double>
ColumnFieldTransfer::interpolate_temperature(std::span<const double> old_values) const {
    return apply_stencils(temperature, old_values);
}

std::vector<double>
ColumnFieldTransfer::interpolate_same_layer(std::span<const double> old_values) const {
    return apply_stencils(same_layer, old_values);
}
