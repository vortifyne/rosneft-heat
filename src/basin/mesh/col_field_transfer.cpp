#include "basin/mesh/col_field_transfer.hpp"

#include "basin/mesh/col_grid.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace {

struct ColSample {
    double x = 0.0;
    ColFieldStencil vertical;
};

ColFieldStencil vertical_stencil(const ColGrid& grid, std::size_t column, double z, int layer_id,
                                 bool same_layer) {
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

ColFieldStencil spatial_stencil(const ColGrid& old_grid, const ColCell& target, bool same_layer) {
    std::vector<ColSample> available;
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
                         [](double value, const ColSample& sample) { return value < sample.x; });
    if (available.size() == 1)
        return available.front().vertical;
    if (same_layer && upper == available.begin())
        return upper->vertical;
    if (same_layer && upper == available.end())
        return available.back().vertical;
    const ColSample& right = upper == available.begin() ? available[1]
                             : upper == available.end() ? available.back()
                                                        : *upper;
    const ColSample& left = upper == available.begin() ? available[0]
                            : upper == available.end() ? available[available.size() - 2]
                                                       : *(upper - 1);
    const double ratio = (point.x - left.x) / (right.x - left.x);
    ColFieldStencil result;
    for (const auto& donor : left.vertical) {
        result.push_back({donor.cell, (1.0 - ratio) * donor.w});
    }
    for (const auto& donor : right.vertical) {
        result.push_back({donor.cell, ratio * donor.w});
    }
    return result;
}

std::vector<double> apply_stencils(std::span<const ColFieldStencil> stencils,
                                   std::span<const double> old_values) {
    std::vector<double> result(stencils.size(), 0.0);
    for (std::size_t cell = 0; cell < stencils.size(); ++cell) {
        for (const ColFieldWeight donor : stencils[cell]) {
            if (donor.cell >= old_values.size()) {
                throw std::invalid_argument("Column transfer source size does not match");
            }
            result[cell] += donor.w * old_values[donor.cell];
        }
    }
    return result;
}

} // namespace

ColFieldTransfer make_col_field_transfer(const ColGrid& old_grid, const ColGrid& new_grid) {
    ColFieldTransfer result;
    result.T_stencils.reserve(new_grid.cells.size());
    result.layer_stencils.reserve(new_grid.cells.size());
    for (const ColCell& cell : new_grid.cells) {
        auto T_stencils = spatial_stencil(old_grid, cell, false);
        if (T_stencils.empty()) {
            throw std::runtime_error("Cannot interpolate temperature outside old grid");
        }
        result.T_stencils.push_back(std::move(T_stencils));
        result.layer_stencils.push_back(spatial_stencil(old_grid, cell, true));
    }
    return result;
}

std::vector<double>
ColFieldTransfer::interpolate_temperature(std::span<const double> old_values) const {
    return apply_stencils(T_stencils, old_values);
}

std::vector<double>
ColFieldTransfer::interpolate_same_layer(std::span<const double> old_values) const {
    return apply_stencils(layer_stencils, old_values);
}
