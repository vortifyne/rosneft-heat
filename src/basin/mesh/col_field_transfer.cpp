#include "basin/mesh/col_field_transfer.hpp"

#include "basin/mesh/col_grid.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace {

// Queries arrive in increasing coordinate order; upper advances only forwards.
std::pair<std::size_t, std::size_t> neighboring_indices(std::span<const std::size_t> indices,
                                                        double value, std::size_t& upper,
                                                        bool extrapolate, const auto& coordinate) {
    while (upper < indices.size() && coordinate(indices[upper]) <= value)
        ++upper;
    if (indices.size() == 1 || (!extrapolate && upper == 0))
        return {indices.front(), indices.front()};
    if (!extrapolate && upper == indices.size())
        return {indices.back(), indices.back()};
    const std::size_t right = std::clamp(upper, std::size_t{1}, indices.size() - 1);
    return {indices[right - 1], indices[right]};
}

double fraction(double value, double left, double right) {
    return left == right ? 0.0 : (value - left) / (right - left);
}

} // namespace

std::vector<double> horizontal_node_depths(const ColGrid& grid) {
    std::vector<double> z(grid.cells.size() + grid.columns.size());
    for (const auto& column : grid.columns) {
        for (const auto index : column) {
            const auto& cell = grid.cells[index];
            z[index + cell.column] = cell.z_top;
        }
        const auto& last = grid.cells[column.back()];
        z[last.index + last.column + 1] = last.z_bottom;
    }
    return z;
}

Vector interpolate_temperature(const ColGrid& old_grid, const ColGrid& new_grid,
                               const Vector& old_T) {
    Vector T(static_cast<Vector::Index>(new_grid.cells.size()));
    std::vector<std::size_t> columns(old_grid.columns.size());
    std::iota(columns.begin(), columns.end(), 0);
    const auto x = [&](std::size_t column) {
        return 0.5 * (old_grid.x[column] + old_grid.x[column + 1]);
    };
    const auto z = [&](std::size_t cell) { return old_grid.cells[cell].center().z; };
    std::size_t column_upper = 0;
    for (std::size_t column = 0; column < new_grid.columns.size(); ++column) {
        const double target_x = 0.5 * (new_grid.x[column] + new_grid.x[column + 1]);
        const auto [left, right] = neighboring_indices(columns, target_x, column_upper, true, x);
        const double wx = fraction(target_x, x(left), x(right));
        std::size_t left_upper = 0, right_upper = 0;
        for (const std::size_t cell : new_grid.columns[column]) {
            const double target_z = new_grid.cells[cell].center().z;
            const auto [l0, l1] =
                neighboring_indices(old_grid.columns[left], target_z, left_upper, true, z);
            const auto [r0, r1] =
                neighboring_indices(old_grid.columns[right], target_z, right_upper, true, z);
            const double wl = fraction(target_z, z(l0), z(l1));
            const double wr = fraction(target_z, z(r0), z(r1));
            const double left_T = std::lerp(old_T[static_cast<Vector::Index>(l0)],
                                            old_T[static_cast<Vector::Index>(l1)], wl);
            const double right_T = std::lerp(old_T[static_cast<Vector::Index>(r0)],
                                             old_T[static_cast<Vector::Index>(r1)], wr);
            T[static_cast<Vector::Index>(cell)] = std::lerp(left_T, right_T, wx);
        }
    }
    return T;
}

std::vector<double> interpolate_reaction_integrals(const ColGrid& old_grid, const ColGrid& new_grid,
                                                   std::span<const double> old_I,
                                                   std::size_t reaction_count, int born_layer_id) {
    std::vector<double> I((new_grid.cells.size() + new_grid.columns.size()) * reaction_count, 0.0);
    const auto old_z = horizontal_node_depths(old_grid);
    const auto new_z = horizontal_node_depths(new_grid);
    std::vector<std::size_t> node_indices(old_z.size());
    std::iota(node_indices.begin(), node_indices.end(), 0);
    std::vector<int> layer_ids;
    for (const auto& cell : old_grid.cells) {
        if (cell.layer_id != born_layer_id &&
            std::find(layer_ids.begin(), layer_ids.end(), cell.layer_id) == layer_ids.end())
            layer_ids.push_back(cell.layer_id);
    }
    const auto x = [&](std::size_t column) {
        return 0.5 * (old_grid.x[column] + old_grid.x[column + 1]);
    };
    const auto z = [&](std::size_t node) { return old_z[node]; };
    for (const int layer_id : layer_ids) {
        std::vector<std::size_t> columns;
        std::vector<std::span<const std::size_t>> rows(old_grid.columns.size());
        for (std::size_t column = 0; column < old_grid.columns.size(); ++column) {
            const auto& cells = old_grid.columns[column];
            const auto in_layer = [&](std::size_t cell) {
                return old_grid.cells[cell].layer_id == layer_id;
            };
            const auto first = std::find_if(cells.begin(), cells.end(), in_layer);
            if (first == cells.end())
                continue;
            const auto last = std::find_if_not(first, cells.end(), in_layer);
            columns.push_back(column);
            rows[column] =
                std::span<const std::size_t>(node_indices)
                    .subspan(*first + column, static_cast<std::size_t>(last - first) + 1);
        }
        std::size_t column_upper = 0;
        for (std::size_t column = 0; column < new_grid.columns.size(); ++column) {
            const double target_x = 0.5 * (new_grid.x[column] + new_grid.x[column + 1]);
            const auto [left, right] =
                neighboring_indices(columns, target_x, column_upper, false, x);
            const double wx = fraction(target_x, x(left), x(right));
            std::size_t left_upper = 0, right_upper = 0;
            const auto& cells = new_grid.columns[column];
            const auto first = std::find_if(cells.begin(), cells.end(), [&](std::size_t cell) {
                return new_grid.cells[cell].layer_id == layer_id;
            });
            if (first == cells.end())
                continue;
            const auto last = std::find_if_not(first, cells.end(), [&](std::size_t cell) {
                return new_grid.cells[cell].layer_id == layer_id;
            });
            const auto end_node = *(last - 1) + column + 1;
            for (std::size_t node = *first + column; node <= end_node; ++node) {
                const double target_z = new_z[node];
                const auto [l0, l1] =
                    neighboring_indices(rows[left], target_z, left_upper, false, z);
                const auto [r0, r1] =
                    neighboring_indices(rows[right], target_z, right_upper, false, z);
                const double wl = fraction(target_z, z(l0), z(l1));
                const double wr = fraction(target_z, z(r0), z(r1));
                for (std::size_t reaction = 0; reaction < reaction_count; ++reaction) {
                    const double left_I = std::lerp(old_I[l0 * reaction_count + reaction],
                                                    old_I[l1 * reaction_count + reaction], wl);
                    const double right_I = std::lerp(old_I[r0 * reaction_count + reaction],
                                                     old_I[r1 * reaction_count + reaction], wr);
                    I[node * reaction_count + reaction] = std::lerp(left_I, right_I, wx);
                }
            }
        }
    }
    for (const auto& cell : new_grid.cells) {
        if (cell.layer_id == born_layer_id)
            std::fill_n(I.begin() + static_cast<std::ptrdiff_t>((cell.index + cell.column) *
                                                                reaction_count),
                        2 * reaction_count, 0.0);
    }
    return I;
}
