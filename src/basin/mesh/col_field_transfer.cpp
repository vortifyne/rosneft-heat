#include "basin/mesh/col_field_transfer.hpp"

#include "basin/mesh/col_grid.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
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

using Weights = std::vector<Eigen::Triplet<double>>;

void add_weights(Weights& entries, std::size_t target, std::size_t l0, std::size_t l1,
                 std::size_t r0, std::size_t r1, double wl, double wr, double wx) {
    const auto add = [&](std::size_t source, double weight) {
        if (weight != 0.0)
            entries.emplace_back(target, source, weight);
    };
    add(l0, (1.0 - wx) * (1.0 - wl));
    add(l1, (1.0 - wx) * wl);
    add(r0, wx * (1.0 - wr));
    add(r1, wx * wr);
}

} // namespace

bool same_transfer_grid(const ColGrid& a, const ColGrid& b) {
    return a.x == b.x && a.columns == b.columns && a.maturity_node_count == b.maturity_node_count &&
           std::equal(a.cells.begin(), a.cells.end(), b.cells.begin(), b.cells.end(),
                      [](const ColCell& p, const ColCell& q) {
                          return p.z_top == q.z_top && p.z_bottom == q.z_bottom &&
                                 p.layer_id == q.layer_id && p.maturity_top == q.maturity_top;
                      });
}

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

ColTransferOperator temperature_transfer(const ColGrid& old_grid, const ColGrid& new_grid,
                                         int prescribed_layer_id) {
    ColTransferOperator P(new_grid.cells.size(), old_grid.cells.size());
    if (prescribed_layer_id == -1 && same_transfer_grid(old_grid, new_grid)) {
        P.setIdentity();
        return P;
    }
    Weights entries;
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
            if (new_grid.cells[cell].layer_id == prescribed_layer_id)
                continue;
            const double target_z = new_grid.cells[cell].center().z;
            const auto [l0, l1] =
                neighboring_indices(old_grid.columns[left], target_z, left_upper, true, z);
            const auto [r0, r1] =
                neighboring_indices(old_grid.columns[right], target_z, right_upper, true, z);
            const double wl = fraction(target_z, z(l0), z(l1));
            const double wr = fraction(target_z, z(r0), z(r1));
            add_weights(entries, cell, l0, l1, r0, r1, wl, wr, wx);
        }
    }
    P.setFromTriplets(entries.begin(), entries.end());
    return P;
}

ColTransferOperator maturity_transfer(const ColGrid& old_grid, const ColGrid& new_grid,
                                      int born_layer_id) {
    ColTransferOperator P(new_grid.maturity_node_count, old_grid.maturity_node_count);
    if (born_layer_id == -1 && same_transfer_grid(old_grid, new_grid)) {
        P.setIdentity();
        return P;
    }
    Weights entries;
    std::vector<double> old_z(old_grid.maturity_node_count);
    for (const auto& cell : old_grid.cells) {
        old_z[cell.maturity_top] = cell.z_top;
        old_z[cell.maturity_top + 1] = cell.z_bottom;
    }
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
            rows[column] = std::span<const std::size_t>(node_indices)
                               .subspan(old_grid.cells[*first].maturity_top,
                                        static_cast<std::size_t>(last - first) + 1);
        }
        std::size_t column_upper = 0;
        for (std::size_t column = 0; column < new_grid.columns.size(); ++column) {
            const double target_x = 0.5 * (new_grid.x[column] + new_grid.x[column + 1]);
            const auto [left, right] =
                neighboring_indices(columns, target_x, column_upper, false, x);
            const double wx = fraction(target_x, x(left), x(right));
            std::size_t left_upper = 0, right_upper = 0;
            const auto& cells = new_grid.columns[column];
            const auto in_layer = [&](std::size_t cell) {
                return new_grid.cells[cell].layer_id == layer_id;
            };
            const auto first = std::find_if(cells.begin(), cells.end(), in_layer);
            if (first == cells.end())
                continue;
            const auto last = std::find_if_not(first, cells.end(), in_layer);
            const auto start_node = new_grid.cells[*first].maturity_top;
            const auto end_node = new_grid.cells[*(last - 1)].maturity_top + 1;
            for (auto node = start_node; node <= end_node; ++node) {
                const double target_z = node == end_node
                                            ? new_grid.cells[*(last - 1)].z_bottom
                                            : new_grid.cells[first[node - start_node]].z_top;
                const auto neighbors = [&](std::span<const std::size_t> row, std::size_t& upper) {
                    if (node == start_node)
                        return std::pair{row.front(), row.front()};
                    if (node == end_node)
                        return std::pair{row.back(), row.back()};
                    return neighboring_indices(row, target_z, upper, false, z);
                };
                const auto [l0, l1] = neighbors(rows[left], left_upper);
                const auto [r0, r1] = neighbors(rows[right], right_upper);
                const double wl = fraction(target_z, z(l0), z(l1));
                const double wr = fraction(target_z, z(r0), z(r1));
                add_weights(entries, node, l0, l1, r0, r1, wl, wr, wx);
            }
        }
    }
    P.setFromTriplets(entries.begin(), entries.end());
    return P;
}

namespace {
// A column of Z has the same node-major layout as I. Each reaction is one field.
void apply_reactions(const ColTransferOperator& P, const double* source, double* target,
                     std::size_t reactions) {
    using Fields = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<Fields> out(target, P.rows(), reactions);
    out.noalias() = P * Eigen::Map<const Fields>(source, P.cols(), reactions);
}
} // namespace

std::vector<double> apply_maturity_transfer(const ColTransferOperator& P, std::span<const double> I,
                                            std::size_t reactions) {
    if (I.size() != static_cast<std::size_t>(P.cols()) * reactions)
        throw std::invalid_argument("Maturity field does not match source grid");
    std::vector<double> result(static_cast<std::size_t>(P.rows()) * reactions);
    apply_reactions(P, I.data(), result.data(), reactions);
    return result;
}

Eigen::MatrixXd apply_maturity_transfer(const ColTransferOperator& P, const Eigen::MatrixXd& Z,
                                        std::size_t reactions) {
    if (static_cast<std::size_t>(Z.rows()) != static_cast<std::size_t>(P.cols()) * reactions)
        throw std::invalid_argument("Maturity sensitivities do not match source grid");
    Eigen::MatrixXd result(P.rows() * static_cast<Eigen::Index>(reactions), Z.cols());
    for (Eigen::Index j = 0; j < Z.cols(); ++j)
        apply_reactions(P, Z.col(j).data(), result.col(j).data(), reactions);
    return result;
}

Vector interpolate_temperature(const ColGrid& old_grid, const ColGrid& new_grid,
                               const Vector& old_T) {
    if (static_cast<std::size_t>(old_T.size()) != old_grid.cells.size())
        throw std::invalid_argument("Temperature field does not match source grid");
    return Vector(temperature_transfer(old_grid, new_grid) * old_T.native());
}

std::vector<double> interpolate_reaction_integrals(const ColGrid& old_grid, const ColGrid& new_grid,
                                                   std::span<const double> old_I,
                                                   std::size_t reaction_count, int born_layer_id) {
    return apply_maturity_transfer(maturity_transfer(old_grid, new_grid, born_layer_id), old_I,
                                   reaction_count);
}
