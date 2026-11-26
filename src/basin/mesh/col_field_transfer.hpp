#pragma once

#include "numerics/algebra/vector.hpp"

#include <Eigen/SparseCore>
#include <cstddef>
#include <span>
#include <vector>

struct ColGrid;

// Each column stores N+1 horizontal nodes. Cell i in column c has nodes i+c and i+c+1.
[[nodiscard]] std::vector<double> horizontal_node_depths(const ColGrid& grid);
[[nodiscard]] bool same_transfer_grid(const ColGrid& a, const ColGrid& b);

// Rows are target locations and columns are source locations. P*T and P*S use the same weights.
using ColTransferOperator = Eigen::SparseMatrix<double, Eigen::RowMajor>;
[[nodiscard]] ColTransferOperator temperature_transfer(const ColGrid& source, const ColGrid& target,
                                                       int prescribed_layer_id = -1);
[[nodiscard]] ColTransferOperator maturity_transfer(const ColGrid& source, const ColGrid& target,
                                                    int born_layer_id);
// Both I and each column of Z store reactions consecutively within a material node.
[[nodiscard]] std::vector<double> apply_maturity_transfer(const ColTransferOperator& P,
                                                          std::span<const double> I,
                                                          std::size_t reactions);
[[nodiscard]] Eigen::MatrixXd apply_maturity_transfer(const ColTransferOperator& P,
                                                      const Eigen::MatrixXd& Z,
                                                      std::size_t reactions);

// Cell-centred interpolation: linear in z in each source column, then in x.
// Temperature is linearly extrapolated beyond the outer source centres.
[[nodiscard]] Vector interpolate_temperature(const ColGrid& old_grid, const ColGrid& new_grid,
                                             const Vector& old_T);

// I[node * reaction_count + reaction] uses layer-local maturity nodes. Endpoints
// transfer by layer side, while interiors interpolate in depth within the same
// layer, clamping to its outer source nodes. The born layer starts with zero I.
// Every other target layer must be present in old_grid.
[[nodiscard]] std::vector<double> interpolate_reaction_integrals(const ColGrid& old_grid,
                                                                 const ColGrid& new_grid,
                                                                 std::span<const double> old_I,
                                                                 std::size_t reaction_count,
                                                                 int born_layer_id);
