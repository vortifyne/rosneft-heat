#pragma once

#include "numerics/algebra/vector.hpp"

#include <cstddef>
#include <span>
#include <vector>

struct ColGrid;

// Each column stores N+1 horizontal nodes. Cell i in column c has nodes i+c and i+c+1.
[[nodiscard]] std::vector<double> horizontal_node_depths(const ColGrid& grid);
void interpolate_horizontal_temperature(const ColGrid& grid, const Vector& T,
                                        std::span<double> node_T);

// Cell-centred interpolation: linear in z in each source column, then in x.
// Temperature is linearly extrapolated beyond the outer source centres.
[[nodiscard]] Vector interpolate_temperature(const ColGrid& old_grid, const ColGrid& new_grid,
                                             const Vector& old_T);

// I[node * reaction_count + reaction] is interpolated within the same layer,
// clamping to its outer source nodes. The born layer starts with zero I.
// Every other target layer must be present in old_grid.
[[nodiscard]] std::vector<double> interpolate_reaction_integrals(const ColGrid& old_grid,
                                                                 const ColGrid& new_grid,
                                                                 std::span<const double> old_I,
                                                                 std::size_t reaction_count,
                                                                 int born_layer_id);
