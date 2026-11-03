#pragma once

#include "numerics/algebra/vector.hpp"

#include <cstddef>
#include <span>
#include <vector>

struct ColGrid;

// Cell-centred interpolation: linear in z in each source column, then in x.
// Temperature is linearly extrapolated beyond the outer source centres.
[[nodiscard]] Vector interpolate_temperature(const ColGrid& old_grid, const ColGrid& new_grid,
                                             const Vector& old_T);

// I[cell * reaction_count + reaction] is interpolated within the same layer,
// clamping to its outer source centres. The born layer starts with zero I.
// Every other target layer must be present in old_grid.
[[nodiscard]] std::vector<double> interpolate_reaction_integrals(const ColGrid& old_grid,
                                                                 const ColGrid& new_grid,
                                                                 std::span<const double> old_I,
                                                                 std::size_t reaction_count,
                                                                 int born_layer_id);
