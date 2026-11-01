#pragma once

#include <cstddef>
#include <span>
#include <vector>

struct ColGrid;

struct ColFieldWeight {
    std::size_t cell = 0;
    double w = 0.0;
};

using ColFieldStencil = std::vector<ColFieldWeight>;

struct ColFieldTransfer {
    std::vector<ColFieldStencil> T_stencils;
    std::vector<ColFieldStencil> layer_stencils;

    [[nodiscard]] std::vector<double>
    interpolate_temperature(std::span<const double> old_values) const;
    [[nodiscard]] std::vector<double>
    interpolate_same_layer(std::span<const double> old_values) const;
};

// Cell-centred bilinear interpolation in x and z. For material histories,
// donor cells are restricted to the same old layer ID. Born layers have an
// empty material stencil and must be initialized by the caller.
[[nodiscard]] ColFieldTransfer make_col_field_transfer(const ColGrid& old_grid,
                                                       const ColGrid& new_grid);
