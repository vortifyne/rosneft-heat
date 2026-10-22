#pragma once

#include "basin/nonmatching_column_grid.hpp"

#include <cstddef>
#include <span>
#include <vector>

struct ColumnFieldWeight {
    std::size_t source_cell = 0;
    double weight = 0.0;
};

using ColumnFieldStencil = std::vector<ColumnFieldWeight>;

struct ColumnFieldTransfer {
    std::vector<ColumnFieldStencil> temperature;
    std::vector<ColumnFieldStencil> same_layer;

    [[nodiscard]] std::vector<double> interpolate_temperature(
        std::span<const double> old_values) const;
    [[nodiscard]] std::vector<double> interpolate_same_layer(
        std::span<const double> old_values) const;
};

// Cell-centred bilinear interpolation in x and z. For material histories,
// donor cells are restricted to the same old layer ID. Born layers have an
// empty same_layer stencil and must be initialized by the caller.
[[nodiscard]] ColumnFieldTransfer make_column_field_transfer(
    const NonmatchingColumnGrid& old_grid, const NonmatchingColumnGrid& new_grid);
