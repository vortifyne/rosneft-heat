#pragma once

#include "basin/mesh/nonmatching_column_grid.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

struct ColumnEpochGrid {
    std::vector<double> x;
    std::vector<int> layer_ids;
    std::vector<std::vector<double>> start_boundaries;
    std::vector<std::vector<double>> end_boundaries;
    std::vector<std::vector<std::size_t>> rows; // [column][old layer]
    double minimum_cell_height = 0.0;
    std::optional<int> born_layer_id;

    // Cell indices and layer IDs remain fixed for the whole epoch.
    [[nodiscard]] NonmatchingColumnGrid at(double fraction) const;
};

[[nodiscard]] ColumnEpochGrid
make_column_epoch_grid(const BasinGeometry& first, const BasinGeometry& second,
                       double horizontal_step, double vertical_step, double minimum_cell_height,
                       std::optional<double> axis_rdp_tolerance = std::nullopt,
                       std::span<const double> required_x = {});
