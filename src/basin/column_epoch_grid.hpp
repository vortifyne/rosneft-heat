#pragma once

#include "basin/column_epoch_geometry.hpp"
#include "basin/nonmatching_column_grid.hpp"

#include <cstddef>
#include <vector>

struct ColumnEpochGrid {
    ColumnEpochGeometry geometry;
    std::vector<double> x;
    std::vector<std::vector<std::size_t>> rows; // [column][old layer]
    double minimum_cell_height = 0.0;

    // Cell indices and layer IDs remain fixed for the whole epoch.
    [[nodiscard]] NonmatchingColumnGrid at(double fraction) const;

    struct Active {
        NonmatchingColumnGrid grid;
        std::vector<std::size_t> full_indices; // active index -> persistent index
    };
    // Eliminate exactly collapsed rows from the current linear system.
    [[nodiscard]] Active at_active(double fraction) const;
};

[[nodiscard]] ColumnEpochGrid make_column_epoch_grid(ColumnEpochGeometry geometry,
                                                     double horizontal_step,
                                                     double vertical_step,
                                                     double minimum_cell_height);
