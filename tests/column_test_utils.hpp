#pragma once

#include "basin/nonmatching_column_grid.hpp"

#include <cstddef>
#include <optional>

inline NonmatchingColumnGrid make_test_column_grid(std::size_t nx, std::size_t nz, double width,
                                                   double depth) {
    NonmatchingColumnGrid grid;
    grid.columns.resize(nx);
    for (std::size_t column = 0; column <= nx; ++column) {
        grid.x.push_back(width * static_cast<double>(column) / static_cast<double>(nx));
    }
    for (std::size_t column = 0; column < nx; ++column) {
        const std::size_t rows = nz + column % 2;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t index = grid.cells.size();
            grid.cells.push_back(
                {.index = index,
                 .column = column,
                 .layer_id = 1,
                 .x_left = grid.x[column],
                 .x_right = grid.x[column + 1],
                 .z_top = depth * static_cast<double>(row) / static_cast<double>(rows),
                 .z_bottom = depth * static_cast<double>(row + 1) / static_cast<double>(rows),
                 .upward = std::nullopt,
                 .downward = std::nullopt});
            grid.columns[column].push_back(index);
        }
    }
    rebuild_nonmatching_column_faces(grid);
    return grid;
}
