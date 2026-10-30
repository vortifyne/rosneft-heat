#pragma once

#include "basin/geometry/basin_geometry.hpp"
#include "basin/mesh/col_grid.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

inline ColGrid make_test_column_grid(std::size_t nx, std::size_t nz, double width, double depth) {
    ColGrid grid;
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
    rebuild_col_faces(grid);
    return grid;
}

inline BasinGeometry make_test_layered_geometry() {
    const std::vector<double> x{0.0, 1.0, 2.0};
    return {.x = x,
            .z = {{0.0, 0.4, 0.0}, {0.8, 1.2, 1.0}, {2.0, 2.4, 2.2}},
            .layers = {{1, 0.0, 2.0}, {2, 0.0, 2.0}}};
}

inline double area_weighted_error(const ColGrid& grid, std::span<const double> solution,
                                  const auto& exact) {
    double squared_error = 0.0;
    for (const auto& cell : grid.cells) {
        const Point2D point = cell.center();
        const double difference = solution[cell.index] - exact(point.x, point.z);
        squared_error += cell.area() * difference * difference;
    }
    return std::sqrt(squared_error / grid.area());
}
