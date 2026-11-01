#pragma once

#include "basin/mesh/point2d.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

struct BasinGeometry;

// Coordinates and lengths [m].
struct ColCell {
    std::size_t index = 0;
    std::size_t column = 0;
    int layer_id = 0;
    double x_left = 0.0;
    double x_right = 0.0;
    double z_top = 0.0;
    double z_bottom = 0.0;
    std::optional<std::size_t> upward;
    std::optional<std::size_t> downward;

    [[nodiscard]] double width() const noexcept;
    [[nodiscard]] double height() const noexcept;
    [[nodiscard]] double area() const noexcept;
    [[nodiscard]] Point2D center() const noexcept;
};

struct HorizontalFace {
    std::size_t upper_cell = 0;
    std::size_t lower_cell = 0;
    double z = 0.0;
    double length = 0.0;
};

struct VerticalSubface {
    std::size_t left_cell = 0;
    std::size_t right_cell = 0;
    double x = 0.0;
    double z_top = 0.0;
    double z_bottom = 0.0;

    [[nodiscard]] double length() const noexcept;
};

struct VerticalBoundaryFace {
    std::size_t cell = 0;
    double x = 0.0;
    double z_top = 0.0;
    double z_bottom = 0.0;
    double outward_x = 0.0;

    [[nodiscard]] double length() const noexcept;
    [[nodiscard]] double center_z() const noexcept;
};

struct ColGrid {
    std::vector<double> x;
    std::vector<ColCell> cells;
    std::vector<std::vector<std::size_t>> columns;
    std::vector<HorizontalFace> horizontal_faces;
    std::vector<VerticalSubface> vertical_subfaces;
    std::vector<VerticalBoundaryFace> surface_vertical_faces;
    std::vector<VerticalBoundaryFace> basal_vertical_faces;
    std::vector<std::size_t> top_cells;
    std::vector<std::size_t> bottom_cells;

    [[nodiscard]] double area() const noexcept;
};

// Rebuild contacts after the depths of material cells change, keeping indices.
void rebuild_col_faces(ColGrid& grid);

// Each geological boundary is replaced in a column by its exact mean depth on
// that x interval. Therefore every layer area is preserved before the independent
// vertical subdivision. Merging neighbouring levels creates subfaces only.
[[nodiscard]] ColGrid make_col_grid(const BasinGeometry& geometry, std::vector<double> x,
                                    double dz);

namespace basin_detail {

// Shared cell construction for epoch grids and static geometry snapshots.
void append_column_cells(ColGrid& grid, std::size_t column, std::span<const int> layer_ids,
                         std::span<const double> boundaries, std::span<const std::size_t> rows,
                         double dz_min);

} // namespace basin_detail
