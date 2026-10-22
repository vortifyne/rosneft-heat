#pragma once

#include "basin/basin_geometry.hpp"
#include "basin/column_layout.hpp"
#include "mesh/point2d.hpp"

#include <cstddef>
#include <optional>
#include <vector>

struct NonmatchingColumnCell {
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

struct NonmatchingHorizontalFace {
    std::size_t upper_cell = 0;
    std::size_t lower_cell = 0;
    double z = 0.0;
    double length = 0.0;
};

struct NonmatchingVerticalSubface {
    std::size_t left_cell = 0;
    std::size_t right_cell = 0;
    double x = 0.0;
    double z_top = 0.0;
    double z_bottom = 0.0;

    [[nodiscard]] double length() const noexcept;
    [[nodiscard]] double center_z() const noexcept;
};

struct NonmatchingVerticalBoundaryFace {
    std::size_t cell = 0;
    double x = 0.0;
    double z_top = 0.0;
    double z_bottom = 0.0;
    double outward_x = 0.0;

    [[nodiscard]] double length() const noexcept;
    [[nodiscard]] double center_z() const noexcept;
};

struct NonmatchingColumnGrid {
    std::vector<double> x;
    std::vector<NonmatchingColumnCell> cells;
    std::vector<std::vector<std::size_t>> columns;
    std::vector<NonmatchingHorizontalFace> horizontal_faces;
    std::vector<NonmatchingVerticalSubface> vertical_subfaces;
    std::vector<NonmatchingVerticalBoundaryFace> surface_vertical_faces;
    std::vector<NonmatchingVerticalBoundaryFace> basal_vertical_faces;
    std::vector<std::size_t> top_cells;
    std::vector<std::size_t> bottom_cells;

    [[nodiscard]] double area() const noexcept;
};

// Rebuild contacts after the depths of material cells change, keeping indices.
void rebuild_nonmatching_column_faces(NonmatchingColumnGrid& grid);

// Each geological boundary is replaced in a column by its exact mean depth on
// that x interval. Therefore every layer area is preserved before the independent
// vertical subdivision. Merging neighbouring levels creates subfaces only.
[[nodiscard]] NonmatchingColumnGrid make_nonmatching_column_grid(const BasinGeometry& geometry,
                                                                 const LayeredMeshLayout& layout);

// Retained contour coordinates and material-support endpoints are mandatory.
// Long intervals are filled quasi-uniformly; phase only moves inserted points.
[[nodiscard]] LayeredMeshLayout make_nonmatching_column_layout(const BasinGeometry& geometry,
                                                               double cell_size);

[[nodiscard]] LayeredMeshLayout make_nonmatching_column_layout(const BasinGeometry& geometry,
                                                               double horizontal_cell_size,
                                                               double vertical_cell_size,
                                                               double phase);
