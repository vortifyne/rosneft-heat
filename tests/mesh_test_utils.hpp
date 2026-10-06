#pragma once

#include "mesh/mesh2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

inline Mesh2D make_rectangular_test_mesh(const double width, const double depth,
                                         const double cell_size, const int region_id = 1,
                                         const double surface_depth = 0.0,
                                         const bool skewed = false) {
    const std::size_t nx =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(width / cell_size)));
    const std::size_t nz =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(depth / cell_size)));
    const auto point = [=](const std::size_t ix, const std::size_t iz) {
        double x = width * static_cast<double>(ix) / static_cast<double>(nx);
        const double z = surface_depth + depth * static_cast<double>(iz) / static_cast<double>(nz);
        if (skewed && ix > 0 && ix < nx && iz > 0 && iz < nz) {
            constexpr double pi = 3.14159265358979323846;
            x += 0.2 * cell_size * std::sin(pi * x / width) *
                 std::sin(pi * (z - surface_depth) / depth);
        }
        return Point2D{x, z};
    };

    std::vector<Mesh2D::Cell> cells;
    cells.reserve(nx * nz);
    for (std::size_t iz = 0; iz < nz; ++iz) {
        for (std::size_t ix = 0; ix < nx; ++ix) {
            cells.push_back({.vertices = {point(ix, iz), point(ix + 1, iz), point(ix + 1, iz + 1),
                                          point(ix, iz + 1)},
                             .edge_kinds =
                                 {
                                     iz == 0 ? BoundaryKind::top : BoundaryKind::interface,
                                     ix + 1 == nx ? BoundaryKind::right : BoundaryKind::interface,
                                     iz + 1 == nz ? BoundaryKind::bottom : BoundaryKind::interface,
                                     ix == 0 ? BoundaryKind::left : BoundaryKind::interface,
                                 },
                             .id = region_id});
        }
    }
    return Mesh2D::from_cells(cells);
}
