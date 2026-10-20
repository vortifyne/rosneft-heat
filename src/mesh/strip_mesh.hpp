#pragma once

#include "mesh/mesh2d.hpp"

#include <span>
#include <vector>

struct MeshRegion {
    int id = 0;
    // A counterclockwise polygon; common interfaces may use different subdivisions.
    std::vector<Point2D> boundary;
    std::vector<BoundaryKind> edge_kinds;
};

// Fill horizontal background rows, inserting local x columns through contour
// corners. Align shared interfaces and fit cut cells without changing materials.
[[nodiscard]] Mesh2D make_strip_mesh(std::span<const MeshRegion> regions, double cell_size);
