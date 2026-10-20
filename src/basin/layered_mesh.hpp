#pragma once

#include "basin/basin_geometry.hpp"
#include "mesh/mesh2d.hpp"

#include <span>
#include <vector>

struct LayeredMeshLayout {
    std::vector<double> x;
    double cell_size = 0.0;
};

[[nodiscard]] LayeredMeshLayout
make_layered_mesh_layout(std::span<const BasinConfiguration> configurations, double cell_size);

[[nodiscard]] LayeredMeshLayout make_layered_mesh_layout(std::span<const BasinGeometry> geometries,
                                                         double cell_size);

[[nodiscard]] Mesh2D make_layered_mesh(const BasinConfiguration& configuration,
                                       const LayeredMeshLayout& layout);

[[nodiscard]] Mesh2D make_layered_mesh(const BasinGeometry& geometry,
                                       const LayeredMeshLayout& layout);

// Layer-following columns with a locally chosen number of uniformly spaced
// vertical edges. Unequal counts are joined by triangles and quadrilaterals.
[[nodiscard]] Mesh2D make_variable_layered_mesh(const BasinGeometry& geometry,
                                               const LayeredMeshLayout& layout);

[[nodiscard]] Mesh2D make_variable_layered_mesh(const BasinConfiguration& configuration,
                                               const LayeredMeshLayout& layout);

[[nodiscard]] Mesh2D make_layer_strip_mesh(const BasinGeometry& geometry, double cell_size);

// Check every layer independently against its piecewise-linear profile, including
// holes and overlaps whose areas could cancel. The tolerance is a distance in metres.
void validate_layered_mesh_coverage(const Mesh2D& mesh, const BasinConfiguration& configuration,
                                    double tolerance = 1.0e-7);

void validate_layered_mesh_coverage(const Mesh2D& mesh, const BasinGeometry& geometry,
                                    double tolerance = 1.0e-7);
