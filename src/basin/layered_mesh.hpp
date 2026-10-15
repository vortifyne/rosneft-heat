#pragma once

#include "basin/basin_input.hpp"
#include "mesh/mesh2d.hpp"

#include <span>
#include <vector>

struct LayeredMeshLayout {
    std::vector<double> x;
    double cell_size = 0.0;
};

[[nodiscard]] LayeredMeshLayout
make_layered_mesh_layout(std::span<const BasinConfiguration> configurations, double cell_size);

[[nodiscard]] Mesh2D make_layered_mesh(const BasinConfiguration& configuration,
                                       const LayeredMeshLayout& layout);

// Check every layer independently against its piecewise-linear profile, including
// holes and overlaps whose areas could cancel. The tolerance is a distance in metres.
void validate_layered_mesh_coverage(const Mesh2D& mesh, const BasinConfiguration& configuration,
                                    double tolerance = 1.0e-7);
