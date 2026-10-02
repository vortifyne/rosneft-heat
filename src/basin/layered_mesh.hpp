#pragma once

#include "basin/basin_input.hpp"
#include "mesh/mesh2d.hpp"

#include <span>
#include <unordered_map>
#include <vector>

struct LayeredMeshLayout {
    std::vector<double> x;
    std::unordered_map<int, std::size_t> layer_rows;
};

[[nodiscard]] LayeredMeshLayout
make_layered_mesh_layout(std::span<const BasinConfiguration> configurations, double cell_size);

[[nodiscard]] Mesh2D make_layered_mesh(const BasinConfiguration& configuration,
                                       const LayeredMeshLayout& layout);
