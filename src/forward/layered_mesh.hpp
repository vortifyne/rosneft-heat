#pragma once

#include "forward/basin_input.hpp"

#include <cstddef>
#include <span>
#include <unordered_map>
#include <vector>

struct LayeredCellId {
    int layer_id = 0;
    std::size_t column = 0;
    std::size_t row = 0;

    auto operator<=>(const LayeredCellId&) const = default;
};

struct LayeredMeshLayout {
    std::vector<double> x;
    std::unordered_map<int, std::size_t> layer_rows;
};

struct LayeredMesh {
    Mesh2D mesh;
    std::vector<LayeredCellId> cell_ids;
};

[[nodiscard]] LayeredMeshLayout
make_layered_mesh_layout(std::span<const BasinConfiguration> configurations, double cell_size);

[[nodiscard]] LayeredMesh make_layered_mesh(const BasinConfiguration& configuration,
                                            const LayeredMeshLayout& layout);
