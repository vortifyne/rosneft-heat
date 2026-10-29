#pragma once

#include "basin/basin_input.hpp"

#include <vector>

// A sampled layer view derived from the authoritative boundary objects below.
struct BasinLayerGeometry {
    int id = 0;
    std::vector<double> x;
    std::vector<double> top;
    std::vector<double> bottom;

    [[nodiscard]] double top_at(double coordinate) const;
    [[nodiscard]] double bottom_at(double coordinate) const;
};

struct BasinBoundary {
    ScalarProfile depth;
};

struct BasinLayerRegion {
    int id = 0;
    std::size_t top_boundary = 0;
    std::size_t bottom_boundary = 0;
    double x_min = 0.;
    double x_max = 0.;
};

struct BasinGeometry {
    double age_ma = 0.0;
    std::vector<BasinBoundary> boundaries;
    std::vector<BasinLayerRegion> layers;

    [[nodiscard]] BasinLayerGeometry layer(int id) const;
};

// Preserve the source polylines and represent each shared contact once.
[[nodiscard]] BasinGeometry make_basin_geometry(const BasinConfiguration& configuration);
