#pragma once

#include <cstddef>
#include <span>
#include <vector>

struct BasinConfiguration;

struct BasinLayerRegion {
    int id = 0;
    double x_min = 0.0;
    double x_max = 0.0;
};

struct BasinGeometry {
    double age_ma = 0.0;
    std::vector<double> x;
    // Layer i lies between contacts i and i + 1, on the shared x axis.
    std::vector<std::vector<double>> boundaries;
    std::vector<BasinLayerRegion> layers;

    [[nodiscard]] double boundary_at(std::size_t contact, double coordinate) const;
    [[nodiscard]] std::vector<double> mean_boundaries(double left, double right) const;
};

[[nodiscard]] BasinGeometry make_basin_geometry(const BasinConfiguration& configuration);

// Retain source knots and required material coordinates; subdivide long intervals.
[[nodiscard]] std::vector<double> make_column_axis(std::span<const double> knots,
                                                   double horizontal_step,
                                                   std::span<const double> required_x = {},
                                                   double phase = 0.0);
