#pragma once

#include "basin/mesh/col_grid.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

struct ColEpochGrid {
    std::vector<double> x;
    std::vector<int> layer_ids;
    std::vector<std::vector<double>> z0; // Boundary depths [column][contact], m.
    std::vector<std::vector<double>> z1;
    std::vector<std::vector<std::size_t>> rows; // [column][old layer]
    double dz_min = 0.0;
    std::optional<int> born_layer_id;

    // Cell indices and layer IDs remain fixed for the whole epoch.
    [[nodiscard]] ColGrid at(double theta) const;
};

[[nodiscard]] ColEpochGrid make_col_epoch_grid(const BasinGeometry& first,
                                               const BasinGeometry& second, double dx, double dz,
                                               double dz_min,
                                               std::optional<double> axis_rdp_tol = std::nullopt,
                                               std::span<const double> required_x = {});
