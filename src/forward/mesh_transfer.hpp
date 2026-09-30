#pragma once

#include "mesh/mesh2d.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <vector>

struct CellTransferMap {
    std::vector<Point2D> target_points;
    std::vector<std::optional<std::size_t>> donor_cells;
};

[[nodiscard]] CellTransferMap make_cell_transfer_map(const Mesh2D& source, const Mesh2D& target);

[[nodiscard]] std::optional<double>
sample_cell_field(const Mesh2D& mesh, std::span<const double> values, Point2D point);

[[nodiscard]] std::vector<double>
transfer_cell_field(const Mesh2D& source, std::span<const double> source_values,
                    const CellTransferMap& transfer_map,
                    const std::function<double(std::size_t, Point2D)>& initialize_missing);
