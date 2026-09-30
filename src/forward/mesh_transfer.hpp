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

using CellFieldSampler = std::function<std::optional<double>(Point2D)>;
using CellIndexLocator = std::function<std::optional<std::size_t>(Point2D)>;

[[nodiscard]] CellTransferMap make_cell_transfer_map(const Mesh2D& source, const Mesh2D& target);

[[nodiscard]] CellIndexLocator make_cell_index_locator(const Mesh2D& mesh);

[[nodiscard]] std::optional<double>
sample_cell_field(const Mesh2D& mesh, std::span<const double> values, Point2D point);

[[nodiscard]] CellFieldSampler make_cell_field_sampler(const Mesh2D& mesh,
                                                       std::span<const double> values);

[[nodiscard]] std::vector<double>
transfer_cell_field(const Mesh2D& source, std::span<const double> source_values,
                    const CellTransferMap& transfer_map,
                    const std::function<double(std::size_t, Point2D)>& initialize_missing);
