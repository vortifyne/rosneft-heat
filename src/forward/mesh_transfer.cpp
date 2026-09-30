#include "forward/mesh_transfer.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

bool contains(const TQMesh::Facet& cell, const Point2D point) {
    double sign = 0.0;
    double scale = 1.0;
    for (std::size_t index = 0; index < cell.n_vertices(); ++index) {
        const auto& first = cell.vertex(index).xy();
        const auto& second = cell.vertex((index + 1) % cell.n_vertices()).xy();
        const double cross = ((second.x - first.x) * (point.z - first.y)) -
                             ((second.y - first.y) * (point.x - first.x));
        scale = std::max(scale, std::abs(first.x) + std::abs(first.y));
        if (std::abs(cross) <= 1.0e-10 * scale) {
            continue;
        }
        if (sign == 0.0) {
            sign = cross;
        } else if (sign * cross < 0.0) {
            return false;
        }
    }
    return true;
}

std::optional<std::size_t> find_cell(const Mesh2D& mesh, const Point2D point) {
    for (const TQMesh::Facet* cell : mesh.cells()) {
        if (contains(*cell, point)) {
            return static_cast<std::size_t>(cell->index());
        }
    }
    return std::nullopt;
}

std::vector<std::vector<std::size_t>> neighbors(const Mesh2D& mesh) {
    std::vector<std::vector<std::size_t>> result(mesh.cells().size());
    for (const TQMesh::Edge* edge : mesh.internal_edges()) {
        const std::size_t owner = static_cast<std::size_t>(mesh.owner(*edge).index());
        const std::size_t neighbor = static_cast<std::size_t>(mesh.neighbor(*edge).index());
        result[owner].push_back(neighbor);
        result[neighbor].push_back(owner);
    }
    return result;
}

std::vector<Eigen::Vector2d>
reconstruct_gradients(const Mesh2D& mesh, const std::span<const double> values,
                      const std::vector<std::vector<std::size_t>>& adjacency) {
    std::vector<Eigen::Vector2d> result(mesh.cells().size(), Eigen::Vector2d::Zero());
    for (const TQMesh::Facet* cell : mesh.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        Eigen::Matrix2d normal = Eigen::Matrix2d::Zero();
        Eigen::Vector2d rhs = Eigen::Vector2d::Zero();
        for (const std::size_t neighbor : adjacency[index]) {
            const auto& neighbor_cell = *mesh.cells()[neighbor];
            const Eigen::Vector2d displacement(neighbor_cell.xy().x - cell->xy().x,
                                               neighbor_cell.xy().y - cell->xy().y);
            const double weight = 1.0 / std::max(displacement.squaredNorm(), 1.0e-30);
            normal += weight * displacement * displacement.transpose();
            rhs += weight * displacement * (values[neighbor] - values[index]);
        }
        if (std::abs(normal.determinant()) > 1.0e-14) {
            result[index] = normal.ldlt().solve(rhs);
        }
    }
    return result;
}

double reconstruct(const Mesh2D& mesh, const std::span<const double> values,
                   const std::vector<std::vector<std::size_t>>& adjacency,
                   const std::vector<Eigen::Vector2d>& gradients, const std::size_t donor,
                   const Point2D point) {
    const auto& cell = *mesh.cells()[donor];
    const Eigen::Vector2d offset(point.x - cell.xy().x, point.z - cell.xy().y);
    double value = values[donor] + gradients[donor].dot(offset);
    double minimum = values[donor];
    double maximum = values[donor];
    for (const std::size_t neighbor : adjacency[donor]) {
        minimum = std::min(minimum, values[neighbor]);
        maximum = std::max(maximum, values[neighbor]);
    }
    return std::clamp(value, minimum, maximum);
}

} // namespace

CellTransferMap make_cell_transfer_map(const Mesh2D& source, const Mesh2D& target) {
    CellTransferMap result;
    result.target_points.resize(target.cells().size());
    result.donor_cells.resize(target.cells().size());
    for (const TQMesh::Facet* cell : target.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        result.target_points[index] = {cell->xy().x, cell->xy().y};
        result.donor_cells[index] = find_cell(source, result.target_points[index]);
    }
    return result;
}

std::optional<double> sample_cell_field(const Mesh2D& mesh, const std::span<const double> values,
                                        const Point2D point) {
    if (values.size() != mesh.cells().size()) {
        throw std::invalid_argument("Cell field size must match the mesh");
    }
    const auto donor = find_cell(mesh, point);
    if (!donor) {
        return std::nullopt;
    }
    const auto adjacency = neighbors(mesh);
    const auto gradients = reconstruct_gradients(mesh, values, adjacency);
    return reconstruct(mesh, values, adjacency, gradients, *donor, point);
}

std::vector<double>
transfer_cell_field(const Mesh2D& source, const std::span<const double> source_values,
                    const CellTransferMap& transfer_map,
                    const std::function<double(std::size_t, Point2D)>& initialize_missing) {
    if (source_values.size() != source.cells().size() ||
        transfer_map.target_points.size() != transfer_map.donor_cells.size()) {
        throw std::invalid_argument("Invalid cell-transfer field sizes");
    }
    const auto adjacency = neighbors(source);
    const auto gradients = reconstruct_gradients(source, source_values, adjacency);
    std::vector<double> result(transfer_map.target_points.size());
    for (std::size_t index = 0; index < result.size(); ++index) {
        if (transfer_map.donor_cells[index]) {
            result[index] =
                reconstruct(source, source_values, adjacency, gradients,
                            *transfer_map.donor_cells[index], transfer_map.target_points[index]);
        } else {
            result[index] = initialize_missing(index, transfer_map.target_points[index]);
        }
    }
    return result;
}
