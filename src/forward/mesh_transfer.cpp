#include "forward/mesh_transfer.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

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

class CellLocator {
public:
    explicit CellLocator(const Mesh2D& mesh) : mesh_(mesh) {
        double minimum_x = std::numeric_limits<double>::max();
        double minimum_z = std::numeric_limits<double>::max();
        double maximum_x = std::numeric_limits<double>::lowest();
        double maximum_z = std::numeric_limits<double>::lowest();
        double maximum_diameter = 0.0;
        for (const TQMesh::Facet* cell : mesh.cells()) {
            for (std::size_t vertex = 0; vertex < cell->n_vertices(); ++vertex) {
                const auto& point = cell->vertex(vertex).xy();
                minimum_x = std::min(minimum_x, point.x);
                minimum_z = std::min(minimum_z, point.y);
                maximum_x = std::max(maximum_x, point.x);
                maximum_z = std::max(maximum_z, point.y);
            }
            maximum_diameter = std::max(maximum_diameter, cell->max_edge_length());
        }
        origin_ = {minimum_x, minimum_z};
        maximum_ = {maximum_x, maximum_z};
        bucket_size_ = std::max(maximum_diameter, 1.0e-12);
        columns_ = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil((maximum_x - minimum_x) / bucket_size_)) + 1);

        for (const TQMesh::Facet* cell : mesh.cells()) {
            double cell_minimum_x = std::numeric_limits<double>::max();
            double cell_minimum_z = std::numeric_limits<double>::max();
            double cell_maximum_x = std::numeric_limits<double>::lowest();
            double cell_maximum_z = std::numeric_limits<double>::lowest();
            for (std::size_t vertex = 0; vertex < cell->n_vertices(); ++vertex) {
                const auto& point = cell->vertex(vertex).xy();
                cell_minimum_x = std::min(cell_minimum_x, point.x);
                cell_minimum_z = std::min(cell_minimum_z, point.y);
                cell_maximum_x = std::max(cell_maximum_x, point.x);
                cell_maximum_z = std::max(cell_maximum_z, point.y);
            }
            const auto [first_column, first_row] = bucket(cell_minimum_x, cell_minimum_z);
            const auto [last_column, last_row] = bucket(cell_maximum_x, cell_maximum_z);
            for (std::size_t row = first_row; row <= last_row; ++row) {
                for (std::size_t column = first_column; column <= last_column; ++column) {
                    buckets_[key(column, row)].push_back(static_cast<std::size_t>(cell->index()));
                }
            }
        }
    }

    [[nodiscard]] std::optional<std::size_t> find(const Point2D point) const {
        if (point.x < origin_.x || point.z < origin_.z || point.x > maximum_.x ||
            point.z > maximum_.z) {
            return std::nullopt;
        }
        const auto [column, row] = bucket(point.x, point.z);
        const auto found = buckets_.find(key(column, row));
        if (found == buckets_.end()) {
            return std::nullopt;
        }
        for (const std::size_t index : found->second) {
            if (contains(*mesh_.cells()[index], point)) {
                return index;
            }
        }
        return std::nullopt;
    }

private:
    [[nodiscard]] std::pair<std::size_t, std::size_t> bucket(const double x, const double z) const {
        const double relative_x = std::max(0.0, x - origin_.x);
        const double relative_z = std::max(0.0, z - origin_.z);
        return {static_cast<std::size_t>(relative_x / bucket_size_),
                static_cast<std::size_t>(relative_z / bucket_size_)};
    }

    [[nodiscard]] std::size_t key(const std::size_t column, const std::size_t row) const {
        return row * columns_ + column;
    }

    const Mesh2D& mesh_;
    Point2D origin_;
    Point2D maximum_;
    double bucket_size_ = 1.0;
    std::size_t columns_ = 1;
    std::unordered_map<std::size_t, std::vector<std::size_t>> buckets_;
};

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
    const CellIndexLocator locator = make_cell_index_locator(source);
    result.target_points.resize(target.cells().size());
    result.donor_cells.resize(target.cells().size());
    for (const TQMesh::Facet* cell : target.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        result.target_points[index] = {cell->xy().x, cell->xy().y};
        result.donor_cells[index] = locator(result.target_points[index]);
    }
    return result;
}

CellIndexLocator make_cell_index_locator(const Mesh2D& mesh) {
    auto locator = std::make_shared<CellLocator>(mesh);
    return [locator = std::move(locator)](const Point2D point) { return locator->find(point); };
}

std::optional<double> sample_cell_field(const Mesh2D& mesh, const std::span<const double> values,
                                        const Point2D point) {
    return make_cell_field_sampler(mesh, values)(point);
}

CellFieldSampler make_cell_field_sampler(const Mesh2D& mesh, const std::span<const double> values) {
    if (values.size() != mesh.cells().size()) {
        throw std::invalid_argument("Cell field size must match the mesh");
    }
    auto locator = std::make_shared<CellLocator>(mesh);
    auto adjacency = std::make_shared<std::vector<std::vector<std::size_t>>>(neighbors(mesh));
    auto gradients = std::make_shared<std::vector<Eigen::Vector2d>>(
        reconstruct_gradients(mesh, values, *adjacency));
    return [&mesh, values, locator = std::move(locator), adjacency = std::move(adjacency),
            gradients = std::move(gradients)](const Point2D point) -> std::optional<double> {
        const auto donor = locator->find(point);
        if (!donor) {
            return std::nullopt;
        }
        return reconstruct(mesh, values, *adjacency, *gradients, *donor, point);
    };
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
