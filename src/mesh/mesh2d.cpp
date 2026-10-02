#include "mesh/mesh2d.hpp"

#include "MeshCleanup.h"
#include "TQMeshSetup.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

struct Mesh2D::Storage {
    std::unique_ptr<TQMesh::Mesh> owned_mesh;
    TQMesh::Mesh* mesh = nullptr;
    double expected_area = 0.0;
    std::vector<const TQMesh::Facet*> cells;
    std::vector<const TQMesh::Edge*> internal_edges;
    std::vector<const TQMesh::Edge*> top_edges;
    std::vector<const TQMesh::Edge*> bottom_edges;
    std::vector<const TQMesh::Edge*> left_edges;
    std::vector<const TQMesh::Edge*> right_edges;
};

namespace {

using CppUtils::Vec2d;
using TQMesh::Facet;
using TQMesh::Mesh;
using TQMesh::MeshCleanup;
using TQMesh::NullFacet;
using TQMesh::TQMeshSetup;

double dot(const Vec2d& lhs, const Vec2d& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y;
}

double signed_polygon_area(std::span<const Point2D> vertices) {
    double twice_area = 0.0;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const Point2D& current = vertices[i];
        const Point2D& next = vertices[(i + 1) % vertices.size()];
        twice_area += current.x * next.z - next.x * current.z;
    }
    return 0.5 * twice_area;
}

BoundaryKind boundary_kind_from_color(const int color) {
    switch (color) {
    case static_cast<int>(BoundaryKind::top):
        return BoundaryKind::top;
    case static_cast<int>(BoundaryKind::bottom):
        return BoundaryKind::bottom;
    case static_cast<int>(BoundaryKind::left):
        return BoundaryKind::left;
    case static_cast<int>(BoundaryKind::right):
        return BoundaryKind::right;
    case static_cast<int>(BoundaryKind::interface):
        return BoundaryKind::interface;
    default:
        throw std::runtime_error("TQMesh returned an unknown boundary color");
    }
}

bool is_null_facet(const Facet* facet) {
    return facet == nullptr || NullFacet::is_null(const_cast<Facet*>(facet));
}

std::uint64_t edge_key(std::size_t first, std::size_t second) {
    if (second < first) {
        std::swap(first, second);
    }
    if (first > static_cast<std::size_t>(UINT32_MAX) ||
        second > static_cast<std::size_t>(UINT32_MAX)) {
        throw std::runtime_error("Mesh vertex index exceeds the validation range");
    }
    return (static_cast<std::uint64_t>(first) << 32U) | static_cast<std::uint64_t>(second);
}

template <typename Storage> void populate_views(Storage& storage) {
    MeshCleanup::assign_mesh_indices(*storage.mesh);
    MeshCleanup::setup_facet_connectivity(*storage.mesh);

    storage.cells.resize(storage.mesh->n_elements());
    for (const auto& cell : storage.mesh->quads()) {
        storage.cells.at(static_cast<std::size_t>(cell->index())) = cell.get();
    }
    for (const auto& cell : storage.mesh->triangles()) {
        storage.cells.at(static_cast<std::size_t>(cell->index())) = cell.get();
    }

    storage.internal_edges.reserve(storage.mesh->n_interior_edges());
    for (const auto& edge : storage.mesh->interior_edges()) {
        storage.internal_edges.push_back(edge.get());
    }
    for (const auto& edge : storage.mesh->boundary_edges()) {
        switch (boundary_kind_from_color(edge->color())) {
        case BoundaryKind::top:
            storage.top_edges.push_back(edge.get());
            break;
        case BoundaryKind::bottom:
            storage.bottom_edges.push_back(edge.get());
            break;
        case BoundaryKind::left:
            storage.left_edges.push_back(edge.get());
            break;
        case BoundaryKind::right:
            storage.right_edges.push_back(edge.get());
            break;
        case BoundaryKind::interface:
            throw std::runtime_error("An unmerged interface remains on the mesh boundary");
        }
    }
}

} // namespace

Mesh2D Mesh2D::from_cells(const std::span<const Cell> cells) {
    if (cells.empty()) {
        throw std::invalid_argument("Mesh2D requires at least one cell");
    }
    double scale = 1.0;
    for (const Cell& cell : cells) {
        if (cell.vertices.size() != 3 && cell.vertices.size() != 4) {
            throw std::invalid_argument("Mesh2D cell must have three or four vertices");
        }
        if (cell.edge_kinds.size() != cell.vertices.size()) {
            throw std::invalid_argument("Mesh2D cell requires one kind per edge");
        }
        if (!(signed_polygon_area(cell.vertices) > 0.0)) {
            throw std::invalid_argument("Mesh2D cell must be counter-clockwise and non-degenerate");
        }
        for (const Point2D point : cell.vertices) {
            if (!std::isfinite(point.x) || !std::isfinite(point.z)) {
                throw std::invalid_argument("Mesh2D cell coordinates must be finite");
            }
            scale = std::max({scale, std::abs(point.x), std::abs(point.z)});
        }
    }
    TQMeshSetup::get_instance().set_quadtree_scale(4.2 * scale);

    struct EdgeRecord {
        std::size_t first = 0;
        std::size_t second = 0;
        BoundaryKind kind = BoundaryKind::interface;
        int count = 0;
    };
    const double coordinate_tolerance = 1.0e-10 * scale;
    const auto coordinate_key = [coordinate_tolerance](const Point2D point) {
        return std::pair{static_cast<std::int64_t>(std::llround(point.x / coordinate_tolerance)),
                         static_cast<std::int64_t>(std::llround(point.z / coordinate_tolerance))};
    };

    auto storage = std::make_unique<Storage>();
    storage->owned_mesh = std::make_unique<Mesh>();
    storage->mesh = storage->owned_mesh.get();
    std::map<std::pair<std::int64_t, std::int64_t>, std::size_t> vertex_indices;
    std::vector<TQMesh::Vertex*> vertices;
    std::unordered_map<std::uint64_t, EdgeRecord> edges;

    for (const Cell& cell : cells) {
        std::vector<std::size_t> indices;
        indices.reserve(cell.vertices.size());
        for (const Point2D point : cell.vertices) {
            const auto [position, inserted] =
                vertex_indices.emplace(coordinate_key(point), vertices.size());
            if (inserted) {
                vertices.push_back(&storage->mesh->add_vertex({point.x, point.z}));
            } else {
                const auto& existing = vertices[position->second]->xy();
                if (std::max(std::abs(existing.x - point.x), std::abs(existing.y - point.z)) >
                    coordinate_tolerance) {
                    throw std::runtime_error("Mesh2D coordinate key collision");
                }
            }
            indices.push_back(position->second);
        }
        if (indices.size() == 3) {
            storage->mesh->add_triangle(*vertices[indices[0]], *vertices[indices[1]],
                                        *vertices[indices[2]], cell.id);
        } else {
            storage->mesh->add_quad(*vertices[indices[0]], *vertices[indices[1]],
                                    *vertices[indices[2]], *vertices[indices[3]], cell.id);
        }
        storage->expected_area += signed_polygon_area(cell.vertices);
        for (std::size_t edge = 0; edge < indices.size(); ++edge) {
            const std::size_t first = indices[edge];
            const std::size_t second = indices[(edge + 1) % indices.size()];
            EdgeRecord& record = edges[edge_key(first, second)];
            if (record.count == 0) {
                record = {
                    .first = first, .second = second, .kind = cell.edge_kinds[edge], .count = 1};
            } else {
                ++record.count;
                if (record.count > 2) {
                    throw std::runtime_error("Mesh2D cell edge belongs to more than two cells");
                }
            }
        }
    }

    for (const auto& [key, edge] : edges) {
        (void)key;
        if (edge.count == 2) {
            storage->mesh->add_interior_edge(*vertices[edge.first], *vertices[edge.second]);
        } else {
            if (edge.kind == BoundaryKind::interface) {
                throw std::runtime_error("Mesh2D has an exposed interface edge");
            }
            storage->mesh->add_boundary_edge(*vertices[edge.first], *vertices[edge.second],
                                             static_cast<int>(edge.kind));
        }
    }
    populate_views(*storage);

    Mesh2D result(std::move(storage));
    result.validate();
    return result;
}

Mesh2D::Mesh2D(std::unique_ptr<Storage> storage) noexcept : storage_(std::move(storage)) {}

Mesh2D::Mesh2D(Mesh2D&&) noexcept = default;
Mesh2D& Mesh2D::operator=(Mesh2D&&) noexcept = default;
Mesh2D::~Mesh2D() = default;

Mesh2D::NativeType& Mesh2D::native() noexcept {
    return *storage_->mesh;
}

const Mesh2D::NativeType& Mesh2D::native() const noexcept {
    return *storage_->mesh;
}

Mesh2D::CellView Mesh2D::cells() const noexcept {
    return storage_->cells;
}

Mesh2D::EdgeView Mesh2D::internal_edges() const noexcept {
    return storage_->internal_edges;
}

Mesh2D::EdgeView Mesh2D::boundary_edges(const BoundaryKind kind) const {
    switch (kind) {
    case BoundaryKind::top:
        return storage_->top_edges;
    case BoundaryKind::bottom:
        return storage_->bottom_edges;
    case BoundaryKind::left:
        return storage_->left_edges;
    case BoundaryKind::right:
        return storage_->right_edges;
    case BoundaryKind::interface:
        throw std::invalid_argument("Layer interfaces are internal mesh edges");
    }
    throw std::invalid_argument("Unknown boundary kind");
}

const TQMesh::Facet& Mesh2D::owner(const TQMesh::Edge& edge) const {
    const Facet* owner_cell = edge.facet_l();
    if (is_null_facet(owner_cell)) {
        throw std::runtime_error("Mesh edge is missing its owner cell");
    }
    return *owner_cell;
}

const TQMesh::Facet& Mesh2D::neighbor(const TQMesh::Edge& edge) const {
    const Facet* neighbor_cell = edge.facet_r();
    if (is_null_facet(neighbor_cell)) {
        throw std::runtime_error("Mesh edge is missing its neighbor cell");
    }
    return *neighbor_cell;
}

Vec2d Mesh2D::normal_from_owner(const TQMesh::Edge& edge) const {
    const Facet& owner_cell = owner(edge);
    Vec2d direction;
    if (edge.is_interior()) {
        direction = neighbor(edge).xy() - owner_cell.xy();
    } else {
        direction = edge.xy() - owner_cell.xy();
    }
    Vec2d normal = edge.normal();
    if (dot(normal, direction) < 0.0) {
        normal *= -1.0;
    }
    return normal;
}

BoundaryKind Mesh2D::boundary_kind(const TQMesh::Edge& edge) const {
    if (!edge.on_boundary()) {
        throw std::invalid_argument("Boundary kind requested for an interior edge");
    }
    return boundary_kind_from_color(edge.color());
}

double Mesh2D::area() const noexcept {
    double result = 0.0;
    for (const auto& cell : native().quads()) {
        result += cell->area();
    }
    for (const auto& cell : native().triangles()) {
        result += cell->area();
    }
    return result;
}

std::vector<Point2D> Mesh2D::vertex_coordinates() const {
    std::vector<Point2D> result(native().n_vertices());
    for (const auto& vertex : native().vertices()) {
        result.at(vertex->index()) = {vertex->xy().x, vertex->xy().y};
    }
    return result;
}

void Mesh2D::set_vertex_coordinates(const std::span<const Point2D> coordinates) {
    if (coordinates.size() != native().n_vertices()) {
        throw std::invalid_argument("Mesh coordinate count must match the vertex count");
    }
    for (const Point2D coordinate : coordinates) {
        if (!std::isfinite(coordinate.x) || !std::isfinite(coordinate.z)) {
            throw std::invalid_argument("Mesh vertex coordinates must be finite");
        }
    }
    for (auto& vertex : native().vertices()) {
        const Point2D coordinate = coordinates[vertex->index()];
        MeshCleanup::set_vertex_coordinates(*vertex, {coordinate.x, coordinate.z});
    }
    storage_->expected_area = area();
    validate();
}

void Mesh2D::validate(const double relative_tolerance) const {
    if (!std::isfinite(relative_tolerance) || relative_tolerance < 0.0) {
        throw std::invalid_argument("Mesh validation tolerance must be non-negative");
    }
    const Mesh& mesh = native();
    if (mesh.n_vertices() == 0 || mesh.n_elements() == 0) {
        throw std::runtime_error("Mesh must contain vertices and cells");
    }

    std::unordered_map<std::uint64_t, int> cell_edge_counts;
    cell_edge_counts.reserve(mesh.n_edges());
    const auto validate_cell = [&](const Facet& cell) {
        if (cell.n_vertices() != 3 && cell.n_vertices() != 4) {
            throw std::runtime_error("Mesh cell must have three or four vertices");
        }
        if (!std::isfinite(cell.area()) || !(cell.area() > 0.0) || !std::isfinite(cell.xy().x) ||
            !std::isfinite(cell.xy().y)) {
            throw std::runtime_error(
                "Mesh cell geometry is invalid: index=" + std::to_string(cell.index()) +
                " area=" + std::to_string(cell.area()) + " x=" + std::to_string(cell.xy().x) +
                " z=" + std::to_string(cell.xy().y));
        }
        for (std::size_t i = 0; i < cell.n_vertices(); ++i) {
            const std::size_t first = cell.vertex(i).index();
            const std::size_t second = cell.vertex((i + 1) % cell.n_vertices()).index();
            if (first >= mesh.n_vertices() || second >= mesh.n_vertices() || first == second) {
                throw std::runtime_error("Mesh cell has an invalid vertex index");
            }
            ++cell_edge_counts[edge_key(first, second)];
        }
    };
    for (const auto& cell : mesh.quads()) {
        validate_cell(*cell);
    }
    for (const auto& cell : mesh.triangles()) {
        validate_cell(*cell);
    }

    std::unordered_map<std::uint64_t, int> face_counts;
    face_counts.reserve(mesh.n_edges());
    for (const auto& edge : mesh.interior_edges()) {
        const Facet& owner_cell = owner(*edge);
        const Facet& neighbor_cell = neighbor(*edge);
        if (&owner_cell == &neighbor_cell || !(edge->length() > 0.0) ||
            !std::isfinite(edge->length())) {
            throw std::runtime_error("Interior edge geometry or connectivity is invalid");
        }
        if (!(dot(normal_from_owner(*edge), neighbor_cell.xy() - owner_cell.xy()) > 0.0)) {
            throw std::runtime_error("Interior edge normal has invalid direction");
        }
        ++face_counts[edge_key(edge->v1().index(), edge->v2().index())];
    }
    for (const auto& edge : mesh.boundary_edges()) {
        const Facet& owner_cell = owner(*edge);
        if (!(edge->length() > 0.0) || !std::isfinite(edge->length())) {
            throw std::runtime_error("Boundary edge length is invalid");
        }
        if (boundary_kind(*edge) == BoundaryKind::interface) {
            throw std::runtime_error("An unmerged interface remains on the mesh boundary near (" +
                                     std::to_string(edge->xy().x) + ", " +
                                     std::to_string(edge->xy().y) + ")");
        }
        if (!(dot(normal_from_owner(*edge), edge->xy() - owner_cell.xy()) > 0.0)) {
            throw std::runtime_error("Boundary edge normal does not point outward");
        }
        ++face_counts[edge_key(edge->v1().index(), edge->v2().index())];
    }

    if (face_counts.size() != cell_edge_counts.size()) {
        throw std::runtime_error("Mesh contains missing or duplicate edges");
    }
    for (const auto& [key, cell_count] : cell_edge_counts) {
        const auto face = face_counts.find(key);
        if (face == face_counts.end() || face->second != 1 ||
            (cell_count != 1 && cell_count != 2)) {
            throw std::runtime_error("Mesh edge connectivity is inconsistent");
        }
    }

    const double scale = std::max(1.0, std::abs(storage_->expected_area));
    if (std::abs(area() - storage_->expected_area) > relative_tolerance * scale) {
        throw std::runtime_error("Mesh area " + std::to_string(area()) +
                                 " does not match region area " +
                                 std::to_string(storage_->expected_area));
    }
}
