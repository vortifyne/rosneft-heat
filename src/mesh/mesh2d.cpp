#include "mesh/mesh2d.hpp"

#include "Domain.h"
#include "MeshChecker.h"
#include "MeshCleanup.h"
#include "MeshGenerator.h"
#include "TQMeshSetup.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

struct Mesh2D::Storage {
    std::vector<std::unique_ptr<TQMesh::Domain>> domains;
    TQMesh::MeshGenerator generator;
    TQMesh::Mesh* mesh = nullptr;
    double expected_area = 0.0;
};

namespace {

using CppUtils::Vec2d;
using TQMesh::Domain;
using TQMesh::Facet;
using TQMesh::Mesh;
using TQMesh::MeshChecker;
using TQMesh::MeshCleanup;
using TQMesh::MeshExportType;
using TQMesh::NullFacet;
using TQMesh::TQMeshSetup;
using TQMesh::UserSizeFunction;

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

void validate_generation_input(std::span<const Mesh2D::Region> regions,
                               const Mesh2D::GenerationOptions& options) {
    if (regions.empty()) {
        throw std::invalid_argument("Mesh2D requires at least one region");
    }
    if (!options.cell_size) {
        throw std::invalid_argument("Mesh2D requires a cell-size function");
    }
    if (options.smoothing_iterations < 0) {
        throw std::invalid_argument("Mesh2D smoothing iteration count cannot be negative");
    }

    for (const auto& region : regions) {
        if (region.vertices.size() < 3) {
            throw std::invalid_argument("A mesh region requires at least three vertices");
        }
        if (region.edge_kinds.size() != region.vertices.size()) {
            throw std::invalid_argument("A mesh region requires one edge kind per polygon edge");
        }
        if (!(signed_polygon_area(region.vertices) > 0.0)) {
            throw std::invalid_argument(
                "Mesh region vertices must be counter-clockwise and non-degenerate");
        }
        for (const Point2D point : region.vertices) {
            if (!std::isfinite(point.x) || !std::isfinite(point.z)) {
                throw std::invalid_argument("Mesh region coordinates must be finite");
            }
        }
    }
}

double quadtree_scale(std::span<const Mesh2D::Region> regions) {
    double scale = 1.0;
    for (const auto& region : regions) {
        for (const Point2D point : region.vertices) {
            scale = std::max(scale, std::abs(point.x));
            scale = std::max(scale, std::abs(point.z));
        }
    }
    return 2.1 * scale;
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

} // namespace

Mesh2D Mesh2D::generate(std::span<const Region> regions, const GenerationOptions& options) {
    validate_generation_input(regions, options);
    TQMeshSetup::get_instance().set_quadtree_scale(quadtree_scale(regions));

    auto storage = std::make_unique<Storage>();
    storage->domains.reserve(regions.size());
    std::vector<Mesh*> meshes;
    meshes.reserve(regions.size());

    for (std::size_t region_index = 0; region_index < regions.size(); ++region_index) {
        const Region& region = regions[region_index];
        UserSizeFunction size_function = [&options](const Vec2d& point) {
            const double size = options.cell_size({point.x, point.y});
            if (!std::isfinite(size) || !(size > 0.0)) {
                throw std::runtime_error("Mesh cell size must be finite and positive");
            }
            return size;
        };

        auto domain = std::make_unique<Domain>(std::move(size_function));
        std::vector<Vec2d> coordinates;
        std::vector<int> colors;
        coordinates.reserve(region.vertices.size());
        colors.reserve(region.edge_kinds.size());
        for (const Point2D point : region.vertices) {
            coordinates.emplace_back(point.x, point.z);
        }
        for (const BoundaryKind kind : region.edge_kinds) {
            colors.push_back(static_cast<int>(kind));
        }
        domain->add_exterior_boundary().set_shape_from_coordinates(coordinates, colors);

        Mesh& mesh =
            storage->generator.new_mesh(*domain, static_cast<int>(region_index), region.id);
        if (!storage->generator.triangulation(mesh).generate_elements()) {
            throw std::runtime_error("TQMesh failed to generate mesh elements for region " +
                                     std::to_string(region_index) + " (identifier " +
                                     std::to_string(region.id) + ")");
        }
        if (options.make_quadrilateral) {
            storage->generator.tri2quad_modification(mesh).modify();
            if (!storage->generator.quad_refinement(mesh).refine()) {
                throw std::runtime_error("TQMesh failed to create an all-quad mesh");
            }
        }
        if (options.smoothing_iterations > 0) {
            storage->generator.mixed_smoothing(mesh).smooth(options.smoothing_iterations);
        }
        MeshChecker checker(mesh, *domain);
        if (!checker.check_completeness()) {
            throw std::runtime_error("TQMesh generated an incomplete mesh");
        }

        meshes.push_back(&mesh);
        storage->expected_area += signed_polygon_area(region.vertices);
        storage->domains.push_back(std::move(domain));
    }

    storage->mesh = meshes.front();
    std::vector<bool> merged(meshes.size(), false);
    merged.front() = true;
    std::size_t remaining = meshes.size() - 1;
    while (remaining > 0) {
        bool made_progress = false;
        for (std::size_t i = 1; i < meshes.size(); ++i) {
            if (merged[i]) {
                continue;
            }
            if (storage->generator.merge_meshes(*storage->mesh, *meshes[i])) {
                merged[i] = true;
                --remaining;
                made_progress = true;
            }
        }
        if (!made_progress) {
            throw std::runtime_error(
                "TQMesh could not connect all mesh regions through shared edges");
        }
    }
    MeshCleanup::assign_mesh_indices(*storage->mesh);
    MeshCleanup::setup_facet_connectivity(*storage->mesh);

    Mesh2D result(std::move(storage));
    result.validate();
    if (options.diagnostic_vtu) {
        result.write_vtu(*options.diagnostic_vtu);
    }
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

void Mesh2D::write_vtu(const std::filesystem::path& path) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    if (!storage_->generator.write_mesh(native(), path.string(), MeshExportType::VTU)) {
        throw std::runtime_error("TQMesh failed to write diagnostic VTU output");
    }
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
            throw std::runtime_error("Mesh cell geometry is invalid");
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
