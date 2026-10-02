#pragma once

#include "Mesh.h"

#include <array>
#include <memory>
#include <span>
#include <vector>

struct Point2D {
    double x = 0.0;
    double z = 0.0;
};

enum class BoundaryKind : int {
    top = 1,
    bottom = 2,
    left = 3,
    right = 4,
    interface = 5,
};

class Mesh2D {
public:
    using NativeType = TQMesh::Mesh;
    using CellView = std::span<const TQMesh::Facet* const>;
    using EdgeView = std::span<const TQMesh::Edge* const>;

    struct Cell {
        std::vector<Point2D> vertices;
        std::vector<BoundaryKind> edge_kinds;
        int id = 0;
    };

    static Mesh2D from_cells(std::span<const Cell> cells);

    Mesh2D(Mesh2D&&) noexcept;
    Mesh2D& operator=(Mesh2D&&) noexcept;
    Mesh2D(const Mesh2D&) = delete;
    Mesh2D& operator=(const Mesh2D&) = delete;
    ~Mesh2D();

    [[nodiscard]] NativeType& native() noexcept;
    [[nodiscard]] const NativeType& native() const noexcept;

    [[nodiscard]] CellView cells() const noexcept;
    [[nodiscard]] EdgeView internal_edges() const noexcept;
    [[nodiscard]] EdgeView boundary_edges(BoundaryKind kind) const;

    [[nodiscard]] const TQMesh::Facet& owner(const TQMesh::Edge& edge) const;
    [[nodiscard]] const TQMesh::Facet& neighbor(const TQMesh::Edge& edge) const;
    [[nodiscard]] CppUtils::Vec2d normal_from_owner(const TQMesh::Edge& edge) const;
    [[nodiscard]] BoundaryKind boundary_kind(const TQMesh::Edge& edge) const;
    [[nodiscard]] double area() const noexcept;
    [[nodiscard]] std::vector<Point2D> vertex_coordinates() const;

    void set_vertex_coordinates(std::span<const Point2D> coordinates);

    void validate(double relative_tolerance = 1.0e-10) const;

private:
    struct Storage;

    explicit Mesh2D(std::unique_ptr<Storage> storage) noexcept;

    std::unique_ptr<Storage> storage_;
};
