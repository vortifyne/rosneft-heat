#include "mesh/mesh2d.hpp"
#include "mesh_test_utils.hpp"

#include <gtest/gtest.h>
#include <set>
#include <stdexcept>
#include <vector>

TEST(Mesh2D, BuildsExplicitCellsWithoutCopyingNativeMesh) {
    Mesh2D mesh = make_rectangular_test_mesh(4.0, 2.0, 0.5, 17);

    const auto& native = mesh.native();
    EXPECT_GT(native.n_vertices(), 0U);
    EXPECT_GT(native.n_elements(), 0U);
    EXPECT_GT(native.n_interior_edges(), 0U);
    EXPECT_GT(native.n_boundary_edges(), 0U);
    EXPECT_NEAR(mesh.area(), 8.0, 1.0e-10);

    std::set<BoundaryKind> boundary_kinds;
    for (const auto& edge : native.boundary_edges()) {
        boundary_kinds.insert(mesh.boundary_kind(*edge));
        EXPECT_NEAR(mesh.normal_from_owner(*edge).norm(), 1.0, 1.0e-12);
    }
    EXPECT_EQ(boundary_kinds, (std::set{BoundaryKind::top, BoundaryKind::bottom, BoundaryKind::left,
                                        BoundaryKind::right}));
    for (const auto& cell : native.quads()) {
        EXPECT_EQ(cell->color(), 17);
    }
    EXPECT_NO_THROW(mesh.validate());
}

TEST(Mesh2D, RejectsClockwiseCell) {
    const Mesh2D::Cell cell{.vertices = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}},
                            .edge_kinds = {BoundaryKind::top, BoundaryKind::left,
                                           BoundaryKind::bottom, BoundaryKind::right},
                            .id = 1};

    EXPECT_THROW(Mesh2D::from_cells(std::span<const Mesh2D::Cell>(&cell, 1)),
                 std::invalid_argument);
}

TEST(Mesh2D, UpdatesVertexCoordinatesWithoutChangingConnectivity) {
    Mesh2D mesh = make_rectangular_test_mesh(4.0, 2.0, 0.5, 17);
    const std::size_t cell_count = mesh.cells().size();
    const std::size_t edge_count = mesh.internal_edges().size();
    std::vector<Point2D> coordinates = mesh.vertex_coordinates();
    for (Point2D& point : coordinates) {
        point.z *= 1.25;
    }

    mesh.set_vertex_coordinates(coordinates);

    EXPECT_EQ(mesh.cells().size(), cell_count);
    EXPECT_EQ(mesh.internal_edges().size(), edge_count);
    EXPECT_NEAR(mesh.area(), 10.0, 1.0e-9);
    EXPECT_NO_THROW(mesh.validate());
}

TEST(Mesh2D, BuildsQuadrilateralAndTriangleCells) {
    const std::vector<Mesh2D::Cell> cells = {
        {.vertices = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}},
         .edge_kinds = {BoundaryKind::top, BoundaryKind::right, BoundaryKind::bottom,
                        BoundaryKind::left},
         .id = 11},
        {.vertices = {{1.0, 0.0}, {2.0, 0.5}, {1.0, 1.0}},
         .edge_kinds = {BoundaryKind::top, BoundaryKind::bottom, BoundaryKind::interface},
         .id = 12},
    };

    Mesh2D mesh = Mesh2D::from_cells(cells);

    EXPECT_EQ(mesh.native().n_quads(), 1U);
    EXPECT_EQ(mesh.native().n_triangles(), 1U);
    EXPECT_EQ(mesh.internal_edges().size(), 1U);
    EXPECT_NEAR(mesh.area(), 1.5, 1.0e-12);
    EXPECT_NO_THROW(mesh.validate());
}
