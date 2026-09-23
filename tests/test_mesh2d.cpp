#include "mesh/mesh2d.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

Mesh2D::Region rectangle(const double x0, const double z0, const double x1, const double z1,
                         const int id, std::array<BoundaryKind, 4> kinds) {
    return {
        .vertices = {{x0, z0}, {x1, z0}, {x1, z1}, {x0, z1}},
        .edge_kinds = {kinds.begin(), kinds.end()},
        .id = id,
    };
}

TEST(Mesh2D, GeneratesRectangleWithoutCopyingNativeMesh) {
    const Mesh2D::Region region = rectangle(
        0.0, 0.0, 4.0, 2.0, 17,
        {BoundaryKind::bottom, BoundaryKind::right, BoundaryKind::top, BoundaryKind::left});
    Mesh2D::GenerationOptions options;
    options.cell_size = [](Point2D) { return 0.5; };
    Mesh2D mesh = Mesh2D::generate(std::span<const Mesh2D::Region>(&region, 1), options);

    const auto& native = mesh.native();
    EXPECT_GT(native.n_vertices(), 0U);
    EXPECT_GT(native.n_elements(), 0U);
    EXPECT_GT(native.n_interior_edges(), 0U);
    EXPECT_GT(native.n_boundary_edges(), 0U);
    EXPECT_NEAR(mesh.area(), 8.0, 1.0e-10);

    std::set<BoundaryKind> boundary_kinds;
    for (const auto& edge : native.boundary_edges()) {
        boundary_kinds.insert(mesh.boundary_kind(*edge));
        const auto normal = mesh.normal_from_owner(*edge);
        EXPECT_NEAR(normal.norm(), 1.0, 1.0e-12);
    }
    EXPECT_EQ(boundary_kinds, (std::set{BoundaryKind::top, BoundaryKind::bottom, BoundaryKind::left,
                                        BoundaryKind::right}));
    for (const auto& cell : native.triangles()) {
        EXPECT_EQ(cell->color(), 17);
    }
    EXPECT_NO_THROW(mesh.validate());
}

TEST(Mesh2D, MergesTwoRegionsAcrossLayerInterface) {
    const std::vector<Mesh2D::Region> regions = {
        rectangle(0.0, 0.0, 4.0, 1.0, 1001,
                  {BoundaryKind::bottom, BoundaryKind::right, BoundaryKind::interface,
                   BoundaryKind::left}),
        rectangle(
            0.0, 1.0, 4.0, 2.0, 1002,
            {BoundaryKind::interface, BoundaryKind::right, BoundaryKind::top, BoundaryKind::left}),
    };
    Mesh2D::GenerationOptions options;
    options.cell_size = [](Point2D) { return 0.5; };
    Mesh2D mesh = Mesh2D::generate(regions, options);

    EXPECT_NEAR(mesh.area(), 8.0, 1.0e-10);
    EXPECT_NO_THROW(mesh.validate());

    bool found_layer_interface = false;
    for (const auto& edge : mesh.native().interior_edges()) {
        found_layer_interface |= mesh.owner(*edge).color() != mesh.neighbor(*edge).color();
    }
    EXPECT_TRUE(found_layer_interface);
    for (const auto& edge : mesh.native().boundary_edges()) {
        EXPECT_NE(mesh.boundary_kind(*edge), BoundaryKind::interface);
    }
}

TEST(Mesh2D, CanGenerateAllQuadrilateralDiagnosticMesh) {
    const Mesh2D::Region region = rectangle(
        0.0, 0.0, 2.0, 1.0, 3,
        {BoundaryKind::bottom, BoundaryKind::right, BoundaryKind::top, BoundaryKind::left});
    const std::filesystem::path output =
        std::filesystem::temp_directory_path() / "heat_mesh2d_rectangle.vtu";
    Mesh2D mesh = Mesh2D::generate(std::span<const Mesh2D::Region>(&region, 1),
                                   {.cell_size = [](Point2D) { return 0.4; },
                                    .make_quadrilateral = true,
                                    .diagnostic_vtu = output});

    EXPECT_EQ(mesh.native().n_triangles(), 0U);
    EXPECT_GT(mesh.native().n_quads(), 0U);
    EXPECT_TRUE(std::filesystem::is_regular_file(output));
    std::filesystem::remove(output);
}

TEST(Mesh2D, RejectsClockwiseRegion) {
    Mesh2D::Region region = rectangle(
        0.0, 0.0, 1.0, 1.0, 1,
        {BoundaryKind::bottom, BoundaryKind::right, BoundaryKind::top, BoundaryKind::left});
    std::reverse(region.vertices.begin(), region.vertices.end());

    Mesh2D::GenerationOptions options;
    options.cell_size = [](Point2D) { return 0.25; };
    EXPECT_THROW(Mesh2D::generate(std::span<const Mesh2D::Region>(&region, 1), options),
                 std::invalid_argument);
}

} // namespace
