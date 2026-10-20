#include "basin/layered_mesh.hpp"
#include "mesh/strip_mesh.hpp"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <utility>

namespace {

BasinGeometry basin_geometry(std::vector<BasinLayerGeometry> layers) {
    BasinConfiguration input;
    for (auto& layer : layers) {
        input.layers.push_back({.id = layer.id,
                                .name = {},
                                .x = std::move(layer.x),
                                .top = std::move(layer.top),
                                .bottom = std::move(layer.bottom),
                                .lithotype = {},
                                .porosity = {}});
    }
    return make_basin_geometry(input);
}

MeshRegion rectangle(const int id, const double top, const double bottom) {
    return {.id = id,
            .boundary = {{0., top}, {4., top}, {4., bottom}, {0., bottom}},
            .edge_kinds = {BoundaryKind::top, BoundaryKind::right, BoundaryKind::bottom,
                           BoundaryKind::left}};
}

TEST(StripMesh, BuildsPredominantlyConvexQuadsInWideRegion) {
    const auto region = rectangle(1, 0., 2.);
    const auto mesh = make_strip_mesh({&region, 1}, .5);
    const auto quads = std::count_if(mesh.cells().begin(), mesh.cells().end(),
                                     [](const auto* c) { return c->n_vertices() == 4; });
    EXPECT_GT(quads * 2, static_cast<std::ptrdiff_t>(mesh.cells().size()));
    const auto geometry =
        basin_geometry({{.id = 1, .x = {0., 4.}, .top = {0., 0.}, .bottom = {2., 2.}}});
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, geometry));
}

TEST(StripMesh, KeepsSharedInterfaceConformingAcrossDifferentSubdivisions) {
    auto first = rectangle(1, 0., 1.);
    auto second = rectangle(2, 1., 3.);
    second.boundary.insert(second.boundary.begin() + 1, {1.25, 1.});
    second.edge_kinds.insert(second.edge_kinds.begin(), BoundaryKind::top);
    const std::array regions{first, second};
    const auto mesh = make_strip_mesh(regions, .5);
    const auto geometry =
        basin_geometry({{.id = 1, .x = {0., 4.}, .top = {0., 0.}, .bottom = {1., 1.}},
                        {.id = 2, .x = {0., 4.}, .top = {1., 1.}, .bottom = {3., 3.}}});
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, geometry));
    EXPECT_NEAR(mesh.area(), 12., 1e-10);
}

TEST(StripMesh, AlignsSlopingContactBetweenLayersWithDifferentRowOrigins) {
    const std::array regions{MeshRegion{.id = 1,
                                        .boundary = {{0., 0.}, {4., 0.}, {4., 3.}, {0., 1.}},
                                        .edge_kinds = {BoundaryKind::top, BoundaryKind::right,
                                                       BoundaryKind::bottom, BoundaryKind::left}},
                             MeshRegion{.id = 2,
                                        .boundary = {{0., 1.}, {4., 3.}, {4., 5.2}, {0., 5.2}},
                                        .edge_kinds = {BoundaryKind::top, BoundaryKind::right,
                                                       BoundaryKind::bottom, BoundaryKind::left}}};
    const auto mesh = make_strip_mesh(regions, .5);
    const auto geometry =
        basin_geometry({{.id = 1, .x = {0., 4.}, .top = {0., 0.}, .bottom = {1., 3.}},
                        {.id = 2, .x = {0., 4.}, .top = {1., 3.}, .bottom = {5.2, 5.2}}});
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, geometry));
    EXPECT_NEAR(mesh.area(), 20.8, 1e-10);
}

TEST(StripMesh, TerminatesStripsInNarrowingLayerWithoutLosingCoverage) {
    MeshRegion region{.id = 1,
                      .boundary = {{0., 0.}, {10., 4.5}, {10., 5.}, {0., 5.}},
                      .edge_kinds = {BoundaryKind::top, BoundaryKind::right, BoundaryKind::bottom,
                                     BoundaryKind::left}};
    const auto mesh = make_strip_mesh({&region, 1}, .5);
    const auto geometry =
        basin_geometry({{.id = 1, .x = {0., 10.}, .top = {0., 4.5}, .bottom = {5., 5.}}});
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, geometry));
    EXPECT_TRUE(std::any_of(mesh.cells().begin(), mesh.cells().end(),
                            [](const auto* c) { return c->n_vertices() == 3; }));
}

TEST(StripMesh, AvoidsFlatTriangleMadeOfNearlyCollinearContourVertices) {
    // Reproduces a boundary-only needle from data2; the small corner must not
    // become a separate triangle whose three vertices all lie on the contour.
    const auto geometry = basin_geometry({{.id = 1,
                                           .x = {0., 17.01001, 19.01001},
                                           .top = {0., .409058, .457154},
                                           .bottom = {50., 50., 50.}}});
    const auto mesh = make_layer_strip_mesh(geometry, 50.);
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, geometry));
    for (const auto* cell : mesh.cells()) {
        const double length = cell->max_edge_length();
        EXPECT_LT(length * length / cell->area(), 1000.);
    }
}

TEST(StripMesh, RefinesBackgroundColumnsNearContourCornerWithoutExtendingThroughDepth) {
    const auto geometry = basin_geometry(
        {{.id = 1, .x = {0., 1.25, 4.}, .top = {0., .2, 0.}, .bottom = {8., 8., 8.}}});
    const auto mesh = make_layer_strip_mesh(geometry, 1.);
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, geometry));
    bool near = false, far = false;
    for (const auto point : mesh.vertex_coordinates()) {
        if (std::abs(point.x - 1.25) < 1e-10) {
            near = near || std::abs(point.z - 1.) < 1e-10;
            far = far || std::abs(point.z - 5.) < 1e-10;
        }
    }
    EXPECT_TRUE(near);
    EXPECT_FALSE(far);
}

TEST(StripMesh, RemovesThinGridRemainderWithoutMovingSlopingMaterialBoundary) {
    const auto geometry =
        basin_geometry({{.id = 1, .x = {0., 4.}, .top = {.99, 1.02}, .bottom = {3., 3.}}});
    const auto mesh = make_layer_strip_mesh(geometry, 1.);
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, geometry));
    EXPECT_NEAR(mesh.area(), 7.98, 1e-10);
    // Allow ordinary transition triangles; reject the hundredth-height slivers
    // created when the contour nearly coincides with a horizontal background line.
    for (const auto* cell : mesh.cells()) {
        const double length = cell->max_edge_length();
        EXPECT_LT(length * length / cell->area(), 10.);
    }
}

TEST(StripMesh, PreservesSharedContactCornerWithAlmostCoincidentCollinearNeighbours) {
    constexpr double offset = 1e-6;
    const double middle = 1.9, right = 1.800004;
    MeshRegion upper{.id = 1,
                     .boundary = {{0., 0.},
                                  {4., 0.},
                                  {4., right},
                                  {2. + offset, middle + .5 * offset * (right - middle)},
                                  {2., middle},
                                  {2. - offset, middle + .5 * offset * (2. - middle)},
                                  {0., 2.}},
                     .edge_kinds = {BoundaryKind::top, BoundaryKind::right, BoundaryKind::bottom,
                                    BoundaryKind::bottom, BoundaryKind::bottom,
                                    BoundaryKind::bottom, BoundaryKind::left}};
    MeshRegion lower{.id = 2,
                     .boundary = {{0., 2.}, {2., middle}, {4., right}, {4., 3.}, {0., 3.}},
                     .edge_kinds = {BoundaryKind::top, BoundaryKind::top, BoundaryKind::right,
                                    BoundaryKind::bottom, BoundaryKind::left}};
    const std::array regions{upper, lower};
    const auto mesh = make_strip_mesh(regions, 1.);
    const auto geometry = basin_geometry(
        {{.id = 1, .x = {0., 2., 4.}, .top = {0., 0., 0.}, .bottom = {2., middle, right}},
         {.id = 2, .x = {0., 2., 4.}, .top = {2., middle, right}, .bottom = {3., 3., 3.}}});
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, geometry));
    EXPECT_NEAR(mesh.area(), 12., 1e-10);
}

} // namespace
