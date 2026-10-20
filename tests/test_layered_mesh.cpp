#include "basin/layered_mesh.hpp"
#include "basin/mesh_motion.hpp"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {

BasinLayerProfile layer(const int id, const double top, const double bottom) {
    return {.id = id,
            .name = "layer",
            .x = {0.0, 2.0},
            .top = {top, top},
            .bottom = {bottom, bottom},
            .lithotype = {1, 1},
            .porosity = {0.1, 0.1}};
}

BasinConfiguration configuration(std::vector<BasinLayerProfile> layers) {
    return {.age_ma = 0.0,
            .layers = std::move(layers),
            .surface_temperature = {},
            .basal_heat_flux = {}};
}

TEST(LayeredMesh, BuildsConformingQuadrilateralsAcrossLayers) {
    const BasinConfiguration basin = configuration({layer(1, 0.0, 1.0), layer(2, 1.0, 2.0)});
    const LayeredMeshLayout layout = make_layered_mesh_layout({&basin, 1}, 1.0);

    Mesh2D result = make_layered_mesh(basin, layout);

    EXPECT_EQ(result.cells().size(), 4U);
    EXPECT_EQ(result.native().n_quads(), 4U);
    EXPECT_NEAR(result.area(), 4.0, 1.0e-12);
    EXPECT_NO_THROW(result.validate());
}

TEST(LayeredMesh, UsesTriangleFanAtLayerPinch) {
    BasinLayerProfile wedge = layer(7, 0.0, 1.0);
    wedge.bottom = {0.0, 2.0};
    const BasinConfiguration basin = configuration({std::move(wedge)});
    const LayeredMeshLayout layout = make_layered_mesh_layout({&basin, 1}, 1.0);

    Mesh2D result = make_layered_mesh(basin, layout);

    EXPECT_GT(result.native().n_triangles(), 0U);
    EXPECT_NEAR(result.area(), 2.0, 1.0e-12);
    EXPECT_NO_THROW(result.validate());
}

TEST(LayeredMesh, PreservesNearbyPinchesAndProfileBreaks) {
    BasinLayerProfile wedge = layer(7, 0.0, 1.0);
    wedge.x = {0.0, 80.0, 100.0, 200.0};
    wedge.top = {0.0, 0.0, 0.0, 0.0};
    wedge.bottom = {1.0, 0.0, 0.0, 1.0};
    wedge.lithotype = {1, 1, 1, 1};
    wedge.porosity = {0.1, 0.1, 0.1, 0.1};
    const BasinConfiguration basin = configuration({std::move(wedge)});

    const LayeredMeshLayout layout = make_layered_mesh_layout({&basin, 1}, 25.0);

    for (std::size_t index = 1; index < layout.x.size(); ++index) {
        EXPECT_GT(layout.x[index] - layout.x[index - 1], 0.0);
        EXPECT_LE(layout.x[index] - layout.x[index - 1], 25.0 + 1.0e-12);
    }
    EXPECT_EQ(std::count(layout.x.begin(), layout.x.end(), 80.0), 1);
    EXPECT_EQ(std::count(layout.x.begin(), layout.x.end(), 100.0), 1);
    const Mesh2D mesh = make_layered_mesh(basin, layout);
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, basin));
}

BasinConfiguration basin_with_nearby_layer_start() {
    auto upper = layer(1, 0.0, 1.0);
    upper.x = {0.0, 0.8, 1.2, 2.0};
    upper.top = {0.0, 0.0, 0.0, 0.0};
    upper.bottom = {1.0, 1.0, 1.0, 1.0};
    auto wedge = layer(2, 1.0, 2.0);
    wedge.x = {0.8, 1.2, 2.0};
    wedge.top = {1.0, 1.0, 1.0};
    wedge.bottom = {1.0, 2.0, 2.0};
    auto lower = layer(3, 1.0, 3.0);
    lower.x = upper.x;
    lower.top = {1.0, 1.0, 2.0, 2.0};
    lower.bottom = {3.0, 3.0, 3.0, 3.0};
    return configuration({std::move(upper), std::move(wedge), std::move(lower)});
}

TEST(LayeredMesh, CoversLayerStartingInsideFormerCoarseColumn) {
    const auto basin = basin_with_nearby_layer_start();
    const auto layout = make_layered_mesh_layout({&basin, 1}, 1.0);
    EXPECT_EQ(std::count(layout.x.begin(), layout.x.end(), 0.8), 1);
    EXPECT_EQ(std::count(layout.x.begin(), layout.x.end(), 1.2), 1);
    const Mesh2D mesh = make_layered_mesh(basin, layout);
    EXPECT_NEAR(mesh.area(), 6.0, 1.0e-12);
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, basin));

    // The former generator silently accepted this layout and left a triangular hole.
    const LayeredMeshLayout incomplete{.x = {0.0, 1.0, 2.0}, .cell_size = 1.0};
    EXPECT_THROW(make_layered_mesh(basin, incomplete), std::runtime_error);
}

TEST(LayeredMesh, StripMeshCoversNearbyLayerStartAndSharedContacts) {
    const auto geometry = make_basin_geometry(basin_with_nearby_layer_start());
    const Mesh2D mesh = make_layer_strip_mesh(geometry, .25);
    EXPECT_NEAR(mesh.area(), 6., 1e-10);
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, geometry));
}

TEST(LayeredMesh, StripMeshSeparatesComponentsAtZeroThickness) {
    auto lobes = layer(1, 0., 1.);
    lobes.x = {0., 1., 2., 3., 4.};
    lobes.top = {0., 0., 0., 0., 0.};
    lobes.bottom = {0., 1., 0., 1., 0.};
    const auto geometry = make_basin_geometry(configuration({lobes}));
    const Mesh2D mesh = make_layer_strip_mesh(geometry, .25);
    EXPECT_NEAR(mesh.area(), 2., 1e-10);
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, geometry));
}

TEST(LayeredMesh, CoverageRejectsCompensatingOverlapAndGap) {
    const auto basin = configuration({layer(1, 0.0, 1.0)});
    const std::vector<Mesh2D::Cell> cells{
        {.vertices = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}},
         .edge_kinds = {BoundaryKind::top, BoundaryKind::right, BoundaryKind::bottom,
                        BoundaryKind::left},
         .id = 1},
        {.vertices = {{0.5, 0.0}, {1.5, 0.0}, {1.5, 1.0}, {0.5, 1.0}},
         .edge_kinds = {BoundaryKind::top, BoundaryKind::right, BoundaryKind::bottom,
                        BoundaryKind::left},
         .id = 1}};
    const Mesh2D mesh = Mesh2D::from_cells(cells);
    EXPECT_NEAR(mesh.area(), 2.0, 1.0e-12);
    EXPECT_THROW(validate_layered_mesh_coverage(mesh, basin), std::runtime_error);
}

TEST(LayeredMesh, CoverageRejectsUnmatchedLayerContacts) {
    const auto basin = configuration({layer(1, 0.0, 1.0), layer(2, 1.0, 2.0)});
    const auto quad = [](double left, double right, double top, double bottom, int id) {
        return Mesh2D::Cell{
            .vertices = {{left, top}, {right, top}, {right, bottom}, {left, bottom}},
            .edge_kinds = {BoundaryKind::top, BoundaryKind::right, BoundaryKind::bottom,
                           BoundaryKind::left},
            .id = id};
    };
    const std::vector cells{quad(0.0, 2.0, 0.0, 1.0, 1), quad(0.0, 1.0, 1.0, 2.0, 2),
                            quad(1.0, 2.0, 1.0, 2.0, 2)};
    const Mesh2D mesh = Mesh2D::from_cells(cells);
    EXPECT_NEAR(mesh.area(), 4.0, 1.0e-12);
    EXPECT_THROW(validate_layered_mesh_coverage(mesh, basin), std::runtime_error);
}

TEST(LayeredMesh, ChoosesLayerRowsForEachConfiguration) {
    const BasinConfiguration thin = configuration({layer(1, 0.0, 1.0)});
    const BasinConfiguration thick = configuration({layer(1, 0.0, 10.0)});
    const std::array configurations{thin, thick};
    const LayeredMeshLayout layout = make_layered_mesh_layout(configurations, 1.0);

    const Mesh2D thin_mesh = make_layered_mesh(thin, layout);
    const Mesh2D thick_mesh = make_layered_mesh(thick, layout);

    EXPECT_EQ(thin_mesh.cells().size(), 2U);
    EXPECT_EQ(thick_mesh.cells().size(), 20U);
}

TEST(LayeredMesh, PreservesMinimumThicknessForEveryRowOfDisappearingLayer) {
    const BasinConfiguration first = configuration({layer(1, 0.0, 4.0)});
    const BasinConfiguration second = configuration({layer(1, 0.0, 0.0)});
    const std::array configurations{first, second};
    const LayeredMeshLayout layout = make_layered_mesh_layout(configurations, 1.0);
    Mesh2D mesh = make_layered_mesh(first, layout);

    EXPECT_THROW(BasinMeshMotion(mesh, first, second, 1.0, 1.0), std::runtime_error);
    BasinMeshMotion motion(mesh, first, second, 1.0, 1.0, true);
    motion.set_position(1.0);

    for (const TQMesh::Facet* cell : mesh.cells()) {
        double maximum_edge = 0.0;
        for (std::size_t edge = 0; edge < cell->n_vertices(); ++edge) {
            maximum_edge =
                std::max(maximum_edge, (cell->vertex((edge + 1) % cell->n_vertices()).xy() -
                                        cell->vertex(edge).xy())
                                           .norm());
        }
        EXPECT_GE(cell->area() / maximum_edge, 0.5 - 1.0e-12);
    }
}

TEST(LayeredMesh, VariableColumnsReduceRowsAndKeepSharedContactsConforming) {
    auto upper = layer(1, 0., 5.);
    upper.x = {0., 1., 2., 3., 4., 5., 6.};
    upper.top.assign(upper.x.size(), 0.);
    upper.bottom = {5., 5., 4., 3., 2., 1., 1.};
    upper.lithotype.assign(upper.x.size(), 1);
    upper.porosity.assign(upper.x.size(), .1);
    auto lower = upper;
    lower.id = 2;
    lower.top = upper.bottom;
    lower.bottom.assign(lower.x.size(), 8.);
    const auto basin = configuration({upper, lower});
    const auto layout = make_layered_mesh_layout({&basin, 1}, 1.);
    const auto old_mesh = make_layered_mesh(basin, layout);
    const auto mesh = make_variable_layered_mesh(basin, layout);

    EXPECT_LT(mesh.cells().size(), old_mesh.cells().size());
    EXPECT_GT(mesh.native().n_triangles(), 0U);
    EXPECT_GT(mesh.native().n_quads(), 0U);
    EXPECT_NEAR(mesh.area(), 48., 1e-10);
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, basin));
}

TEST(LayeredMesh, VariableColumnsResolveTransverseRatherThanVerticalThickness) {
    auto sheet = layer(1, 0., 4.);
    sheet.x = {0., 10.};
    sheet.top = {0., 200.};
    sheet.bottom = {4., 204.};
    const auto first = configuration({sheet});
    sheet.top = {200., 100.};
    sheet.bottom = {204., 104.};
    const auto second = configuration({sheet});
    const std::array configurations{first, second};
    const auto layout = make_layered_mesh_layout(configurations, 1.);
    auto mesh = make_variable_layered_mesh(first, layout);

    EXPECT_EQ(mesh.cells().size(), layout.x.size() - 1);
    EXPECT_EQ(mesh.native().n_triangles(), 0U);
    EXPECT_NEAR(mesh.area(), 40., 1e-10);
    const auto vertices = mesh.native().n_vertices();
    BasinMeshMotion motion(mesh, first, second, 1., 1.);
    for (const double fraction : {.25, .5, 1.}) {
        EXPECT_NO_THROW(motion.set_position(fraction));
        EXPECT_EQ(mesh.native().n_vertices(), vertices);
        EXPECT_NEAR(mesh.area(), 40., 1e-10);
    }
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, second));
}

TEST(LayeredMesh, VariableColumnsPreservePinchesAndLayerSupports) {
    const auto basin = basin_with_nearby_layer_start();
    const auto layout = make_layered_mesh_layout({&basin, 1}, .25);
    const auto mesh = make_variable_layered_mesh(basin, layout);
    EXPECT_NEAR(mesh.area(), 6., 1e-10);
    EXPECT_NO_THROW(validate_layered_mesh_coverage(mesh, basin));

    auto lobes = layer(1, 0., 1.);
    lobes.x = {0., 1., 2., 3., 4.};
    lobes.top = {0., 0., 0., 0., 0.};
    lobes.bottom = {0., 1., 0., 1., 0.};
    lobes.lithotype.assign(5, 1);
    lobes.porosity.assign(5, .1);
    const auto separated = configuration({lobes});
    const auto separated_layout = make_layered_mesh_layout({&separated, 1}, .25);
    const auto separated_mesh = make_variable_layered_mesh(separated, separated_layout);
    EXPECT_NEAR(separated_mesh.area(), 2., 1e-10);
    EXPECT_NO_THROW(validate_layered_mesh_coverage(separated_mesh, separated));
}

} // namespace
