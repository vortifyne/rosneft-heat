#include "forward/layered_mesh.hpp"

#include <gtest/gtest.h>
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
            .regions = {},
            .layers = std::move(layers),
            .surface_temperature = {},
            .basal_heat_flux = {}};
}

TEST(LayeredMesh, BuildsConformingQuadrilateralsAcrossLayers) {
    const BasinConfiguration basin = configuration({layer(1, 0.0, 1.0), layer(2, 1.0, 2.0)});
    const LayeredMeshLayout layout = make_layered_mesh_layout({&basin, 1}, 1.0);

    LayeredMesh result = make_layered_mesh(basin, layout);

    EXPECT_EQ(result.mesh.cells().size(), 4U);
    EXPECT_EQ(result.mesh.native().n_quads(), 4U);
    EXPECT_EQ(result.cell_ids.size(), result.mesh.cells().size());
    EXPECT_NEAR(result.mesh.area(), 4.0, 1.0e-12);
    EXPECT_NO_THROW(result.mesh.validate());
}

TEST(LayeredMesh, UsesTriangleFanAtLayerPinch) {
    BasinLayerProfile wedge = layer(7, 0.0, 1.0);
    wedge.bottom = {0.0, 2.0};
    const BasinConfiguration basin = configuration({std::move(wedge)});
    const LayeredMeshLayout layout = make_layered_mesh_layout({&basin, 1}, 1.0);

    LayeredMesh result = make_layered_mesh(basin, layout);

    EXPECT_GT(result.mesh.native().n_triangles(), 0U);
    EXPECT_NEAR(result.mesh.area(), 2.0, 1.0e-12);
    EXPECT_NO_THROW(result.mesh.validate());
}

TEST(LayeredMesh, KeepsMaterialCellIdentifiersAcrossCompatibleConfigurations) {
    const BasinConfiguration first = configuration({layer(1, 0.0, 1.0), layer(2, 1.0, 2.0)});
    const BasinConfiguration second = configuration({layer(1, 0.2, 1.4), layer(2, 1.4, 2.8)});
    const std::vector<BasinConfiguration> history = {first, second};
    const LayeredMeshLayout layout = make_layered_mesh_layout(history, 1.0);

    LayeredMesh old_mesh = make_layered_mesh(first, layout);
    LayeredMesh new_mesh = make_layered_mesh(second, layout);

    EXPECT_EQ(old_mesh.cell_ids, new_mesh.cell_ids);
}

} // namespace
