#include "basin/layered_mesh.hpp"

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

} // namespace
