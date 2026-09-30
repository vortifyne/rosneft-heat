#include "forward/mesh_transfer.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {

Mesh2D make_mesh(const double size, const int region_id = 1) {
    const Mesh2D::Region region = {
        .vertices = {{0.0, 0.0}, {2.0, 0.0}, {2.0, 1.0}, {0.0, 1.0}},
        .edge_kinds = {BoundaryKind::top, BoundaryKind::right, BoundaryKind::bottom,
                       BoundaryKind::left},
        .id = region_id,
    };
    return Mesh2D::generate(std::span<const Mesh2D::Region>(&region, 1),
                            {.cell_size = [size](Point2D) { return size; },
                             .region_cell_size = {},
                             .diagnostic_vtu = std::nullopt});
}

TEST(MeshTransfer, UsesPhysicalPositionWhenRegionIdentifierChanges) {
    const Mesh2D source = make_mesh(0.35, 1);
    const Mesh2D target = make_mesh(0.25, 2);
    const std::vector<double> source_values(source.cells().size(), 321.0);
    const CellTransferMap map = make_cell_transfer_map(source, target);

    EXPECT_TRUE(std::all_of(map.donor_cells.begin(), map.donor_cells.end(),
                            [](const auto& donor) { return donor.has_value(); }));
    const std::vector<double> transferred =
        transfer_cell_field(source, source_values, map, [](std::size_t, Point2D) -> double {
            throw std::runtime_error("unexpected missing donor");
        });
    EXPECT_TRUE(std::all_of(transferred.begin(), transferred.end(),
                            [](const double value) { return value == 321.0; }));
}

} // namespace

TEST(MeshTransfer, PreservesConstantFieldBetweenDifferentMeshes) {
    const Mesh2D source = make_mesh(0.35);
    const Mesh2D target = make_mesh(0.25);
    const std::vector<double> source_values(source.cells().size(), 321.0);
    const CellTransferMap map = make_cell_transfer_map(source, target);

    const std::vector<double> transferred =
        transfer_cell_field(source, source_values, map, [](std::size_t, Point2D) -> double {
            throw std::runtime_error("unexpected missing donor");
        });

    for (const double value : transferred) {
        EXPECT_DOUBLE_EQ(value, 321.0);
    }
}

TEST(MeshTransfer, SamplesMeshAfterCoordinatesMove) {
    Mesh2D mesh = make_mesh(0.35);
    std::vector<Point2D> coordinates = mesh.vertex_coordinates();
    for (Point2D& point : coordinates) {
        point.z += 3.0;
    }
    mesh.set_vertex_coordinates(coordinates);
    const std::vector<double> values(mesh.cells().size(), 321.0);
    const auto& center = mesh.cells().front()->xy();

    const std::optional<double> sampled = sample_cell_field(mesh, values, {center.x, center.y});

    ASSERT_TRUE(sampled.has_value());
    EXPECT_DOUBLE_EQ(*sampled, 321.0);
}

TEST(MeshTransfer, MarksPointsOutsideOldPhysicalArea) {
    const Mesh2D source = make_mesh(0.35);
    const Mesh2D::Region larger_region = {
        .vertices = {{0.0, -0.5}, {2.0, -0.5}, {2.0, 1.0}, {0.0, 1.0}},
        .edge_kinds = {BoundaryKind::top, BoundaryKind::right, BoundaryKind::bottom,
                       BoundaryKind::left},
        .id = 1,
    };
    const Mesh2D target = Mesh2D::generate(std::span<const Mesh2D::Region>(&larger_region, 1),
                                           {.cell_size = [](Point2D) { return 0.25; },
                                            .region_cell_size = {},
                                            .diagnostic_vtu = std::nullopt});
    const CellTransferMap map = make_cell_transfer_map(source, target);

    EXPECT_TRUE(std::any_of(map.donor_cells.begin(), map.donor_cells.end(),
                            [](const auto& donor) { return !donor.has_value(); }));

    const std::vector<double> source_values(source.cells().size(), 321.0);
    const std::vector<double> transferred = transfer_cell_field(
        source, source_values, map, [](std::size_t, Point2D) { return 273.15; });
    for (std::size_t index = 0; index < transferred.size(); ++index) {
        EXPECT_DOUBLE_EQ(transferred[index], map.donor_cells[index] ? 321.0 : 273.15);
    }
}

TEST(MeshTransfer, ImprovesLinearFieldOverPiecewiseConstantTransfer) {
    const Mesh2D source = make_mesh(0.35);
    const Mesh2D target = make_mesh(0.25);
    std::vector<double> source_values(source.cells().size());
    for (const TQMesh::Facet* cell : source.cells()) {
        source_values[static_cast<std::size_t>(cell->index())] =
            2.0 + (3.0 * cell->xy().x) - (4.0 * cell->xy().y);
    }
    const CellTransferMap map = make_cell_transfer_map(source, target);
    const std::vector<double> transferred =
        transfer_cell_field(source, source_values, map, [](std::size_t, Point2D) -> double {
            throw std::runtime_error("missing donor");
        });
    double linear_error = 0.0;
    double constant_error = 0.0;
    const auto [minimum, maximum] = std::minmax_element(source_values.begin(), source_values.end());
    for (std::size_t index = 0; index < transferred.size(); ++index) {
        const Point2D point = map.target_points[index];
        const double exact = 2.0 + (3.0 * point.x) - (4.0 * point.z);
        linear_error += std::abs(transferred[index] - exact);
        constant_error += std::abs(source_values[*map.donor_cells[index]] - exact);
        EXPECT_GE(transferred[index], *minimum);
        EXPECT_LE(transferred[index], *maximum);
    }
    EXPECT_LT(linear_error, constant_error);
}
