#include "basin/basin_geometry.hpp"
#include "basin/basin_input.hpp"
#include "basin/nonmatching_column_grid.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kSnapshotAgeMa = 5.0;

const BasinConfiguration& configuration_at(const BasinInput& input, const double age_ma) {
    const auto configurations = input.configurations();
    const auto found = std::find_if(configurations.begin(), configurations.end(),
                                    [age_ma](const auto& value) { return value.age_ma == age_ma; });
    if (found == configurations.end()) {
        throw std::runtime_error("Requested snapshot age is absent from basin input");
    }
    return *found;
}

double integral(const ScalarProfile& profile, const double left, const double right) {
    std::vector<double> points{left, right};
    for (const double x : profile.x) {
        if (x > left && x < right) {
            points.push_back(x);
        }
    }
    std::sort(points.begin(), points.end());
    double result = 0.0;
    for (std::size_t i = 1; i < points.size(); ++i) {
        const double a = points[i - 1];
        const double b = points[i];
        result += 0.5 * (profile.at(a) + profile.at(b)) * (b - a);
    }
    return result;
}

double quantile(std::vector<double> values, const double probability) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = probability * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    return std::lerp(values[lower], values[upper], position - static_cast<double>(lower));
}

double sum_step_lengths(const std::vector<NonmatchingVerticalBoundaryFace>& faces) {
    double result = 0.0;
    for (const NonmatchingVerticalBoundaryFace& face : faces) {
        result += face.length();
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            throw std::invalid_argument(
                "usage: nonmatching_geometry_audit INPUT GEOMETRY_LABEL HX HZ PHASE");
        }
        const BasinInput input = BasinInput::read(argv[1]);
        const std::string geometry_label = argv[2];
        const double hx = std::stod(argv[3]);
        const double hz = std::stod(argv[4]);
        const double phase = std::stod(argv[5]);
        const BasinGeometry geometry = make_basin_geometry(configuration_at(input, kSnapshotAgeMa));
        const LayeredMeshLayout layout = make_nonmatching_column_layout(geometry, hx, hz, phase);
        const NonmatchingColumnGrid grid = make_nonmatching_column_grid(geometry, layout);

        std::map<int, double> expected_layer_areas;
        for (const BasinLayerRegion& region : geometry.layers) {
            expected_layer_areas[region.id] +=
                integral(geometry.boundaries[region.bottom_boundary].depth, region.x_min,
                         region.x_max) -
                integral(geometry.boundaries[region.top_boundary].depth, region.x_min, region.x_max);
        }
        std::map<int, double> grid_layer_areas;
        std::vector<double> heights;
        std::vector<double> aspect_ratios;
        for (const NonmatchingColumnCell& cell : grid.cells) {
            grid_layer_areas[cell.layer_id] += cell.area();
            heights.push_back(cell.height());
            aspect_ratios.push_back(
                std::max(cell.width() / cell.height(), cell.height() / cell.width()));
        }
        double expected_area = 0.0;
        double maximum_layer_area_relative_defect = 0.0;
        for (const auto& [layer, area] : expected_layer_areas) {
            expected_area += area;
            maximum_layer_area_relative_defect =
                std::max(maximum_layer_area_relative_defect,
                         std::abs(grid_layer_areas[layer] - area) / std::max(1.0, std::abs(area)));
        }

        std::vector<double> widths;
        widths.reserve(grid.columns.size());
        for (std::size_t column = 0; column < grid.columns.size(); ++column) {
            widths.push_back(grid.x[column + 1] - grid.x[column]);
        }
        std::vector<double> subface_lengths;
        std::vector<double> interface_coverage(grid.columns.size() - 1, 0.0);
        std::size_t material_subfaces = 0;
        for (const NonmatchingVerticalSubface& face : grid.vertical_subfaces) {
            subface_lengths.push_back(face.length());
            const auto position = std::lower_bound(grid.x.begin(), grid.x.end(), face.x);
            if (position == grid.x.end() || *position != face.x || position == grid.x.begin() ||
                position + 1 == grid.x.end()) {
                throw std::runtime_error("Vertical subface is not on an internal column line");
            }
            const std::size_t interface = static_cast<std::size_t>(position - grid.x.begin() - 1);
            interface_coverage[interface] += face.length();
            material_subfaces += static_cast<std::size_t>(
                grid.cells[face.left_cell].layer_id != grid.cells[face.right_cell].layer_id);
        }
        double maximum_interface_coverage_defect = 0.0;
        for (std::size_t interface = 0; interface < interface_coverage.size(); ++interface) {
            const auto& left = grid.columns[interface];
            const auto& right = grid.columns[interface + 1];
            const double common_top = std::max(grid.cells[left.front()].z_top,
                                               grid.cells[right.front()].z_top);
            const double common_bottom = std::min(grid.cells[left.back()].z_bottom,
                                                  grid.cells[right.back()].z_bottom);
            maximum_interface_coverage_defect =
                std::max(maximum_interface_coverage_defect,
                         std::abs(interface_coverage[interface] - (common_bottom - common_top)));
        }

        double horizontal_surface_length = 0.0;
        double horizontal_basal_length = 0.0;
        for (const std::size_t cell : grid.top_cells) {
            horizontal_surface_length += grid.cells[cell].width();
        }
        for (const std::size_t cell : grid.bottom_cells) {
            horizontal_basal_length += grid.cells[cell].width();
        }
        const double vertical_surface_length = sum_step_lengths(grid.surface_vertical_faces);
        const double vertical_basal_length = sum_step_lengths(grid.basal_vertical_faces);
        const double left_length = grid.cells[grid.columns.front().back()].z_bottom -
                                   grid.cells[grid.columns.front().front()].z_top;
        const double right_length = grid.cells[grid.columns.back().back()].z_bottom -
                                    grid.cells[grid.columns.back().front()].z_top;

        std::cout << std::setprecision(17)
                  << "geometry,hx,hz,phase,columns,cells,horizontal_faces,vertical_subfaces,"
                     "material_subfaces,surface_vertical_faces,basal_vertical_faces,expected_area,"
                     "grid_area,area_relative_defect,max_layer_area_relative_defect,"
                     "max_interface_coverage_defect,horizontal_surface_length,"
                     "vertical_surface_length,horizontal_basal_length,vertical_basal_length,"
                     "left_length,right_length,dx_min,dx_median,dx_p95,dx_max,dz_min,dz_median,"
                     "dz_p95,dz_max,aspect_median,aspect_p95,aspect_max,subface_min,subface_median,"
                     "subface_p95,subface_max\n"
                  << geometry_label << ',' << hx << ',' << hz << ',' << phase << ','
                  << grid.columns.size() << ',' << grid.cells.size() << ','
                  << grid.horizontal_faces.size() << ',' << grid.vertical_subfaces.size() << ','
                  << material_subfaces << ',' << grid.surface_vertical_faces.size() << ','
                  << grid.basal_vertical_faces.size() << ',' << expected_area << ',' << grid.area()
                  << ',' << std::abs(grid.area() - expected_area) /
                                std::max(1.0, std::abs(expected_area))
                  << ',' << maximum_layer_area_relative_defect << ','
                  << maximum_interface_coverage_defect << ',' << horizontal_surface_length << ','
                  << vertical_surface_length << ',' << horizontal_basal_length << ','
                  << vertical_basal_length << ',' << left_length << ',' << right_length << ','
                  << quantile(widths, 0.0) << ',' << quantile(widths, 0.5) << ','
                  << quantile(widths, 0.95) << ',' << quantile(widths, 1.0) << ','
                  << quantile(heights, 0.0) << ',' << quantile(heights, 0.5) << ','
                  << quantile(heights, 0.95) << ',' << quantile(heights, 1.0) << ','
                  << quantile(aspect_ratios, 0.5) << ',' << quantile(aspect_ratios, 0.95) << ','
                  << quantile(aspect_ratios, 1.0) << ',' << quantile(subface_lengths, 0.0) << ','
                  << quantile(subface_lengths, 0.5) << ',' << quantile(subface_lengths, 0.95) << ','
                  << quantile(subface_lengths, 1.0) << '\n';

        const double area_relative_defect =
            std::abs(grid.area() - expected_area) / std::max(1.0, std::abs(expected_area));
        return area_relative_defect <= 1.0e-11 &&
                       maximum_layer_area_relative_defect <= 1.0e-11 &&
                       maximum_interface_coverage_defect <= 1.0e-8
                   ? 0
                   : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
