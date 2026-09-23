#include "mesh/mesh2d.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        if (!field.empty() && field.back() == '\r') {
            field.pop_back();
        }
        fields.push_back(field);
    }
    return fields;
}

BoundaryKind parse_boundary_kind(const int value) {
    switch (value) {
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
        throw std::runtime_error("Unknown boundary kind in region file");
    }
}

std::vector<Mesh2D::Region> read_regions(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open region file: " + path.string());
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Region file is empty");
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    if (line != "region_id,layer_id,x,z,edge_kind") {
        throw std::runtime_error("Unexpected region file header");
    }

    std::vector<Mesh2D::Region> regions;
    std::unordered_map<int, std::size_t> positions;
    std::size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_csv_line(line);
        if (fields.size() != 5) {
            throw std::runtime_error("Invalid field count at line " + std::to_string(line_number));
        }

        const int region_id = std::stoi(fields[0]);
        const int layer_id = std::stoi(fields[1]);
        auto [position, inserted] = positions.emplace(region_id, regions.size());
        if (inserted) {
            regions.push_back({.vertices = {}, .edge_kinds = {}, .id = layer_id});
        }
        Mesh2D::Region& region = regions.at(position->second);
        if (region.id != layer_id) {
            throw std::runtime_error("Layer identifier changes inside one region");
        }
        region.vertices.push_back({std::stod(fields[2]), std::stod(fields[3])});
        region.edge_kinds.push_back(parse_boundary_kind(std::stoi(fields[4])));
    }
    return regions;
}

} // namespace

int main(const int argc, const char* const argv[]) {
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: heat_mesh <regions.csv> <cell-size> <output.vtu> [quad]\n";
        return EXIT_FAILURE;
    }

    try {
        const std::vector<Mesh2D::Region> regions = read_regions(argv[1]);
        const double cell_size = std::stod(argv[2]);
        const bool make_quadrilateral = argc == 5 && std::string(argv[4]) == "quad";

        const auto start = std::chrono::steady_clock::now();
        Mesh2D mesh =
            Mesh2D::generate(regions, {.cell_size = [cell_size](Point2D) { return cell_size; },
                                       .make_quadrilateral = make_quadrilateral,
                                       .diagnostic_vtu = std::filesystem::path(argv[3])});
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        const auto& native = mesh.native();
        std::cout << "regions=" << regions.size() << " vertices=" << native.n_vertices()
                  << " cells=" << native.n_elements() << " triangles=" << native.n_triangles()
                  << " quads=" << native.n_quads()
                  << " interior_faces=" << native.n_interior_edges()
                  << " boundary_faces=" << native.n_boundary_edges() << " area=" << mesh.area()
                  << " seconds=" << seconds << '\n';
    } catch (const std::exception& error) {
        std::cerr << "heat_mesh: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
