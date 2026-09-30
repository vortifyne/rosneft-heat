#include "forward/vtu_output.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

void write_vtu(const Mesh2D& mesh, const std::filesystem::path& path,
               const std::span<const VtuCellField> fields) {
    for (const auto& field : fields) {
        if (field.values.size() != mesh.cells().size()) {
            throw std::invalid_argument("VTU cell field size must match the mesh");
        }
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open VTU output: " + path.string());
    }
    output << std::setprecision(17);
    output << "<?xml version=\"1.0\"?>\n"
              "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
              "<UnstructuredGrid>\n<Piece NumberOfPoints=\""
           << mesh.native().n_vertices() << "\" NumberOfCells=\"" << mesh.cells().size()
           << "\">\n<Points>\n<DataArray type=\"Float64\" NumberOfComponents=\"3\" "
              "format=\"ascii\">\n";
    std::vector<Point2D> vertices = mesh.vertex_coordinates();
    for (const Point2D point : vertices) {
        output << point.x << ' ' << point.z << " 0\n";
    }
    output << "</DataArray>\n</Points>\n<Cells>\n"
              "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const TQMesh::Facet* cell : mesh.cells()) {
        for (std::size_t vertex = 0; vertex < cell->n_vertices(); ++vertex) {
            output << cell->vertex(vertex).index() << ' ';
        }
        output << '\n';
    }
    output << "</DataArray>\n<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    std::size_t offset = 0;
    for (const TQMesh::Facet* cell : mesh.cells()) {
        offset += cell->n_vertices();
        output << offset << '\n';
    }
    output << "</DataArray>\n<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (const TQMesh::Facet* cell : mesh.cells()) {
        output << (cell->n_vertices() == 3 ? 5 : 9) << '\n';
    }
    output << "</DataArray>\n</Cells>\n<CellData>\n";
    for (const auto& field : fields) {
        output << "<DataArray type=\"Float64\" Name=\"" << field.name << "\" format=\"ascii\">\n";
        for (const double value : field.values) {
            output << value << '\n';
        }
        output << "</DataArray>\n";
    }
    output << "</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
}

void write_pvd(const std::filesystem::path& path, const std::span<const PvdEntry> entries) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open PVD output: " + path.string());
    }
    output << std::setprecision(17)
           << "<?xml version=\"1.0\"?>\n"
              "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
              "<Collection>\n";
    for (const auto& entry : entries) {
        output << "<DataSet timestep=\"" << entry.time << "\" group=\"\" part=\"0\" file=\""
               << entry.file.generic_string() << "\"/>\n";
    }
    output << "</Collection>\n</VTKFile>\n";
}
