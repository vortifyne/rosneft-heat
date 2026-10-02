#pragma once

#include "mesh/mesh2d.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

struct VtuCellField {
    std::string name;
    std::span<const double> values;
};

struct PvdEntry {
    double time;
    std::filesystem::path file;
};

void write_vtu(const Mesh2D& mesh, const std::filesystem::path& path,
               std::span<const VtuCellField> fields);

void write_pvd(const std::filesystem::path& path, std::span<const PvdEntry> entries);
