#pragma once

#include "basin/column_basin_solver.hpp"

#include <filesystem>

// Native little-endian version-1 C-grid snapshot; fields are stored as arrays.
void write_column_basin_binary(const std::filesystem::path& path,
                               const ColumnBasinSnapshot& snapshot);
