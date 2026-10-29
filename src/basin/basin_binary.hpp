#pragma once

#include "basin/basin_forward_solver.hpp"

#include <filesystem>

// Native little-endian version-1 column-grid snapshot; fields are stored as arrays.
void write_basin_binary(const std::filesystem::path& path, const BasinForwardSnapshot& snapshot);
