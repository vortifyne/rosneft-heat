#pragma once

#include <filesystem>

// Native little-endian version-1 column-grid snapshot; fields are stored as arrays.
struct BasinSnapshot;

void write_basin_binary(const std::filesystem::path& path, const BasinSnapshot& snapshot);
