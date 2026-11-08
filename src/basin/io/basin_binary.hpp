#pragma once

#include <filesystem>

// Native little-endian version-2 snapshot: cell temperatures, horizontal-node I/Ro.
struct BasinSnapshot;

void write_basin_binary(const std::filesystem::path& path, const BasinSnapshot& snapshot);
