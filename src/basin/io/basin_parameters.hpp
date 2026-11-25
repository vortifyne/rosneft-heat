#pragma once

#include "basin/inverse/basin_heat_parameter_space.hpp"

#include <filesystem>

[[nodiscard]] BasinHeatParameterSpace read_basin_parameters(const std::filesystem::path& path);
void write_basin_parameters(const std::filesystem::path& path,
                            const BasinHeatParameterSpace& space);
