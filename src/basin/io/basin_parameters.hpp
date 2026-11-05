#pragma once

#include "basin/basin_problem.hpp"
#include "numerics/algebra/vector.hpp"

#include <filesystem>
#include <span>

[[nodiscard]] Vector read_basin_parameters(const std::filesystem::path& path,
                                           std::span<const BasinConfig> configs);
void write_basin_parameters(const std::filesystem::path& path, std::span<const BasinConfig> configs,
                            const Vector& parameters);
