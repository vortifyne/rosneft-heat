#pragma once

#include <filesystem>
#include <iosfwd>

struct BasinForwardResult;

void write_basin_statistics(std::ostream& output, const BasinForwardResult& result,
                            double program_s, double initialization_s, double input_s);

void write_basin_intervals(const std::filesystem::path& directory,
                           const BasinForwardResult& result);
