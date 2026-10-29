#pragma once

#include "basin/basin_forward_solver.hpp"

#include <iosfwd>
#include <string_view>

void set_basin_forward_option(BasinForwardSettings& settings, std::string_view option,
                              std::string_view value);
void write_basin_forward_settings(std::ostream& output, const BasinForwardSettings& settings);
