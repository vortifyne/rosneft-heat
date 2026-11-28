#pragma once

#include "basin/inverse/basin_heat_observation_data.hpp"

#include <iosfwd>

struct BasinSection1D;

// An empty result lets the caller skip the 1D fit and keep its constant initial flux.
[[nodiscard]] std::vector<BasinObservation>
select_section_observations(const BasinSection1D& section,
                            std::span<const BasinObservation> observations,
                            std::span<const std::string> wells, std::ostream& log);
