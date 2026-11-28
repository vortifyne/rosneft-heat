#include "basin_1d/inverse/section_observations.hpp"

#include "basin_1d/geometry/basin_section_1d.hpp"

#include <algorithm>
#include <ostream>
#include <stdexcept>

std::vector<BasinObservation>
select_section_observations(const BasinSection1D& section,
                            std::span<const BasinObservation> observations,
                            std::span<const std::string> wells, std::ostream& log) {
    if (wells.empty())
        throw std::invalid_argument("Select at least one well for the 1D fit");

    const auto& contacts = section.configs.back().contacts;
    std::vector<BasinObservation> result;
    std::size_t outside = 0;
    for (const auto& point : observations) {
        if (std::find(wells.begin(), wells.end(), point.well) == wells.end())
            continue;
        if (point.z < contacts.front() || point.z > contacts.back()) {
            ++outside;
            continue;
        }
        result.push_back(point);
        result.back().x = section.x;
    }
    log << "1D: using " << result.size() << " of " << result.size() + outside
        << " observations from selected wells (" << outside << " outside section)\n";
    return result;
}
