#include "basin/basin_problem.hpp"

#include "basin/geometry/profile_interpolation.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

double ScalarProfile::at(const double coordinate) const {
    return basin_detail::interpolate(x, values, coordinate);
}

double BasinLayerProfile::top_at(const double coordinate) const {
    return basin_detail::interpolate(x, top, coordinate);
}

double BasinLayerProfile::bottom_at(const double coordinate) const {
    return basin_detail::interpolate(x, bottom, coordinate);
}

double BasinLayerProfile::phi_at(const double coordinate) const {
    return basin_detail::interpolate(x, phi, coordinate);
}

int BasinLayerProfile::lithotype_at(const double coordinate) const {
    if (x.empty() || x.size() != lithotype.size()) {
        throw std::logic_error("Invalid lithotype profile");
    }
    const auto found = std::lower_bound(x.begin(), x.end(), coordinate);
    if (found == x.begin()) {
        return lithotype.front();
    }
    if (found == x.end()) {
        return lithotype.back();
    }
    const std::size_t right = static_cast<std::size_t>(found - x.begin());
    const std::size_t left = right - 1;
    return coordinate - x[left] <= x[right] - coordinate ? lithotype[left] : lithotype[right];
}

const BasinLayerProfile& BasinConfig::layer(const int id) const {
    const auto found = std::find_if(layers.begin(), layers.end(),
                                    [id](const auto& value) { return value.id == id; });
    if (found == layers.end()) {
        throw std::out_of_range("Layer is absent from basin configuration");
    }
    return *found;
}
