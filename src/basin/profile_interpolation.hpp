#pragma once

#include <algorithm>
#include <span>
#include <stdexcept>

namespace basin_detail {

inline double interpolate(const std::span<const double> x, const std::span<const double> values,
                          const double coordinate) {
    if (x.empty() || x.size() != values.size()) {
        throw std::logic_error("Invalid scalar profile");
    }
    if (coordinate <= x.front()) {
        return values.front();
    }
    if (coordinate >= x.back()) {
        return values.back();
    }
    const auto upper = std::upper_bound(x.begin(), x.end(), coordinate);
    const std::size_t right = static_cast<std::size_t>(upper - x.begin());
    const std::size_t left = right - 1;
    const double fraction = (coordinate - x[left]) / (x[right] - x[left]);
    return ((1.0 - fraction) * values[left]) + (fraction * values[right]);
}

} // namespace basin_detail
