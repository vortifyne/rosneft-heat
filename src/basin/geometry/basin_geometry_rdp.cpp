#include "basin/geometry/basin_geometry_rdp.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void retain_rdp(const std::vector<double>& x, const std::vector<double>& z, std::size_t first,
                std::size_t last, double tolerance, std::vector<bool>& retained) {
    if (last <= first + 1)
        return;
    const double dx = x[last] - x[first];
    const double dz = z[last] - z[first];
    const double length = std::hypot(dx, dz);
    double maximum = -1.0;
    std::size_t candidate = first;
    for (std::size_t i = first + 1; i < last; ++i) {
        const double distance = std::abs(dx * (z[i] - z[first]) - dz * (x[i] - x[first])) / length;
        if (distance > maximum) {
            maximum = distance;
            candidate = i;
        }
    }
    if (maximum > tolerance) {
        retained[candidate] = true;
        retain_rdp(x, z, first, candidate, tolerance, retained);
        retain_rdp(x, z, candidate, last, tolerance, retained);
    }
}

std::size_t index_of(const std::vector<double>& x, double coordinate) {
    const auto found = std::lower_bound(x.begin(), x.end(), coordinate);
    if (found == x.end() || *found != coordinate) {
        throw std::invalid_argument("Layer support endpoint is absent from shared axis");
    }
    return static_cast<std::size_t>(found - x.begin());
}

} // namespace

BasinGeometry simplify_basin_geometry_rdp(const BasinGeometry& source, double tolerance) {
    if (!std::isfinite(tolerance) || tolerance < 0.0 || source.boundaries.empty()) {
        throw std::invalid_argument("Invalid RDP geometry or tolerance");
    }
    if (tolerance == 0.0)
        return source;
    const auto& x = source.x;
    const std::size_t count = x.size();
    std::vector<bool> mandatory(count, false);
    mandatory.front() = mandatory.back() = true;
    for (std::size_t layer_index = 0; layer_index < source.layers.size(); ++layer_index) {
        const auto& layer = source.layers[layer_index];
        const std::size_t left = index_of(x, layer.x_min);
        const std::size_t right = index_of(x, layer.x_max);
        mandatory[left] = mandatory[right] = true;
        const auto& top = source.boundaries[layer_index];
        const auto& bottom = source.boundaries[layer_index + 1];
        std::size_t peak = left;
        for (std::size_t i = left; i <= right; ++i) {
            if (bottom[i] - top[i] > bottom[peak] - top[peak])
                peak = i;
            const bool zero = std::abs(bottom[i] - top[i]) <= 1.0e-10;
            const bool left_positive = i > left && bottom[i - 1] - top[i - 1] > 1.0e-10;
            const bool right_positive = i < right && bottom[i + 1] - top[i + 1] > 1.0e-10;
            if (zero && (left_positive || right_positive))
                mandatory[i] = true;
        }
        mandatory[peak] = true;
    }
    std::vector<bool> retained = mandatory;
    std::vector<std::size_t> pins;
    for (std::size_t i = 0; i < count; ++i) {
        if (mandatory[i])
            pins.push_back(i);
    }
    for (const auto& boundary : source.boundaries) {
        for (std::size_t i = 1; i < pins.size(); ++i) {
            retain_rdp(x, boundary, pins[i - 1], pins[i], tolerance, retained);
        }
    }
    // A shared RDP axis preserves exact contacts at retained points. Add
    // offending source knots if linear interpolation changes layer topology.
    for (;;) {
        std::vector<std::size_t> kept;
        for (std::size_t i = 0; i < count; ++i)
            if (retained[i])
                kept.push_back(i);
        bool changed = false;
        for (std::size_t interval = 1; interval < kept.size(); ++interval) {
            const std::size_t left = kept[interval - 1];
            const std::size_t right = kept[interval];
            for (std::size_t i = left + 1; i < right; ++i) {
                const double alpha = (x[i] - x[left]) / (x[right] - x[left]);
                for (std::size_t layer_index = 0; layer_index < source.layers.size();
                     ++layer_index) {
                    const auto& layer = source.layers[layer_index];
                    if (x[i] < layer.x_min || x[i] > layer.x_max)
                        continue;
                    const auto& top = source.boundaries[layer_index];
                    const auto& bottom = source.boundaries[layer_index + 1];
                    const double actual = bottom[i] - top[i];
                    const double prepared =
                        std::lerp(bottom[left] - top[left], bottom[right] - top[right], alpha);
                    if (prepared < -1.0e-8 || (actual > 1.0e-8 && prepared <= 1.0e-8) ||
                        (std::abs(actual) <= 1.0e-10 && std::abs(prepared) > 1.0e-8)) {
                        retained[i] = true;
                        changed = true;
                        break;
                    }
                }
            }
        }
        if (!changed)
            break;
    }
    BasinGeometry result{.x = {},
                         .boundaries = std::vector<std::vector<double>>(source.boundaries.size()),
                         .layers = source.layers};
    for (std::size_t i = 0; i < count; ++i) {
        if (!retained[i])
            continue;
        result.x.push_back(x[i]);
        for (std::size_t boundary = 0; boundary < source.boundaries.size(); ++boundary) {
            result.boundaries[boundary].push_back(source.boundaries[boundary][i]);
        }
    }
    return result;
}
