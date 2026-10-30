#include "basin/geometry/basin_geometry.hpp"

#include "basin/basin_problem.hpp"
#include "basin/geometry/profile_interpolation.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

double BasinGeometry::boundary_at(std::size_t contact, double coordinate) const {
    return basin_detail::interpolate(x, z[contact], coordinate);
}

std::vector<double> BasinGeometry::mean_boundaries(double left, double right) const {
    std::vector<double> points{left, right};
    for (const double coordinate : x) {
        if (coordinate > left && coordinate < right)
            points.push_back(coordinate);
    }
    std::sort(points.begin(), points.end());
    std::vector<double> result(z.size(), 0.0);
    for (std::size_t contact = 0; contact < z.size(); ++contact) {
        for (std::size_t i = 1; i < points.size(); ++i) {
            result[contact] +=
                0.5 * (boundary_at(contact, points[i - 1]) + boundary_at(contact, points[i])) *
                (points[i] - points[i - 1]);
        }
        result[contact] /= right - left;
    }
    return result;
}

BasinGeometry make_basin_geometry(const BasinConfig& cfg) {
    std::set<int> identifiers;
    for (const auto& layer : cfg.layers) {
        if (layer.x.size() < 2 || layer.top.size() != layer.x.size() ||
            layer.bottom.size() != layer.x.size() || !identifiers.insert(layer.id).second) {
            throw std::invalid_argument("Invalid basin geometry profile");
        }
        for (std::size_t i = 0; i < layer.x.size(); ++i) {
            if (!std::isfinite(layer.x[i]) || !std::isfinite(layer.top[i]) ||
                !std::isfinite(layer.bottom[i]) || layer.bottom[i] < layer.top[i] ||
                (i > 0 && !(layer.x[i] > layer.x[i - 1]))) {
                throw std::invalid_argument(
                    "Basin geometry requires ordered, non-negative profiles");
            }
        }
    }
    BasinGeometry result;
    if (cfg.layers.empty())
        return result;
    auto& x = result.x;
    for (const auto& layer : cfg.layers)
        x.insert(x.end(), layer.x.begin(), layer.x.end());
    std::sort(x.begin(), x.end());
    x.erase(std::unique(x.begin(), x.end()), x.end());
    std::vector<std::vector<double>> depths(cfg.layers.size() + 1);
    for (const double p : x) {
        double cursor = cfg.layers.front().top_at(p);
        for (const auto& layer : cfg.layers) {
            if (p >= layer.x.front() && p <= layer.x.back() &&
                layer.bottom_at(p) > layer.top_at(p)) {
                cursor = layer.top_at(p);
                break;
            }
        }
        depths.front().push_back(cursor);
        for (std::size_t i = 0; i < cfg.layers.size(); ++i) {
            const auto& layer = cfg.layers[i];
            if (p >= layer.x.front() && p <= layer.x.back() &&
                layer.bottom_at(p) > layer.top_at(p)) {
                if (std::abs(layer.top_at(p) - cursor) > 1e-7)
                    throw std::runtime_error("Layer boundaries do not form a continuous section");
                cursor = layer.bottom_at(p);
            }
            // An absent layer has coincident boundaries at the physical contact.
            // Its unused source zero-thickness curve does not define another surface.
            depths[i + 1].push_back(cursor);
        }
    }
    result.z = std::move(depths);
    for (std::size_t i = 0; i < cfg.layers.size(); ++i) {
        const auto& layer = cfg.layers[i];
        result.layers.push_back({layer.id, layer.x.front(), layer.x.back()});
    }
    return result;
}

std::vector<double> make_column_axis(std::span<const double> knots, double dx,
                                     std::span<const double> required_x) {
    if (knots.size() < 2 || !std::isfinite(dx) || !(dx > 0.0)) {
        throw std::invalid_argument("Column axis requires knots and a positive step");
    }
    std::vector<double> events(knots.begin(), knots.end());
    const auto [left, right] = std::minmax_element(knots.begin(), knots.end());
    for (const double coordinate : required_x) {
        if (!std::isfinite(coordinate) || coordinate < *left || coordinate > *right)
            throw std::invalid_argument("Required column coordinate is outside the domain");
        events.push_back(coordinate);
    }
    std::sort(events.begin(), events.end());
    events.erase(std::unique(events.begin(), events.end()), events.end());
    std::vector<double> axis;
    for (std::size_t i = 1; i < events.size(); ++i) {
        const auto count = static_cast<std::size_t>(std::ceil((events[i] - events[i - 1]) / dx));
        axis.push_back(events[i - 1]);
        for (std::size_t j = 1; j < count; ++j)
            axis.push_back(std::lerp(events[i - 1], events[i],
                                     static_cast<double>(j) / static_cast<double>(count)));
    }
    axis.push_back(events.back());
    return axis;
}
