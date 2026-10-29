#include "basin/basin_geometry.hpp"

#include "basin/profile_interpolation.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

double BasinLayerGeometry::top_at(const double coordinate) const {
    return basin_detail::interpolate(x, top, coordinate);
}

double BasinLayerGeometry::bottom_at(const double coordinate) const {
    return basin_detail::interpolate(x, bottom, coordinate);
}

BasinLayerGeometry BasinGeometry::layer(const int id) const {
    const auto found = std::find_if(layers.begin(), layers.end(),
                                    [id](const auto& value) { return value.id == id; });
    if (found == layers.end())
        throw std::out_of_range("Unknown basin geometry layer");
    const auto& top = boundaries.at(found->top_boundary).depth;
    const auto& bottom = boundaries.at(found->bottom_boundary).depth;
    std::vector<double> x{found->x_min, found->x_max};
    for (const auto* curve : {&top, &bottom})
        for (const double p : curve->x) {
            if (p > found->x_min && p < found->x_max)
                x.push_back(p);
        }
    std::sort(x.begin(), x.end());
    x.erase(std::unique(x.begin(), x.end()), x.end());
    BasinLayerGeometry result{.id = id, .x = std::move(x), .top = {}, .bottom = {}};
    for (const double p : result.x) {
        result.top.push_back(top.at(p));
        result.bottom.push_back(bottom.at(p));
    }
    return result;
}

namespace {

struct PreparedLayers {
    double age_ma;
    std::vector<BasinLayerGeometry> layers;
};

BasinGeometry boundary_geometry(const PreparedLayers& prepared) {
    BasinGeometry result{.age_ma = prepared.age_ma, .boundaries = {}, .layers = {}};
    if (prepared.layers.empty())
        return result;
    std::vector<double> x;
    for (const auto& layer : prepared.layers)
        x.insert(x.end(), layer.x.begin(), layer.x.end());
    std::sort(x.begin(), x.end());
    x.erase(std::unique(x.begin(), x.end()), x.end());
    std::vector<std::vector<double>> depths(prepared.layers.size() + 1);
    for (const double p : x) {
        double cursor = prepared.layers.front().top_at(p);
        for (const auto& layer : prepared.layers) {
            if (p >= layer.x.front() && p <= layer.x.back() &&
                layer.bottom_at(p) > layer.top_at(p)) {
                cursor = layer.top_at(p);
                break;
            }
        }
        depths.front().push_back(cursor);
        for (std::size_t i = 0; i < prepared.layers.size(); ++i) {
            const auto& layer = prepared.layers[i];
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
    for (const auto& values : depths)
        result.boundaries.push_back({ScalarProfile{x, values}});
    for (std::size_t i = 0; i < prepared.layers.size(); ++i) {
        const auto& layer = prepared.layers[i];
        result.layers.push_back({layer.id, i, i + 1, layer.x.front(), layer.x.back()});
    }
    return result;
}

} // namespace

BasinGeometry make_basin_geometry(const BasinConfiguration& configuration) {
    PreparedLayers result{.age_ma = configuration.age_ma, .layers = {}};
    std::set<int> identifiers;
    for (const auto& layer : configuration.layers) {
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
        result.layers.push_back({layer.id, layer.x, layer.top, layer.bottom});
    }
    return boundary_geometry(result);
}
