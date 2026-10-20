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

std::set<double> arc_length_nodes(const ScalarProfile& source, const double step) {
    std::set<double> nodes{source.x.front(), source.x.back()};
    for (double x = std::ceil(source.x.front() / step) * step; x < source.x.back(); x += step)
        nodes.insert(x);
    // Arc-length quantiles alone can straddle a narrow peak and shorten the
    // contour substantially. Retain source turning points before interpolation.
    for (std::size_t i = 1; i + 1 < source.x.size(); ++i) {
        const double left = source.values[i] - source.values[i - 1];
        const double right = source.values[i + 1] - source.values[i];
        if ((left < 0. && right > 0.) || (left > 0. && right < 0.))
            nodes.insert(source.x[i]);
    }
    const std::vector<double> coarse(nodes.begin(), nodes.end());
    for (std::size_t interval = 1; interval < coarse.size(); ++interval) {
        const double left = coarse[interval - 1], right = coarse[interval];
        std::vector<double> x{left};
        const auto end = std::lower_bound(source.x.begin(), source.x.end(), right);
        x.insert(x.end(), std::upper_bound(source.x.begin(), source.x.end(), left), end);
        x.push_back(right);
        // Integral of rho=sqrt(1+z'^2): exact arc length on the input polyline.
        std::vector<double> w{0.};
        for (std::size_t j = 1; j < x.size(); ++j)
            w.push_back(w.back() +
                        std::hypot(x[j] - x[j - 1], source.at(x[j]) - source.at(x[j - 1])));
        const double desired_count = w.back() / step;
        if (desired_count > std::numeric_limits<int>::max())
            throw std::invalid_argument("Requested contour step produces too many nodes");
        // Nearest integer avoids splitting almost horizontal segments just because
        // their arc is infinitesimally longer than step. Arc steps are <=1.5*step.
        const auto count = static_cast<std::size_t>(std::max(1LL, std::llround(desired_count)));
        for (std::size_t j = 1; j < count; ++j) {
            const double target = w.back() * static_cast<double>(j) / static_cast<double>(count);
            const auto k =
                static_cast<std::size_t>(std::upper_bound(w.begin(), w.end(), target) - w.begin());
            const double fraction = (target - w[k - 1]) / (w[k] - w[k - 1]);
            nodes.insert(x[k - 1] + fraction * (x[k] - x[k - 1]));
        }
    }
    return nodes;
}

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

BasinGeometry make_basin_grid_geometry(const BasinConfiguration& configuration,
                                       const double cell_size) {
    if (!std::isfinite(cell_size) || !(cell_size > 0.))
        throw std::invalid_argument("Grid geometry requires a positive step");
    auto exact = make_basin_geometry(configuration);
    if (exact.layers.empty())
        return exact;

    const auto& source_x = exact.boundaries.front().depth.x;
    std::vector<std::set<double>> nodes;
    for (const auto& boundary : exact.boundaries)
        nodes.push_back(arc_length_nodes(boundary.depth, cell_size));
    // Preserve external thermal boundaries exactly. Their length and position
    // enter the imposed basal flux and surface temperature directly.
    for (const auto boundary : {std::size_t{0}, exact.boundaries.size() - 1})
        nodes[boundary].insert(source_x.begin(), source_x.end());
    for (const auto& region : exact.layers) {
        const auto& top = exact.boundaries[region.top_boundary].depth;
        const auto& bottom = exact.boundaries[region.bottom_boundary].depth;
        for (const auto b : {region.top_boundary, region.bottom_boundary}) {
            nodes[b].insert(region.x_min);
            nodes[b].insert(region.x_max);
        }
        for (std::size_t j = 0; j < source_x.size(); ++j) {
            if (source_x[j] < region.x_min || source_x[j] > region.x_max)
                continue;
            const auto positive = [&](std::size_t i) { return bottom.values[i] > top.values[i]; };
            if (!positive(j) &&
                ((j > 0 && positive(j - 1)) || (j + 1 < source_x.size() && positive(j + 1)))) {
                nodes[region.top_boundary].insert(source_x[j]);
                nodes[region.bottom_boundary].insert(source_x[j]);
            }
        }
    }
    const auto sample = [&](std::size_t b) {
        ScalarProfile profile;
        for (const double x : nodes[b]) {
            profile.x.push_back(x);
            profile.values.push_back(exact.boundaries[b].depth.at(x));
        }
        return profile;
    };
    const auto refine = [&](std::size_t b, double x) {
        const auto right = nodes[b].upper_bound(x);
        if (right == nodes[b].end() || right == nodes[b].begin())
            throw std::runtime_error("Cannot refine contour outside its support");
        // Keep material present and ordered without imposing a shape tolerance.
        // The mesher inserts this x into the local background row as well.
        return nodes[b].insert(x).second;
    };

    // Arc-length adaptation is geometry-driven, without an approximation tolerance.
    // Subsequent insertions only preserve material topology and shared contacts.
    for (;;) {
        bool changed = false;
        // A zero-thickness material region identifies two horizons as the same
        // physical contact. Synchronize only those coincident intervals.
        for (const auto& region : exact.layers) {
            const auto t = region.top_boundary, b = region.bottom_boundary;
            const auto& top = exact.boundaries[t].depth.values;
            const auto& bottom = exact.boundaries[b].depth.values;
            for (std::size_t j = 1; j < source_x.size(); ++j) {
                // Outside the source support the material is absent too. Its two
                // canonical horizons still represent one physical contact there.
                if (bottom[j - 1] != top[j - 1] || bottom[j] != top[j])
                    continue;
                std::set<double> common;
                for (const auto boundary : {t, b}) {
                    const auto end = nodes[boundary].upper_bound(source_x[j]);
                    common.insert(nodes[boundary].lower_bound(source_x[j - 1]), end);
                }
                for (const double x : common) {
                    changed = nodes[t].insert(x).second || changed;
                    changed = nodes[b].insert(x).second || changed;
                }
            }
        }
        std::vector<ScalarProfile> profiles;
        for (std::size_t b = 0; b < nodes.size(); ++b)
            profiles.push_back(sample(b));
        for (const auto& region : exact.layers) {
            const auto t = region.top_boundary, b = region.bottom_boundary;
            std::set<double> check(source_x.begin(), source_x.end());
            check.insert(nodes[t].begin(), nodes[t].end());
            check.insert(nodes[b].begin(), nodes[b].end());
            for (const double x : check) {
                if (x < region.x_min || x > region.x_max)
                    continue;
                const double thickness =
                    exact.boundaries[b].depth.at(x) - exact.boundaries[t].depth.at(x);
                const double prepared = profiles[b].at(x) - profiles[t].at(x);
                if ((thickness > 0. && prepared <= 0.) || (thickness == 0. && prepared != 0.)) {
                    // Refine the horizon responsible for the thickness error. A
                    // straight opposite horizon must not inherit its subdivisions.
                    const double top_error =
                        std::abs(profiles[t].at(x) - exact.boundaries[t].depth.at(x));
                    const double bottom_error =
                        std::abs(profiles[b].at(x) - exact.boundaries[b].depth.at(x));
                    const auto boundary = top_error > bottom_error ? t : b;
                    if (!nodes[boundary].contains(x))
                        changed = refine(boundary, x) || changed;
                }
            }
        }
        if (!changed)
            break;
    }
    BasinGeometry result{.age_ma = exact.age_ma, .boundaries = {}, .layers = exact.layers};
    for (std::size_t b = 0; b < nodes.size(); ++b)
        result.boundaries.push_back({sample(b)});
    return result;
}
