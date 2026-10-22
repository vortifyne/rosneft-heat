#include "basin/column_epoch_geometry.hpp"
#include "basin/basin_geometry_rdp.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

bool close(double a, double b) {
    return std::abs(a - b) <= 1.0e-9 * std::max({1.0, std::abs(a), std::abs(b)});
}

double profile_mean(const ScalarProfile& profile, double left, double right) {
    std::vector<double> points{left, right};
    for (const double x : profile.x) {
        if (x > left && x < right) {
            points.push_back(x);
        }
    }
    std::sort(points.begin(), points.end());
    double integral = 0.0;
    for (std::size_t i = 1; i < points.size(); ++i) {
        integral += 0.5 * (profile.at(points[i - 1]) + profile.at(points[i])) *
                    (points[i] - points[i - 1]);
    }
    return integral / (right - left);
}

std::vector<double> means(const BasinGeometry& geometry, double left, double right) {
    std::vector<double> result;
    result.reserve(geometry.boundaries.size());
    for (const BasinBoundary& boundary : geometry.boundaries) {
        result.push_back(profile_mean(boundary.depth, left, right));
    }
    return result;
}

} // namespace

ColumnEpochGeometry make_column_epoch_geometry(const BasinConfiguration& first,
                                               const BasinConfiguration& second,
                                               double rdp_tolerance) {
    ColumnEpochGeometry epoch{.start = simplify_basin_geometry_rdp(
                                  make_basin_geometry(first), rdp_tolerance),
                              .after_event = simplify_basin_geometry_rdp(
                                  make_basin_geometry(second), rdp_tolerance),
                              .born_layer_id = std::nullopt};
    if (epoch.start.layers.empty() || epoch.after_event.layers.size() < epoch.start.layers.size() ||
        epoch.after_event.layers.size() > epoch.start.layers.size() + 1) {
        throw std::invalid_argument("Epoch must retain old layers and birth at most one layer");
    }
    const auto& a = epoch.start.boundaries.front().depth.x;
    const auto& b = epoch.after_event.boundaries.front().depth.x;
    if (a.empty() || b.empty() || !close(a.front(), b.front()) ||
        !close(a.back(), b.back())) {
        throw std::invalid_argument("Epoch geometries must have the same horizontal domain");
    }
    const bool birth = epoch.after_event.layers.size() == epoch.start.layers.size() + 1;
    const std::size_t offset = birth ? 1 : 0;
    for (std::size_t layer = 0; layer < epoch.start.layers.size(); ++layer) {
        if (epoch.start.layers[layer].id != epoch.after_event.layers[layer + offset].id) {
            throw std::invalid_argument("Old layer order or identity changed across the epoch");
        }
    }
    if (birth) {
        epoch.born_layer_id = epoch.after_event.layers.front().id;
    }
    return epoch;
}

std::vector<double> make_column_epoch_axis(const ColumnEpochGeometry& epoch,
                                           double horizontal_step,
                                           std::span<const double> required_x) {
    if (!std::isfinite(horizontal_step) || !(horizontal_step > 0.0)) {
        throw std::invalid_argument("Column width must be positive");
    }
    std::vector<double> events;
    for (const BasinGeometry* geometry : {&epoch.start, &epoch.after_event}) {
        for (const BasinBoundary& boundary : geometry->boundaries) {
            events.insert(events.end(), boundary.depth.x.begin(), boundary.depth.x.end());
        }
        for (const BasinLayerRegion& layer : geometry->layers) {
            events.push_back(layer.x_min);
            events.push_back(layer.x_max);
        }
    }
    const double domain_left = epoch.start.boundaries.front().depth.x.front();
    const double domain_right = epoch.start.boundaries.front().depth.x.back();
    for (const double coordinate : required_x) {
        if (!std::isfinite(coordinate) || coordinate < domain_left ||
            coordinate > domain_right) {
            throw std::invalid_argument("Required column coordinate is outside the domain");
        }
        events.push_back(coordinate);
    }
    std::sort(events.begin(), events.end());
    events.erase(std::unique(events.begin(), events.end()), events.end());
    std::vector<double> axis;
    for (std::size_t i = 1; i < events.size(); ++i) {
        const double left = events[i - 1];
        const double right = events[i];
        const auto count = static_cast<std::size_t>(std::ceil((right - left) / horizontal_step));
        for (std::size_t j = 0; j < count; ++j) {
            axis.push_back(std::lerp(left, right, static_cast<double>(j) /
                                                     static_cast<double>(count)));
        }
    }
    axis.push_back(events.back());
    return axis;
}

ColumnEpochSection sample_column_epoch(const ColumnEpochGeometry& epoch,
                                        double x_left, double x_right, double fraction) {
    if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0 ||
        !std::isfinite(x_left) || !std::isfinite(x_right) || !(x_right > x_left)) {
        throw std::invalid_argument("Invalid column interval or epoch fraction");
    }
    const double middle = 0.5 * (x_left + x_right);
    const std::size_t old_count = epoch.start.layers.size();
    const std::size_t offset = epoch.born_layer_id ? 1 : 0;
    for (const BasinGeometry* geometry : {&epoch.start, &epoch.after_event}) {
        for (const BasinLayerRegion& layer : geometry->layers) {
            if ((layer.x_min > x_left && layer.x_min < x_right) ||
                (layer.x_max > x_left && layer.x_max < x_right)) {
                throw std::invalid_argument("Column crosses a layer support endpoint");
            }
        }
    }
    const auto start_depths = means(epoch.start, x_left, x_right);
    const auto after_depths = means(epoch.after_event, x_left, x_right);
    std::vector<double> end_depths(old_count + 1);
    if (offset == 0) {
        end_depths = after_depths;
    } else {
        // The caller splits columns at support endpoints. Give the future
        // new-layer volume to the locally uppermost old layer.
        std::size_t owner = old_count;
        for (std::size_t i = 0; i < old_count; ++i) {
            const auto& region = epoch.after_event.layers[i + 1];
            if (middle >= region.x_min && middle <= region.x_max &&
                after_depths[i + 2] - after_depths[i + 1] > 1.0e-12) {
                owner = i;
                break;
            }
        }
        if (owner == old_count) {
            throw std::invalid_argument("New layer has no old material below it");
        }
        end_depths.front() = after_depths.front();
        for (std::size_t i = 0; i < old_count; ++i) {
            end_depths[i + 1] = i < owner ? after_depths.front() : after_depths[i + 2];
        }
    }
    ColumnEpochSection section{.x_left = x_left, .x_right = x_right,
                               .layer_ids = {}, .boundaries = {}};
    section.layer_ids.reserve(old_count);
    section.boundaries.reserve(old_count + 1);
    for (std::size_t i = 0; i < old_count; ++i) {
        section.layer_ids.push_back(epoch.start.layers[i].id);
    }
    for (std::size_t i = 0; i <= old_count; ++i) {
        section.boundaries.push_back(std::lerp(start_depths[i], end_depths[i], fraction));
        if (i > 0 && section.boundaries[i] < section.boundaries[i - 1] - 1.0e-9) {
            throw std::runtime_error("Epoch column has negative layer thickness");
        }
    }
    return section;
}
