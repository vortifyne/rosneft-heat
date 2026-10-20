#pragma once

#include "mesh/mesh2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mesh_detail {

using Polygon = std::vector<std::size_t>;
using DirectedEdge = std::pair<std::size_t, std::size_t>;

inline double cross(const Point2D a, const Point2D b, const Point2D c) {
    return (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x);
}

class PolygonPoints {
public:
    explicit PolygonPoints(const double scale)
        : tolerance_(std::max(1e-9, 128. * std::numeric_limits<double>::epsilon() * scale)) {}

    std::size_t insert(const Point2D p) {
        const auto x = static_cast<std::int64_t>(std::floor(p.x / tolerance_));
        const auto z = static_cast<std::int64_t>(std::floor(p.z / tolerance_));
        for (std::int64_t i = -1; i <= 1; ++i) {
            for (std::int64_t j = -1; j <= 1; ++j) {
                const auto found = bins_.find({x + i, z + j});
                if (found == bins_.end()) {
                    continue;
                }
                for (const auto index : found->second) {
                    const auto q = coordinates_[index];
                    if (std::max(std::abs(p.x - q.x), std::abs(p.z - q.z)) <= tolerance_) {
                        return index;
                    }
                }
            }
        }
        const auto index = coordinates_.size();
        coordinates_.push_back(p);
        bins_[{x, z}].push_back(index);
        return index;
    }

    [[nodiscard]] Point2D operator[](const std::size_t i) const {
        return coordinates_[i];
    }
    [[nodiscard]] double tolerance() const {
        return tolerance_;
    }

private:
    double tolerance_;
    std::vector<Point2D> coordinates_;
    std::map<std::pair<std::int64_t, std::int64_t>, Polygon> bins_;
};

inline double polygon_area(const Polygon& polygon, const PolygonPoints& points) {
    double result = 0.;
    const auto origin = points[polygon.front()];
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i) {
        result += cross(origin, points[polygon[i]], points[polygon[i + 1]]);
    }
    return .5 * result;
}

inline Polygon compact_polygon(Polygon polygon, const PolygonPoints& points) {
    polygon.erase(std::unique(polygon.begin(), polygon.end()), polygon.end());
    if (polygon.size() > 1 && polygon.front() == polygon.back()) {
        polygon.pop_back();
    }
    while (polygon.size() > 3) {
        std::size_t selected = polygon.size();
        double best = points.tolerance();
        for (std::size_t i = 0; i < polygon.size(); ++i) {
            const auto a = points[polygon[(i + polygon.size() - 1) % polygon.size()]],
                       b = points[polygon[i]], c = points[polygon[(i + 1) % polygon.size()]];
            const double dot = (b.x - a.x) * (c.x - b.x) + (b.z - a.z) * (c.z - b.z);
            if (dot > 0.) {
                const double error = std::abs(cross(a, b, c)) / std::hypot(c.x - a.x, c.z - a.z);
                if (error <= best && (selected == polygon.size() || error < best ||
                                      polygon[i] < polygon[selected])) {
                    selected = i;
                    best = error;
                }
            }
        }
        if (selected == polygon.size())
            break;
        // Recompute neighbours after EACH removal. Simultaneously deleting a
        // corner and its almost-coincident collinear neighbours can erase a real
        // kink, giving different contours on opposite sides of a shared contact.
        polygon.erase(polygon.begin() + static_cast<std::ptrdiff_t>(selected));
    }
    return polygon;
}

inline bool strictly_convex(const Polygon& polygon, const PolygonPoints& points) {
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const auto a = points[polygon[i]], b = points[polygon[(i + 1) % polygon.size()]],
                   c = points[polygon[(i + 2) % polygon.size()]];
        if (cross(a, b, c) <= points.tolerance() * std::hypot(c.x - a.x, c.z - a.z)) {
            return false;
        }
    }
    return true;
}

// Reject crossings and self-touches, including a vertex lying inside another
// boundary edge. The latter becomes a repeated vertex when shared edges align.
inline bool simple_polygon(const Polygon& polygon, const PolygonPoints& points) {
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const auto a = points[polygon[i]], b = points[polygon[(i + 1) % polygon.size()]];
        const double length = std::hypot(b.x - a.x, b.z - a.z);
        for (std::size_t j = 0; j < polygon.size(); ++j) {
            if (j == i || j == (i + 1) % polygon.size())
                continue;
            const auto c = points[polygon[j]];
            const double fraction =
                ((c.x - a.x) * (b.x - a.x) + (c.z - a.z) * (b.z - a.z)) / (length * length);
            if (fraction >= 0. && fraction <= 1. &&
                std::abs(cross(a, b, c)) <= points.tolerance() * length)
                return false;
            const auto d = points[polygon[(j + 1) % polygon.size()]];
            if (cross(a, b, c) * cross(a, b, d) < 0. && cross(c, d, a) * cross(c, d, b) < 0.)
                return false;
        }
    }
    return true;
}

// Ear clipping keeps supplied boundary vertices, including collinear contact nodes.
// The caller may remove redundant vertices before contact subdivisions are aligned.
inline std::vector<Polygon> triangulate_polygon(Polygon polygon, const PolygonPoints& points) {
    polygon.erase(std::unique(polygon.begin(), polygon.end()), polygon.end());
    if (polygon.size() > 1 && polygon.front() == polygon.back())
        polygon.pop_back();
    std::vector<Polygon> result;
    while (polygon.size() > 3) {
        bool found = false;
        std::size_t selected = 0;
        double best = -1.;
        for (std::size_t i = 0; i < polygon.size(); ++i) {
            const auto prev = (i + polygon.size() - 1) % polygon.size(),
                       next = (i + 1) % polygon.size();
            const auto a = points[polygon[prev]], b = points[polygon[i]], c = points[polygon[next]];
            const double tolerance =
                points.tolerance() *
                std::max({std::hypot(b.x - a.x, b.z - a.z), std::hypot(c.x - b.x, c.z - b.z),
                          std::hypot(a.x - c.x, a.z - c.z)});
            if (cross(a, b, c) <= tolerance) {
                continue;
            }
            const double denominator = (b.x - a.x) * (b.x - a.x) + (b.z - a.z) * (b.z - a.z) +
                                       (c.x - b.x) * (c.x - b.x) + (c.z - b.z) * (c.z - b.z) +
                                       (a.x - c.x) * (a.x - c.x) + (a.z - c.z) * (a.z - c.z);
            const double quality = cross(a, b, c) / denominator;
            if (quality <= best)
                continue;
            bool blocked = false;
            for (std::size_t j = 0; j < polygon.size(); ++j) {
                if (j == prev || j == i || j == next) {
                    continue;
                }
                const auto p = points[polygon[j]];
                if (cross(a, b, p) >= -tolerance && cross(b, c, p) >= -tolerance &&
                    cross(c, a, p) >= -tolerance) {
                    blocked = true;
                    break;
                }
            }
            if (!blocked) {
                found = true;
                selected = i;
                best = quality;
            }
        }
        if (!found) {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "Cannot triangulate polygon (area=" << polygon_area(polygon, points)
                    << ", nodes=" << polygon.size() << "):";
            for (const auto vertex : polygon) {
                const auto p = points[vertex];
                message << " (" << p.x << ',' << p.z << ')';
            }
            throw std::runtime_error(message.str());
        }
        const auto prev = (selected + polygon.size() - 1) % polygon.size(),
                   next = (selected + 1) % polygon.size();
        result.push_back({polygon[prev], polygon[selected], polygon[next]});
        polygon.erase(polygon.begin() + static_cast<std::ptrdiff_t>(selected));
    }
    if (polygon.size() == 3 && polygon_area(polygon, points) > 0.) {
        result.push_back(std::move(polygon));
    } else {
        throw std::runtime_error("Degenerate polygon in strip mesh");
    }
    return result;
}

inline Polygon clip_polygon(const Polygon& polygon, PolygonPoints& points, const int axis,
                            const double bound, const bool keep_greater) {
    Polygon result;
    const auto value = [axis](const Point2D p) { return axis == 0 ? p.x : p.z; };
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const auto a = points[polygon[i]], b = points[polygon[(i + 1) % polygon.size()]];
        const bool inside_a = keep_greater ? value(a) >= bound : value(a) <= bound;
        const bool inside_b = keep_greater ? value(b) >= bound : value(b) <= bound;
        if (inside_a) {
            result.push_back(polygon[i]);
        }
        if (inside_a != inside_b) {
            const double fraction = (bound - value(a)) / (value(b) - value(a));
            Point2D intersection{a.x + fraction * (b.x - a.x), a.z + fraction * (b.z - a.z)};
            if (axis == 0)
                intersection.x = bound;
            else
                intersection.z = bound;
            result.push_back(points.insert(intersection));
        }
    }
    result.erase(std::unique(result.begin(), result.end()), result.end());
    if (result.size() > 1 && result.front() == result.back())
        result.pop_back();
    return result;
}

// Cancel internal directed edges of an exact polygon partition.
inline void add_polygon_edges(std::map<DirectedEdge, int>& edges, const Polygon& polygon) {
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const auto a = polygon[i], b = polygon[(i + 1) % polygon.size()];
        if (a == b)
            continue;
        const auto reverse = edges.find({b, a});
        if (reverse != edges.end())
            edges.erase(reverse);
        else if (!edges.emplace(DirectedEdge{a, b}, 1).second) {
            throw std::runtime_error("Overlapping pieces in strip mesh");
        }
    }
}

inline std::vector<Polygon> boundary_loops(const std::map<DirectedEdge, int>& edges,
                                           const PolygonPoints& points) {
    std::multimap<std::size_t, std::size_t> remaining;
    for (const auto& [edge, count] : edges) {
        (void)count;
        remaining.emplace(edge.first, edge.second);
    }
    std::vector<Polygon> result;
    while (!remaining.empty()) {
        const auto start = remaining.begin()->first;
        auto current = remaining.begin()->second;
        remaining.erase(remaining.begin());
        Polygon loop{start};
        auto previous = start;
        while (current != start) {
            loop.push_back(current);
            const auto [begin, end] = remaining.equal_range(current);
            if (begin == end)
                throw std::runtime_error("Open boundary in raster cell");
            auto selected = begin;
            double best = -std::numeric_limits<double>::infinity();
            const auto a = points[previous], b = points[current];
            for (auto candidate = begin; candidate != end; ++candidate) {
                const auto c = points[candidate->second];
                const double angle = std::atan2(cross(a, b, c), (b.x - a.x) * (c.x - b.x) +
                                                                    (b.z - a.z) * (c.z - b.z));
                if (angle > best) {
                    best = angle;
                    selected = candidate;
                }
            }
            previous = current;
            current = selected->second;
            remaining.erase(selected);
        }
        loop = compact_polygon(std::move(loop), points);
        // Clipping adjacent triangles can leave a closed walk along a grid
        // line with exactly zero enclosed area. It is not a material component.
        if (!loop.empty() && polygon_area(loop, points) == 0.)
            continue;
        if (loop.size() < 3 || !(polygon_area(loop, points) > 0.)) {
            std::ostringstream message;
            message << std::setprecision(17) << "Invalid boundary loop (area="
                    << (loop.empty() ? 0. : polygon_area(loop, points)) << "):";
            for (const auto vertex : loop) {
                const auto p = points[vertex];
                message << " (" << p.x << ',' << p.z << ')';
            }
            throw std::runtime_error(message.str());
        }
        result.push_back(std::move(loop));
    }
    return result;
}

} // namespace mesh_detail
