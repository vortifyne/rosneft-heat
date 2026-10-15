#include "basin/layered_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

struct CoverageCell {
    const TQMesh::Facet* cell;
    double left;
    double right;
};

std::pair<double, double> section(const TQMesh::Facet& cell, const double x) {
    double top = std::numeric_limits<double>::infinity();
    double bottom = -top;
    const auto include = [&](const double z) {
        top = std::min(top, z);
        bottom = std::max(bottom, z);
    };
    for (std::size_t vertex = 0; vertex < cell.n_vertices(); ++vertex) {
        const auto& first = cell.vertex(vertex).xy();
        const auto& second = cell.vertex((vertex + 1) % cell.n_vertices()).xy();
        if (x < std::min(first.x, second.x) || x > std::max(first.x, second.x)) {
            continue;
        }
        if (first.x == second.x) {
            include(first.y);
            include(second.y);
        } else {
            const double fraction = (x - first.x) / (second.x - first.x);
            include((1.0 - fraction) * first.y + fraction * second.y);
        }
    }
    if (!std::isfinite(top) || !std::isfinite(bottom)) {
        throw std::runtime_error("Cannot intersect mesh cell with coverage section");
    }
    return {top, bottom};
}

void validate_layer(const BasinLayerProfile& layer, const std::vector<const TQMesh::Facet*>& facets,
                    const double tolerance) {
    std::vector<double> breaks = layer.x;
    std::vector<CoverageCell> cells;
    cells.reserve(facets.size());
    for (const auto* cell : facets) {
        double left = std::numeric_limits<double>::infinity();
        double right = -left;
        for (std::size_t vertex = 0; vertex < cell->n_vertices(); ++vertex) {
            const auto& first = cell->vertex(vertex).xy();
            const auto& second = cell->vertex((vertex + 1) % cell->n_vertices()).xy();
            const auto& third = cell->vertex((vertex + 2) % cell->n_vertices()).xy();
            const auto a = second - first;
            const auto b = third - second;
            if (a.x * b.y - a.y * b.x < 0.0) {
                throw std::runtime_error("Layer coverage requires convex mesh cells");
            }
            left = std::min(left, first.x);
            right = std::max(right, first.x);
            // Every change of polygon edge must delimit a sweep interval.
            breaks.push_back(first.x);
        }
        if (left < layer.x.front() - tolerance || right > layer.x.back() + tolerance) {
            throw std::runtime_error("Mesh cell extends beyond its layer profile");
        }
        cells.push_back({cell, left, right});
    }
    std::sort(breaks.begin(), breaks.end());
    breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
    std::vector<std::vector<std::size_t>> entering(breaks.size()), leaving(breaks.size());
    for (std::size_t index = 0; index < cells.size(); ++index) {
        entering[static_cast<std::size_t>(
                     std::lower_bound(breaks.begin(), breaks.end(), cells[index].left) -
                     breaks.begin())]
            .push_back(index);
        leaving[static_cast<std::size_t>(
                    std::lower_bound(breaks.begin(), breaks.end(), cells[index].right) -
                    breaks.begin())]
            .push_back(index);
    }
    std::set<std::size_t> active;
    for (std::size_t interval = 0; interval + 1 < breaks.size(); ++interval) {
        for (const auto index : leaving[interval]) {
            active.erase(index);
        }
        for (const auto index : entering[interval]) {
            active.insert(index);
        }
        const double middle = 0.5 * (breaks[interval] + breaks[interval + 1]);
        std::vector<std::pair<double, std::size_t>> ordered;
        ordered.reserve(active.size());
        for (const auto index : active) {
            const auto [top, bottom] = section(*cells[index].cell, middle);
            ordered.emplace_back(0.5 * (top + bottom), index);
        }
        std::sort(ordered.begin(), ordered.end());
        // Within this strip all section endpoints and input boundaries are affine.
        // Check both ends in the SAME cell order, so crossing cells cannot exchange
        // positions unnoticed. Zero-length sections at vertices are allowed.
        for (const double x : {breaks[interval], breaks[interval + 1]}) {
            double cursor = layer.top_at(x);
            const double expected_bottom = layer.bottom_at(x);
            if (expected_bottom < cursor - tolerance) {
                throw std::runtime_error("Layer profile has negative thickness");
            }
            const auto fail = [&](const char* reason, const double difference) {
                std::ostringstream message;
                message << "Layer " << layer.id << " coverage " << reason << " at x=" << x
                        << " in strip [" << breaks[interval] << ',' << breaks[interval + 1]
                        << "]: distance=" << difference;
                throw std::runtime_error(message.str());
            };
            for (const auto& [position, index] : ordered) {
                (void)position;
                const auto [top, bottom] = section(*cells[index].cell, x);
                if (top > cursor + tolerance) {
                    fail("gap", top - cursor);
                }
                if (top < cursor - tolerance) {
                    fail("overlap or exterior cell", cursor - top);
                }
                cursor = bottom;
            }
            if (std::abs(cursor - expected_bottom) > tolerance) {
                fail("does not reach its boundary", cursor - expected_bottom);
            }
        }
    }
}

} // namespace

void validate_layered_mesh_coverage(const Mesh2D& mesh, const BasinConfiguration& configuration,
                                    const double tolerance) {
    if (!(tolerance > 0.0) || !std::isfinite(tolerance)) {
        throw std::invalid_argument("Coverage tolerance must be finite and positive");
    }
    std::unordered_map<int, std::vector<const TQMesh::Facet*>> by_layer;
    for (const auto* cell : mesh.cells()) {
        (void)configuration.layer(cell->color());
        by_layer[cell->color()].push_back(cell);
    }
    for (const auto& layer : configuration.layers) {
        validate_layer(layer, by_layer[layer.id], tolerance);
    }
    // A complete partition of each layer can still have unmatched contact edges
    // (e.g. one long edge opposite two short edges). They must not become thermal
    // boundaries merely because the cell builder did not merge them.
    for (const auto kind : {BoundaryKind::top, BoundaryKind::bottom}) {
        for (const auto* edge : mesh.boundary_edges(kind)) {
            const auto point = edge->xy();
            double top = std::numeric_limits<double>::infinity();
            double bottom = -top;
            for (const auto& layer : configuration.layers) {
                if (point.x < layer.x.front() || point.x > layer.x.back()) {
                    continue;
                }
                const double upper = layer.top_at(point.x);
                const double lower = layer.bottom_at(point.x);
                if (lower - upper > tolerance) {
                    top = std::min(top, upper);
                    bottom = std::max(bottom, lower);
                }
            }
            const double expected = kind == BoundaryKind::top ? top : bottom;
            if (!std::isfinite(expected) || std::abs(point.y - expected) > tolerance) {
                throw std::runtime_error("Layer contact incorrectly appears as thermal boundary");
            }
        }
    }
}
