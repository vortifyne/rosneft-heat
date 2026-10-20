#include "mesh/strip_mesh.hpp"

#include "mesh/polygon_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace {
using namespace mesh_detail;
using Coordinate = std::pair<double, double>;
using Edge = std::pair<Coordinate, Coordinate>;

std::vector<MeshRegion> align_boundaries(const std::span<const MeshRegion> regions,
                                         const double scale) {
    // Insert all supplied contact vertices BEFORE generating either adjacent mesh.
    // The backend can reuse an existing subdivision but cannot insert a future
    // region's vertex into a cell that has already been generated.
    std::map<double, std::set<double>> points;
    for (const auto& region : regions) {
        for (const auto p : region.boundary) {
            points[p.x].insert(p.z);
        }
    }
    const double tolerance = PolygonPoints(scale).tolerance();
    std::vector<MeshRegion> result;
    result.reserve(regions.size());
    for (const auto& region : regions) {
        MeshRegion aligned{.id = region.id, .boundary = {}, .edge_kinds = {}};
        for (std::size_t i = 0; i < region.boundary.size(); ++i) {
            const auto a = region.boundary[i],
                       b = region.boundary[(i + 1) % region.boundary.size()];
            const double dx = b.x - a.x, dz = b.z - a.z, length = std::hypot(dx, dz);
            std::vector<std::pair<double, Point2D>> division{{0., a}};
            const auto end = points.upper_bound(std::max(a.x, b.x));
            for (auto x = points.lower_bound(std::min(a.x, b.x)); x != end; ++x) {
                const auto end_z = x->second.upper_bound(std::max(a.z, b.z));
                for (auto z = x->second.lower_bound(std::min(a.z, b.z)); z != end_z; ++z) {
                    const double px = x->first - a.x, pz = *z - a.z;
                    const double fraction = (px * dx + pz * dz) / (length * length);
                    if (fraction * length > tolerance && (1. - fraction) * length > tolerance &&
                        std::abs(dx * pz - dz * px) <= tolerance * length) {
                        division.push_back({fraction, {x->first, *z}});
                    }
                }
            }
            std::sort(division.begin(), division.end(),
                      [](const auto& p, const auto& q) { return p.first < q.first; });
            double previous = -1.;
            for (const auto& [fraction, point] : division) {
                if ((fraction - previous) * length > tolerance) {
                    aligned.boundary.push_back(point);
                    aligned.edge_kinds.push_back(region.edge_kinds[i]);
                    previous = fraction;
                }
            }
        }
        result.push_back(std::move(aligned));
    }
    return result;
}

Edge edge_key(const Point2D a, const Point2D b) {
    const Coordinate first{a.x, a.z}, second{b.x, b.z};
    return first < second ? Edge{first, second} : Edge{second, first};
}

double area(const std::span<const Point2D> vertices) {
    double result = 0.;
    const auto origin = vertices.front();
    for (std::size_t i = 1; i + 1 < vertices.size(); ++i) {
        const auto a = vertices[i], b = vertices[i + 1];
        result += (a.x - origin.x) * (b.z - origin.z) - (a.z - origin.z) * (b.x - origin.x);
    }
    return .5 * result;
}

double triangle_quality(const Mesh2D::Cell& cell) {
    double squared_length = 0.;
    for (std::size_t i = 0; i < 3; ++i) {
        const auto a = cell.vertices[i], b = cell.vertices[(i + 1) % 3];
        squared_length =
            std::max(squared_length, (a.x - b.x) * (a.x - b.x) + (a.z - b.z) * (a.z - b.z));
    }
    return 2. * area(cell.vertices) / squared_length;
}

void improve_triangle_diagonals(std::vector<Mesh2D::Cell>& cells) {
    // Ear clipping can leave a boundary-only needle. Flip its interior diagonal
    // when the two triangles form a convex quadrilateral and both improve as a pair.
    // All outer edges and their material/boundary labels remain unchanged.
    for (int sweep = 0; sweep < 12; ++sweep) {
        std::map<Edge, std::vector<std::size_t>> edges;
        for (std::size_t i = 0; i < cells.size(); ++i) {
            if (cells[i].vertices.size() != 3)
                continue;
            for (std::size_t j = 0; j < 3; ++j)
                edges[edge_key(cells[i].vertices[j], cells[i].vertices[(j + 1) % 3])].push_back(i);
        }
        struct Candidate {
            std::size_t first, second;
            Mesh2D::Cell a, b;
            double old_quality;
        };
        std::vector<Candidate> candidates;
        for (const auto& [edge, adjacent] : edges) {
            if (adjacent.size() != 2)
                continue;
            const auto& first = cells[adjacent[0]];
            const auto& second = cells[adjacent[1]];
            if (first.id != second.id)
                continue;
            const Point2D a{edge.first.first, edge.first.second},
                b{edge.second.first, edge.second.second};
            const auto opposite = [&](const Mesh2D::Cell& cell) {
                for (const auto p : cell.vertices)
                    if (Coordinate{p.x, p.z} != edge.first && Coordinate{p.x, p.z} != edge.second)
                        return p;
                throw std::runtime_error("Triangle has no opposite vertex");
            };
            const auto c = opposite(first), d = opposite(second);
            if (!(cross(a, b, c) * cross(a, b, d) < 0. && cross(c, d, a) * cross(c, d, b) < 0.))
                continue;
            const auto triangle = [&](Point2D p, Point2D q, Point2D r) {
                if (cross(p, q, r) < 0.)
                    std::swap(q, r);
                Mesh2D::Cell result{.vertices = {p, q, r}, .edge_kinds = {}, .id = first.id};
                for (std::size_t j = 0; j < 3; ++j) {
                    const auto key = edge_key(result.vertices[j], result.vertices[(j + 1) % 3]);
                    BoundaryKind kind = BoundaryKind::interface;
                    for (const auto* old : {&first, &second})
                        for (std::size_t k = 0; k < 3; ++k)
                            if (key == edge_key(old->vertices[k], old->vertices[(k + 1) % 3]))
                                kind = old->edge_kinds[k];
                    result.edge_kinds.push_back(kind);
                }
                return result;
            };
            auto one = triangle(c, d, a), two = triangle(d, c, b);
            const double old_quality = std::min(triangle_quality(first), triangle_quality(second));
            const double quality = std::min(triangle_quality(one), triangle_quality(two));
            if (quality > 1.01 * old_quality + 1e-12)
                candidates.push_back(
                    {adjacent[0], adjacent[1], std::move(one), std::move(two), old_quality});
        }
        if (candidates.empty())
            break;
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            if (a.old_quality != b.old_quality)
                return a.old_quality < b.old_quality;
            return std::pair{a.first, a.second} < std::pair{b.first, b.second};
        });
        std::vector<bool> used(cells.size());
        for (auto& candidate : candidates) {
            if (used[candidate.first] || used[candidate.second])
                continue;
            cells[candidate.first] = std::move(candidate.a);
            cells[candidate.second] = std::move(candidate.b);
            used[candidate.first] = used[candidate.second] = true;
        }
    }
}

std::vector<Mesh2D::Cell> merge_convex_pairs(std::vector<Mesh2D::Cell> cells) {
    std::map<Edge, std::vector<std::size_t>> edges;
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const auto& cell = cells[i];
        if (cell.vertices.size() != 3) {
            continue;
        }
        for (std::size_t j = 0; j < 3; ++j) {
            edges[edge_key(cell.vertices[j], cell.vertices[(j + 1) % 3])].push_back(i);
        }
    }
    struct Candidate {
        std::size_t first, second;
        Mesh2D::Cell quad;
        double elongation;
    };
    std::vector<Candidate> candidates;
    for (const auto& [edge, adjacent] : edges) {
        (void)edge;
        if (adjacent.size() != 2) {
            continue;
        }
        const auto& first = cells[adjacent[0]];
        const auto& second = cells[adjacent[1]];
        if (first.id != second.id) {
            continue;
        }
        std::map<Coordinate, Point2D> unique;
        for (const auto* cell : {&first, &second}) {
            for (const auto p : cell->vertices) {
                unique[{p.x, p.z}] = p;
            }
        }
        if (unique.size() != 4) {
            continue;
        }
        Point2D center{};
        Mesh2D::Cell quad{.vertices = {}, .edge_kinds = {}, .id = first.id};
        for (const auto& [key, point] : unique) {
            (void)key;
            center.x += .25 * point.x;
            center.z += .25 * point.z;
            quad.vertices.push_back(point);
        }
        std::sort(quad.vertices.begin(), quad.vertices.end(), [center](const auto a, const auto b) {
            return std::atan2(a.z - center.z, a.x - center.x) <
                   std::atan2(b.z - center.z, b.x - center.x);
        });
        bool convex = true;
        double longest = 0.;
        for (std::size_t i = 0; i < 4; ++i) {
            const auto a = quad.vertices[i], b = quad.vertices[(i + 1) % 4],
                       c = quad.vertices[(i + 2) % 4];
            const double ux = b.x - a.x, uz = b.z - a.z, vx = c.x - b.x, vz = c.z - b.z;
            const double first_length = std::hypot(ux, uz), second_length = std::hypot(vx, vz);
            longest = std::max(longest, first_length);
            // Keep angles between 15 and 165 degrees. Do not inherit the unchecked
            // triangle-to-quad merging of the backend, which can produce concave cells.
            if (!(ux * vz - uz * vx > .25881904510252074 * first_length * second_length)) {
                convex = false;
            }
            BoundaryKind kind = BoundaryKind::interface;
            for (const auto* cell : {&first, &second}) {
                for (std::size_t j = 0; j < 3; ++j) {
                    if (edge_key(a, b) ==
                        edge_key(cell->vertices[j], cell->vertices[(j + 1) % 3])) {
                        kind = cell->edge_kinds[j];
                    }
                }
            }
            quad.edge_kinds.push_back(kind);
        }
        const double quad_area = area(quad.vertices);
        const double elongation = longest * longest / quad_area;
        if (!convex || elongation > 4. ||
            std::abs(quad_area - area(first.vertices) - area(second.vertices)) > 1e-8 * quad_area) {
            continue;
        }
        candidates.push_back({adjacent[0], adjacent[1], std::move(quad), elongation});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.elongation != b.elongation) {
            return a.elongation < b.elongation;
        }
        return std::pair{a.first, a.second} < std::pair{b.first, b.second};
    });
    std::vector<bool> used(cells.size());
    std::vector<Mesh2D::Cell> result;
    result.reserve(cells.size());
    for (auto& candidate : candidates) {
        if (!used[candidate.first] && !used[candidate.second]) {
            used[candidate.first] = used[candidate.second] = true;
            result.push_back(std::move(candidate.quad));
        }
    }
    for (std::size_t i = 0; i < cells.size(); ++i) {
        if (!used[i]) {
            result.push_back(std::move(cells[i]));
        }
    }
    return result;
}

struct Patch {
    Polygon vertices;
    std::size_t region;
    double background_area;
    double background_width;
    bool active = true;
};

struct RasterGrid {
    double z_max, step;
    std::vector<std::vector<double>> row_x;
};

void align_patch_edges(std::vector<Patch>& patches, PolygonPoints& points, const double scale) {
    std::vector<MeshRegion> polygons;
    for (const auto& patch : patches) {
        MeshRegion polygon{.id = 0, .boundary = {}, .edge_kinds = {}};
        for (const auto vertex : patch.vertices)
            polygon.boundary.push_back(points[vertex]);
        polygon.edge_kinds.assign(polygon.boundary.size(), BoundaryKind::interface);
        polygons.push_back(std::move(polygon));
    }
    polygons = align_boundaries(polygons, scale);
    for (std::size_t i = 0; i < patches.size(); ++i) {
        patches[i].vertices.clear();
        for (const auto p : polygons[i].boundary)
            patches[i].vertices.push_back(points.insert(p));
    }
}

RasterGrid make_raster_grid(const std::span<const MeshRegion> regions, PolygonPoints& points,
                            const double x_min, const double x_max, const double z_min,
                            const double z_max, const double step) {
    const int nx = std::max(1, static_cast<int>(std::ceil((x_max - x_min) / step)));
    const int nz = std::max(1, static_cast<int>(std::ceil((z_max - z_min) / step)));
    std::vector<std::set<double>> rows(static_cast<std::size_t>(nz));
    for (auto& row : rows)
        for (int column = 0; column <= nx; ++column)
            row.insert(x_min + column * step);
    // A contour corner splits only the background row through that corner. Both
    // materials see the SAME row subdivision. Straight contour subdivisions do
    // not request refinement, so another horizon's nodes cannot add false bends.
    for (const auto& region : regions) {
        Polygon polygon;
        for (const auto p : region.boundary)
            polygon.push_back(points.insert(p));
        polygon = compact_polygon(std::move(polygon), points);
        for (const auto vertex : polygon) {
            const auto p = points[vertex];
            const double position = (z_max - p.z) / step;
            const int first = static_cast<int>(std::floor(position));
            for (const auto row : {first - 1, first}) {
                if (row >= 0 && row < nz && p.z >= z_max - (row + 1) * step - points.tolerance() &&
                    p.z <= z_max - row * step + points.tolerance())
                    rows[static_cast<std::size_t>(row)].insert(p.x);
            }
        }
    }
    RasterGrid result{z_max, step, {}};
    for (const auto& row : rows)
        result.row_x.emplace_back(row.begin(), row.end());
    return result;
}

void fit_boundary_vertices(std::vector<Patch>& patches, const std::span<const MeshRegion> regions,
                           PolygonPoints& points, const double cell_size) {
    // Collapse short contour subdivisions, or move an interior background vertex
    // vertically to an existing contour intersection in the same column. The
    // contour itself and the area of each material component remain unchanged.
    std::set<std::size_t> fixed;
    std::vector<DirectedEdge> segments;
    for (const auto& region : regions) {
        Polygon polygon;
        for (const auto p : region.boundary)
            polygon.push_back(points.insert(p));
        polygon = compact_polygon(std::move(polygon), points);
        for (std::size_t i = 0; i < polygon.size(); ++i) {
            fixed.insert(polygon[i]);
            segments.emplace_back(polygon[i], polygon[(i + 1) % polygon.size()]);
        }
    }
    std::map<std::size_t, std::set<std::size_t>> incident;
    std::map<double, std::map<double, std::size_t>> used;
    for (std::size_t i = 0; i < patches.size(); ++i)
        for (const auto vertex : patches[i].vertices) {
            incident[vertex].insert(i);
            const auto p = points[vertex];
            used[p.x][p.z] = vertex;
        }
    struct Candidate {
        double distance;
        std::size_t first, second;
        bool keep_second = false;
    };
    std::vector<Candidate> candidates;
    std::set<std::size_t> boundary_nodes = fixed;
    for (const auto& [first, last] : segments) {
        const auto a = points[first], b = points[last];
        const double dx = b.x - a.x, dz = b.z - a.z, length = std::hypot(dx, dz);
        std::vector<std::pair<double, std::size_t>> nodes{{0., first}, {1., last}};
        const auto end = used.upper_bound(std::max(a.x, b.x) + points.tolerance());
        for (auto x = used.lower_bound(std::min(a.x, b.x) - points.tolerance()); x != end; ++x) {
            const auto end_z = x->second.upper_bound(std::max(a.z, b.z) + points.tolerance());
            for (auto z = x->second.lower_bound(std::min(a.z, b.z) - points.tolerance());
                 z != end_z; ++z) {
                const auto p = points[z->second];
                const double fraction = ((p.x - a.x) * dx + (p.z - a.z) * dz) / (length * length);
                if (fraction > 0. && fraction < 1. &&
                    std::abs(cross(a, b, p)) <= points.tolerance() * length)
                    nodes.emplace_back(fraction, z->second);
            }
        }
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        for (const auto& [fraction, vertex] : nodes) {
            (void)fraction;
            boundary_nodes.insert(vertex);
        }
        for (std::size_t i = 1; i < nodes.size(); ++i) {
            const auto p = points[nodes[i - 1].second], q = points[nodes[i].second];
            const double distance = std::hypot(q.x - p.x, q.z - p.z);
            if (distance < .15 * cell_size)
                candidates.push_back({distance, nodes[i - 1].second, nodes[i].second});
        }
    }
    for (const auto boundary : boundary_nodes) {
        const auto target = points[boundary];
        const auto end_x = used.upper_bound(target.x + points.tolerance());
        for (auto x = used.lower_bound(target.x - points.tolerance()); x != end_x; ++x) {
            const auto end_z = x->second.upper_bound(target.z + .15 * cell_size);
            for (auto z = x->second.lower_bound(target.z - .15 * cell_size); z != end_z; ++z) {
                if (boundary_nodes.contains(z->second))
                    continue;
                candidates.push_back({std::abs(z->first - target.z), z->second, boundary, true});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.distance != b.distance)
            return a.distance < b.distance;
        return std::pair{a.first, a.second} < std::pair{b.first, b.second};
    });
    std::map<std::size_t, std::size_t> replacement;
    const auto root = [&](std::size_t v) {
        while (replacement.contains(v))
            v = replacement.at(v);
        return v;
    };
    for (const auto& candidate : candidates) {
        auto from = root(candidate.first), to = root(candidate.second);
        if (from == to)
            continue;
        if (candidate.keep_second) {
            if (boundary_nodes.contains(from) ||
                std::abs(points[from].x - points[to].x) > points.tolerance())
                continue;
        } else {
            if (fixed.contains(from) && fixed.contains(to))
                continue;
            if (fixed.contains(from) || (!fixed.contains(to) && from < to))
                std::swap(from, to);
        }
        const auto a = points[from], b = points[to];
        if (std::hypot(b.x - a.x, b.z - a.z) > .15 * cell_size)
            continue;
        std::map<std::size_t, Polygon> modified;
        std::map<std::size_t, double> area_change;
        double perimeter = 0.;
        bool valid = true;
        for (const auto index : incident[from]) {
            const auto& patch = patches[index];
            Polygon polygon = patch.vertices;
            std::replace(polygon.begin(), polygon.end(), from, to);
            polygon.erase(std::unique(polygon.begin(), polygon.end()), polygon.end());
            if (polygon.size() > 1 && polygon.front() == polygon.back())
                polygon.pop_back();
            const std::set<std::size_t> unique(polygon.begin(), polygon.end());
            if (unique.size() != polygon.size()) {
                valid = false;
                break;
            }
            const double before = polygon_area(patch.vertices, points);
            const double after = polygon.size() < 3 ? 0. : polygon_area(polygon, points);
            if (after < 0. || (after > 0. && !simple_polygon(polygon, points))) {
                valid = false;
                break;
            }
            area_change[patch.region] += after - before;
            for (std::size_t i = 0; i < patch.vertices.size(); ++i) {
                const auto p = points[patch.vertices[i]],
                           q = points[patch.vertices[(i + 1) % patch.vertices.size()]];
                perimeter += std::hypot(q.x - p.x, q.z - p.z);
            }
            modified.emplace(index, std::move(polygon));
        }
        for (const auto& [region, change] : area_change) {
            (void)region;
            if (std::abs(change) > 4. * points.tolerance() * perimeter)
                valid = false;
        }
        if (!valid)
            continue;
        for (auto& [index, polygon] : modified) {
            auto& patch = patches[index];
            for (const auto vertex : patch.vertices)
                incident[vertex].erase(index);
            patch.active = polygon.size() >= 3 && polygon_area(polygon, points) > 0.;
            patch.vertices = std::move(polygon);
            if (patch.active)
                for (const auto vertex : patch.vertices)
                    incident[vertex].insert(index);
        }
        replacement[from] = to;
    }
}

double longest_edge(const Polygon& polygon, const PolygonPoints& points) {
    double longest = 0.;
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const auto a = points[polygon[i]], b = points[polygon[(i + 1) % polygon.size()]];
        longest = std::max(longest, std::hypot(b.x - a.x, b.z - a.z));
    }
    return longest;
}

void absorb_narrow_patches(std::vector<Patch>& patches, const PolygonPoints& points) {
    // Remove grid cuts, not material boundaries. A narrow cut cell becomes part of
    // its same-material neighbour; the original contour is retained exactly.
    for (int sweep = 0; sweep < 4; ++sweep) {
        std::map<DirectedEdge, std::vector<std::size_t>> adjacent;
        for (std::size_t i = 0; i < patches.size(); ++i) {
            if (!patches[i].active)
                continue;
            const auto& p = patches[i].vertices;
            for (std::size_t j = 0; j < p.size(); ++j) {
                const auto a = p[j], b = p[(j + 1) % p.size()];
                adjacent[std::minmax(a, b)].push_back(i);
            }
        }
        bool changed = false;
        for (std::size_t i = 0; i < patches.size(); ++i) {
            auto& patch = patches[i];
            if (!patch.active)
                continue;
            const double area = polygon_area(patch.vertices, points);
            if (area >= .25 * patch.background_area &&
                area / longest_edge(patch.vertices, points) >= .15 * patch.background_width)
                continue;
            std::map<std::size_t, double> candidates;
            for (std::size_t j = 0; j < patch.vertices.size(); ++j) {
                const auto a = patch.vertices[j],
                           b = patch.vertices[(j + 1) % patch.vertices.size()];
                const auto found = adjacent.find(std::minmax(a, b));
                if (found == adjacent.end())
                    continue;
                for (const auto other : found->second) {
                    if (other == i || !patches[other].active ||
                        patches[other].region != patch.region)
                        continue;
                    const auto p = points[a], q = points[b];
                    candidates[other] += std::hypot(q.x - p.x, q.z - p.z);
                }
            }
            std::vector<std::pair<double, std::size_t>> ordered;
            for (const auto& [other, length] : candidates)
                ordered.emplace_back(-length, other);
            std::sort(ordered.begin(), ordered.end());
            for (const auto& [length, other] : ordered) {
                (void)length;
                const double combined = area + polygon_area(patches[other].vertices, points);
                const double background_area =
                    std::max(patch.background_area, patches[other].background_area);
                if (combined > 2.5 * background_area)
                    continue;
                std::map<DirectedEdge, int> edges;
                add_polygon_edges(edges, patch.vertices);
                add_polygon_edges(edges, patches[other].vertices);
                const auto loops = boundary_loops(edges, points);
                if (loops.size() != 1 ||
                    std::abs(polygon_area(loops.front(), points) - combined) > 1e-8 * combined)
                    continue;
                // Two patches can share an edge and also touch elsewhere. Their
                // union then has a pinched boundary, unsuitable for a single cell.
                const auto& loop = loops.front();
                const std::set<std::size_t> unique(loop.begin(), loop.end());
                if (unique.size() != loop.size() || !simple_polygon(loop, points))
                    continue;
                patch.vertices = loops.front();
                patch.background_area = background_area;
                patch.background_width =
                    std::max(patch.background_width, patches[other].background_width);
                patches[other].active = false;
                changed = true;
                break;
            }
        }
        if (!changed)
            break;
    }
}

void rasterize_region(const MeshRegion& region, const std::size_t index, PolygonPoints& points,
                      const RasterGrid& grid, std::vector<Patch>& patches) {
    Polygon polygon;
    double xmin = std::numeric_limits<double>::infinity(), xmax = -xmin, zmin = xmin, zmax = -xmin;
    for (const auto p : region.boundary) {
        polygon.push_back(points.insert(p));
        xmin = std::min(xmin, p.x);
        xmax = std::max(xmax, p.x);
        zmin = std::min(zmin, p.z);
        zmax = std::max(zmax, p.z);
    }
    polygon = compact_polygon(std::move(polygon), points);
    const auto triangles = triangulate_polygon(polygon, points);
    const int nz = static_cast<int>(grid.row_x.size());
    const double dz = grid.step;
    xmin = grid.row_x.front().front();
    zmax = grid.z_max;
    std::map<std::pair<int, int>, std::map<DirectedEdge, int>> rows;
    for (const auto& triangle : triangles) {
        double left = xmax, right = xmin, top = zmax, bottom = zmin;
        for (const auto vertex : triangle) {
            const auto p = points[vertex];
            left = std::min(left, p.x);
            right = std::max(right, p.x);
            top = std::min(top, p.z);
            bottom = std::max(bottom, p.z);
        }
        const int first_z =
            std::clamp(static_cast<int>(std::floor((zmax - bottom) / dz)), 0, nz - 1);
        const int last_z = std::clamp(static_cast<int>(std::floor((zmax - top) / dz)), 0, nz - 1);
        for (int row = first_z; row <= last_z; ++row) {
            const auto& axis = grid.row_x[static_cast<std::size_t>(row)];
            const int nx = static_cast<int>(axis.size()) - 1;
            const int first_x = std::clamp(
                static_cast<int>(std::upper_bound(axis.begin(), axis.end(), left) - axis.begin()) -
                    1,
                0, nx - 1);
            const int last_x = std::clamp(
                static_cast<int>(std::upper_bound(axis.begin(), axis.end(), right) - axis.begin()) -
                    1,
                0, nx - 1);
            for (int column = first_x; column <= last_x; ++column) {
                auto clipped =
                    clip_polygon(triangle, points, 0, axis[static_cast<std::size_t>(column)], true);
                clipped = clip_polygon(clipped, points, 0,
                                       axis[static_cast<std::size_t>(column + 1)], false);
                clipped = clip_polygon(clipped, points, 1, zmax - (row + 1) * dz, true);
                clipped = clip_polygon(clipped, points, 1, zmax - row * dz, false);
                if (clipped.size() < 3 ||
                    polygon_area(clipped, points) <= points.tolerance() * points.tolerance())
                    continue;
                add_polygon_edges(rows[{row, column}], clipped);
            }
        }
    }
    for (const auto& [cell, edges] : rows) {
        const auto& axis = grid.row_x[static_cast<std::size_t>(cell.first)];
        const double width = axis[static_cast<std::size_t>(cell.second + 1)] -
                             axis[static_cast<std::size_t>(cell.second)];
        for (auto& loop : boundary_loops(edges, points))
            patches.push_back({std::move(loop), index, width * dz, std::min(width, dz), true});
    }
}

BoundaryKind boundary_kind(const Point2D a, const Point2D b, const MeshRegion& region,
                           const double tolerance) {
    for (std::size_t i = 0; i < region.boundary.size(); ++i) {
        const auto p = region.boundary[i], q = region.boundary[(i + 1) % region.boundary.size()];
        const double length = std::hypot(q.x - p.x, q.z - p.z);
        const auto on_segment = [&](const Point2D x) {
            const double projection = (x.x - p.x) * (q.x - p.x) + (x.z - p.z) * (q.z - p.z);
            return std::abs(cross(p, q, x)) <= tolerance * length &&
                   projection >= -tolerance * length && projection <= length * (length + tolerance);
        };
        const Point2D midpoint{.5 * (a.x + b.x), .5 * (a.z + b.z)};
        if (on_segment(midpoint) && std::abs(cross(p, q, a)) <= tolerance * length &&
            std::abs(cross(p, q, b)) <= tolerance * length)
            return region.edge_kinds[i];
    }
    return BoundaryKind::interface;
}

std::vector<Polygon> triangulate_cell(const Polygon& polygon, PolygonPoints& points) {
    auto diagonals = triangulate_polygon(polygon, points);
    const auto quality = [&](const Point2D a, const Point2D b, const Point2D c) {
        const auto squared = [](Point2D p, Point2D q) {
            return (p.x - q.x) * (p.x - q.x) + (p.z - q.z) * (p.z - q.z);
        };
        return cross(a, b, c) / std::max({squared(a, b), squared(b, c), squared(c, a)});
    };
    double diagonal_quality = std::numeric_limits<double>::infinity();
    for (const auto& triangle : diagonals)
        diagonal_quality =
            std::min(diagonal_quality,
                     quality(points[triangle[0]], points[triangle[1]], points[triangle[2]]));

    // An area centroid gives a simple interior point candidate. It is usable only
    // if it sees EVERY edge; concave cut cells do not automatically satisfy this.
    const auto origin = points[polygon.front()];
    Point2D offset{};
    double twice_area = 0.;
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const auto a = points[polygon[i]], b = points[polygon[(i + 1) % polygon.size()]];
        const double weight = cross(origin, a, b);
        twice_area += weight;
        offset.x += weight * ((a.x - origin.x) + (b.x - origin.x));
        offset.z += weight * ((a.z - origin.z) + (b.z - origin.z));
    }
    const Point2D center{origin.x + offset.x / (3. * twice_area),
                         origin.z + offset.z / (3. * twice_area)};
    double fan_quality = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const auto a = points[polygon[i]], b = points[polygon[(i + 1) % polygon.size()]];
        if (cross(a, b, center) <= points.tolerance() * std::hypot(b.x - a.x, b.z - a.z))
            return diagonals;
        fan_quality = std::min(fan_quality, quality(a, b, center));
    }
    if (fan_quality <= diagonal_quality)
        return diagonals;
    const auto middle = points.insert(center);
    std::vector<Polygon> fan;
    for (std::size_t i = 0; i < polygon.size(); ++i)
        fan.push_back({polygon[i], polygon[(i + 1) % polygon.size()], middle});
    return fan;
}

} // namespace

Mesh2D make_strip_mesh(const std::span<const MeshRegion> regions, const double cell_size) {
    if (regions.empty() || !std::isfinite(cell_size) || !(cell_size > 0.)) {
        throw std::invalid_argument("Strip meshing requires regions and a positive cell size");
    }
    double scale = 1.;
    double x_min = std::numeric_limits<double>::infinity(), x_max = -x_min, z_min = x_min,
           z_max = -x_min;
    for (const auto& region : regions) {
        if (region.boundary.size() < 3 || region.boundary.size() != region.edge_kinds.size() ||
            !(area(region.boundary) > 0.))
            throw std::invalid_argument("Invalid strip region polygon");
        for (std::size_t i = 0; i < region.boundary.size(); ++i) {
            const auto p = region.boundary[i],
                       q = region.boundary[(i + 1) % region.boundary.size()];
            if (!std::isfinite(p.x) || !std::isfinite(p.z) || (p.x == q.x && p.z == q.z)) {
                throw std::invalid_argument("Invalid strip region vertex");
            }
            scale = std::max({scale, std::abs(p.x), std::abs(p.z)});
            x_min = std::min(x_min, p.x);
            x_max = std::max(x_max, p.x);
            z_min = std::min(z_min, p.z);
            z_max = std::max(z_max, p.z);
        }
    }
    PolygonPoints points(scale);
    // One sheet of graph paper for all layers: a shared contour receives the same
    // grid intersections from either side, including when layer extents differ.
    x_min = std::floor(x_min / cell_size) * cell_size;
    z_max = std::ceil(z_max / cell_size) * cell_size;
    for (const auto& region : regions)
        for (const auto p : region.boundary)
            (void)points.insert(p);
    const auto grid = make_raster_grid(regions, points, x_min, x_max, z_min, z_max, cell_size);
    std::vector<Patch> patches;
    for (std::size_t i = 0; i < regions.size(); ++i)
        rasterize_region(regions[i], i, points, grid, patches);
    // Every incident patch must include the vertex before it is moved, including
    // the coarse side of an interface between differently refined background rows.
    align_patch_edges(patches, points, scale);
    fit_boundary_vertices(patches, regions, points, cell_size);
    absorb_narrow_patches(patches, points);
    std::vector<MeshRegion> polygons;
    for (const auto& patch : patches) {
        if (!patch.active)
            continue;
        MeshRegion polygon{.id = static_cast<int>(patch.region), .boundary = {}, .edge_kinds = {}};
        for (const auto vertex : patch.vertices)
            polygon.boundary.push_back(points[vertex]);
        polygon.edge_kinds.assign(polygon.boundary.size(), BoundaryKind::interface);
        polygons.push_back(std::move(polygon));
    }
    // Local merging can remove grid edges. Split both sides of each remaining
    // shared edge at the union of their vertices before constructing final cells.
    polygons = align_boundaries(polygons, scale);
    std::vector<Mesh2D::Cell> cells;
    for (const auto& polygon : polygons) {
        Polygon vertices;
        for (const auto p : polygon.boundary)
            vertices.push_back(points.insert(p));
        std::vector<Polygon> elements;
        if (vertices.size() <= 4 && strictly_convex(vertices, points))
            elements.push_back(vertices);
        else
            elements = triangulate_cell(vertices, points);
        const auto& region = regions[static_cast<std::size_t>(polygon.id)];
        for (const auto& element : elements) {
            Mesh2D::Cell cell{.vertices = {}, .edge_kinds = {}, .id = region.id};
            for (std::size_t j = 0; j < element.size(); ++j) {
                const auto a = points[element[j]], b = points[element[(j + 1) % element.size()]];
                cell.vertices.push_back(a);
                cell.edge_kinds.push_back(boundary_kind(a, b, region, 4. * points.tolerance()));
            }
            cells.push_back(std::move(cell));
        }
    }
    improve_triangle_diagonals(cells);
    return Mesh2D::from_cells(merge_convex_pairs(std::move(cells)));
}
