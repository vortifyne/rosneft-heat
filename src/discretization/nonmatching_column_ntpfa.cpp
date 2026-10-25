#include "discretization/nonmatching_column_ntpfa.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace {

enum class FaceKind {
    internal,
    dirichlet,
    neumann
};

struct Face {
    FaceKind kind = FaceKind::internal;
    std::size_t cell = 0;
    std::size_t neighbor = 0;
    Point2D center{};
    Point2D normal{};
    double length = 0.0;
};

Point2D subtract(const Point2D a, const Point2D b) {
    return {a.x - b.x, a.z - b.z};
}
Point2D add(const Point2D a, const Point2D b) {
    return {a.x + b.x, a.z + b.z};
}
Point2D scale(const double c, const Point2D a) {
    return {c * a.x, c * a.z};
}
double dot(const Point2D a, const Point2D b) {
    return a.x * b.x + a.z * b.z;
}
double norm(const Point2D a) {
    return std::hypot(a.x, a.z);
}
double cross(const Point2D a, const Point2D b) {
    return a.x * b.z - a.z * b.x;
}

struct Sample {
    Point2D point{};
    bool valid = false;
};

Sample sample(const NonmatchingColumnGrid& grid, const std::span<const double> conductivity,
              const Face& face, const Point2D origin) {
    const double distance = dot(subtract(face.center, origin), face.normal);
    if (!(distance > 0.0)) {
        return {};
    }
    if (face.kind == FaceKind::dirichlet) {
        return {face.center, true};
    }
    if (face.kind == FaceKind::neumann) {
        return {add(origin, scale(distance, face.normal)), true};
    }
    const Point2D neighbor = grid.cells[face.neighbor].center();
    const double other_distance = dot(subtract(neighbor, face.center), face.normal);
    const double current_k = conductivity[face.cell];
    const double other_k = conductivity[face.neighbor];
    const double denominator = current_k * other_distance + other_k * distance;
    if (!(other_distance > 0.0) || !(denominator > 0.0)) {
        return {};
    }
    const double m = other_k * distance / denominator;
    const Point2D harmonic =
        add(add(scale(1.0 - m, origin), scale(m, neighbor)),
            scale(distance * other_distance * (current_k - other_k) / denominator, face.normal));
    return {harmonic, true};
}

struct Cone {
    bool valid = false;
    double first = 0.0;
    double second = 0.0;
    double condition = std::numeric_limits<double>::infinity();
};

Cone positive_cone(const Point2D first, const Point2D second, const Point2D target) {
    const double determinant = cross(first, second);
    const double direction_scale = norm(first) * norm(second);
    if (!(direction_scale > 0.0) || std::abs(determinant) <= 1.0e-12 * direction_scale) {
        return {};
    }
    const double a = cross(target, second) / determinant;
    const double b = cross(first, target) / determinant;
    const double coefficient_scale = std::max({1.0, std::abs(a), std::abs(b)});
    if (a < -1.0e-10 * coefficient_scale || b < -1.0e-10 * coefficient_scale) {
        return {};
    }
    const double trace = dot(first, first) + dot(second, second);
    const double discriminant = std::max(0.0, trace * trace - 4.0 * determinant * determinant);
    const double large_eigenvalue = 0.5 * (trace + std::sqrt(discriminant));
    const double condition = large_eigenvalue * large_eigenvalue / (determinant * determinant);
    return {true, std::max(0.0, a), std::max(0.0, b), condition};
}

void append_failure(ColumnNtpfaGeometryAudit& audit, const Face& face, const char* reason) {
    ++audit.failed_sides;
    if (audit.examples.size() < 12) {
        audit.examples.push_back({face.cell, face.center, reason});
    }
}

// Each side must use the target face and one other face: the auxiliary values
// can then be canceled by the nonlinear weights, leaving two cell unknowns.
void audit_cell_side(ColumnNtpfaGeometryAudit& audit, const NonmatchingColumnGrid& grid,
                     const std::span<const double> conductivity,
                     const std::vector<std::vector<Face>>& incident, const Face& target) {
    const Point2D origin = grid.cells[target.cell].center();
    const Sample primary = sample(grid, conductivity, target, origin);
    if (!primary.valid) {
        append_failure(audit, target, "nonpositive harmonic distance");
        return;
    }
    if (target.kind == FaceKind::internal) {
        const double tangent_offset = target.normal.x != 0.0 ? primary.point.z - target.center.z
                                                             : primary.point.x - target.center.x;
        if (std::abs(tangent_offset) >
            0.5 * target.length + 1.0e-10 * std::max(1.0, target.length)) {
            ++audit.target_harmonic_points_outside_subfaces;
        }
    }
    const Point2D first = subtract(primary.point, origin);
    const Point2D conormal = scale(conductivity[target.cell], target.normal);
    const double selected_vertical_normal = primary.point.z - origin.z >= 0.0 ? -1.0 : 1.0;
    std::optional<Cone> best;
    for (const Face& auxiliary : incident[target.cell]) {
        // Match the actual flux assembly: it uses the horizontal face opposite
        // to the vertical offset of the target harmonic point.
        if (auxiliary.normal.z != selected_vertical_normal) {
            continue;
        }
        const Sample other = sample(grid, conductivity, auxiliary, origin);
        if (!other.valid) {
            continue;
        }
        const Cone candidate = positive_cone(first, subtract(other.point, origin), conormal);
        if (candidate.valid &&
            (!best || candidate.first + candidate.second < best->first + best->second ||
             (candidate.first + candidate.second == best->first + best->second &&
              candidate.condition < best->condition))) {
            best = candidate;
        }
    }
    if (!best) {
        append_failure(audit, target, "no positive target-plus-auxiliary cone");
        return;
    }
    audit.maximum_selected_condition = std::max(audit.maximum_selected_condition, best->condition);
}

void audit_boundary_side(ColumnNtpfaGeometryAudit& audit, const NonmatchingColumnGrid& grid,
                         const std::span<const double> conductivity,
                         const std::vector<std::vector<Face>>& incident, const Face& target) {
    const Point2D origin = target.center;
    const Point2D first = subtract(grid.cells[target.cell].center(), origin);
    const Point2D conormal = scale(-conductivity[target.cell], target.normal);
    const double selected_vertical_normal =
        origin.z < grid.cells[target.cell].center().z ? -1.0 : 1.0;
    std::optional<Cone> best;
    for (const Face& auxiliary : incident[target.cell]) {
        // Boundary reconstruction uses the horizontal face on the same side
        // of the cell center as the vertical Dirichlet segment center.
        if (auxiliary.normal.z != selected_vertical_normal) {
            continue;
        }
        const Sample other = sample(grid, conductivity, auxiliary, origin);
        if (!other.valid) {
            continue;
        }
        const Cone candidate = positive_cone(first, subtract(other.point, origin), conormal);
        if (candidate.valid &&
            (!best || candidate.first + candidate.second < best->first + best->second)) {
            best = candidate;
        }
    }
    if (!best) {
        append_failure(audit, target, "no positive boundary-side cone");
        return;
    }
    audit.maximum_selected_condition = std::max(audit.maximum_selected_condition, best->condition);
}

} // namespace

ColumnNtpfaGeometryAudit audit_column_ntpfa_geometry(const NonmatchingColumnGrid& grid,
                                                     const std::span<const double> conductivity) {
    if (conductivity.size() != grid.cells.size()) {
        throw std::invalid_argument("NTPFA audit conductivity size does not match grid");
    }
    for (const double value : conductivity) {
        if (!std::isfinite(value) || !(value > 0.0)) {
            throw std::invalid_argument("NTPFA audit requires positive finite conductivity");
        }
    }
    std::vector<std::vector<Face>> incident(grid.cells.size());
    const auto append_internal = [&](const std::size_t a, const std::size_t b, const Point2D center,
                                     const Point2D normal, const double length) {
        incident[a].push_back({FaceKind::internal, a, b, center, normal, length});
        incident[b].push_back({FaceKind::internal, b, a, center, scale(-1.0, normal), length});
    };
    const auto append_boundary = [&](const std::size_t cell, const Point2D center,
                                     const Point2D normal, const double length,
                                     const FaceKind kind) {
        incident[cell].push_back({kind, cell, cell, center, normal, length});
    };
    for (const NonmatchingHorizontalFace& face : grid.horizontal_faces) {
        const double x = grid.cells[face.upper_cell].center().x;
        append_internal(face.upper_cell, face.lower_cell, {x, face.z}, {0.0, 1.0}, face.length);
    }
    for (const NonmatchingVerticalSubface& face : grid.vertical_subfaces) {
        append_internal(face.left_cell, face.right_cell, {face.x, face.center_z()}, {1.0, 0.0},
                        face.length());
    }
    for (const std::size_t index : grid.top_cells) {
        const NonmatchingColumnCell& cell = grid.cells[index];
        append_boundary(index, {cell.center().x, cell.z_top}, {0.0, -1.0}, cell.width(),
                        FaceKind::dirichlet);
    }
    for (const std::size_t index : grid.bottom_cells) {
        const NonmatchingColumnCell& cell = grid.cells[index];
        append_boundary(index, {cell.center().x, cell.z_bottom}, {0.0, 1.0}, cell.width(),
                        FaceKind::neumann);
    }
    for (const NonmatchingVerticalBoundaryFace& face : grid.surface_vertical_faces) {
        append_boundary(face.cell, {face.x, face.center_z()}, {face.outward_x, 0.0}, face.length(),
                        FaceKind::dirichlet);
    }
    for (const NonmatchingVerticalBoundaryFace& face : grid.basal_vertical_faces) {
        append_boundary(face.cell, {face.x, face.center_z()}, {face.outward_x, 0.0}, face.length(),
                        FaceKind::neumann);
    }
    for (const std::size_t index : grid.columns.front()) {
        const NonmatchingColumnCell& cell = grid.cells[index];
        append_boundary(index, {cell.x_left, cell.center().z}, {-1.0, 0.0}, cell.height(),
                        FaceKind::neumann);
    }
    for (const std::size_t index : grid.columns.back()) {
        const NonmatchingColumnCell& cell = grid.cells[index];
        append_boundary(index, {cell.x_right, cell.center().z}, {1.0, 0.0}, cell.height(),
                        FaceKind::neumann);
    }
    ColumnNtpfaGeometryAudit audit;
    for (const auto& faces : incident) {
        for (const Face& face : faces) {
            if (face.kind == FaceKind::internal && face.normal.x != 0.0) {
                ++audit.internal_sides;
                audit_cell_side(audit, grid, conductivity, incident, face);
            } else if (face.kind == FaceKind::dirichlet && face.normal.x != 0.0) {
                ++audit.dirichlet_cell_sides;
                audit_cell_side(audit, grid, conductivity, incident, face);
                ++audit.dirichlet_boundary_sides;
                audit_boundary_side(audit, grid, conductivity, incident, face);
            }
        }
    }
    return audit;
}
