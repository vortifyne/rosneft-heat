#include "discretization/multipoint_flux.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

Eigen::Vector2d vector(const CppUtils::Vec2d& value) {
    return {value.x, value.y};
}

Point2D point(const Eigen::Vector2d& value) {
    return {.x = value.x(), .z = value.y()};
}

void add_term(MultipointFluxExpression& expression, const std::size_t cell,
              const double coefficient) {
    if (coefficient == 0.0) {
        return;
    }
    const auto found = std::find_if(expression.terms.begin(), expression.terms.end(),
                                    [cell](const auto& term) { return term.cell == cell; });
    if (found == expression.terms.end()) {
        expression.terms.push_back({cell, coefficient});
    } else {
        found->coefficient += coefficient;
    }
}

void add_scaled(MultipointFluxExpression& target, const MultipointFluxExpression& source,
                const double scale) {
    for (const MultipointFluxTerm term : source.terms) {
        add_term(target, term.cell, scale * term.coefficient);
    }
    target.constant += scale * source.constant;
}

MultipointFluxExpression scaled(MultipointFluxExpression expression, const double scale) {
    for (MultipointFluxTerm& term : expression.terms) {
        term.coefficient *= scale;
    }
    expression.constant *= scale;
    return expression;
}

MultipointFluxExpression difference(const std::size_t cell,
                                    const MultipointFluxExpression& sample) {
    MultipointFluxExpression result;
    add_term(result, cell, 1.0);
    add_scaled(result, sample, -1.0);
    return result;
}

std::size_t stencil_size(const MultipointFluxExpression& expression) {
    return static_cast<std::size_t>(
        std::count_if(expression.terms.begin(), expression.terms.end(),
                      [](const auto& term) { return std::abs(term.coefficient) > 1.0e-14; }));
}

} // namespace

struct MultipointFlux::Sample {
    Eigen::Vector2d displacement = Eigen::Vector2d::Zero();
    MultipointFluxExpression value;
    Eigen::Vector2d outward_normal = Eigen::Vector2d::Zero();
    double neighbor_weight = 0.0;
};

struct MultipointFlux::OneSidedFlux {
    MultipointFluxExpression expression;
    double direct_coefficient = 0.0;
};

double MultipointFluxExpression::value(const std::span<const double> solution) const {
    double result = constant;
    for (const MultipointFluxTerm term : terms) {
        result += term.coefficient * solution[term.cell];
    }
    return result;
}

MultipointFlux::MultipointFlux(const Mesh2D& mesh) {
    const std::size_t cell_count = mesh.cells().size();
    cell_centers_.resize(cell_count);
    incident_faces_.resize(cell_count);
    for (const TQMesh::Facet* cell : mesh.cells()) {
        cell_centers_[static_cast<std::size_t>(cell->index())] = vector(cell->xy());
    }

    internal_faces_.reserve(mesh.internal_edges().size());
    for (const TQMesh::Edge* edge : mesh.internal_edges()) {
        const TQMesh::Facet& owner = mesh.owner(*edge);
        const TQMesh::Facet& neighbor = mesh.neighbor(*edge);
        const std::size_t owner_index = static_cast<std::size_t>(owner.index());
        const std::size_t neighbor_index = static_cast<std::size_t>(neighbor.index());
        const Eigen::Vector2d center = vector(edge->xy());
        const Eigen::Vector2d normal = vector(mesh.normal_from_owner(*edge));
        const double owner_distance = normal.dot(center - cell_centers_[owner_index]);
        const double neighbor_distance = normal.dot(cell_centers_[neighbor_index] - center);
        if (!(owner_distance > 0.0) || !(neighbor_distance > 0.0)) {
            throw std::runtime_error("Invalid internal face for multipoint flux");
        }
        const std::size_t index = internal_faces_.size();
        internal_faces_.push_back({owner_index, neighbor_index, center, normal, edge->length(),
                                   owner_distance, neighbor_distance});
        incident_faces_[owner_index].push_back({true, index, true});
        incident_faces_[neighbor_index].push_back({true, index, false});
    }

    const auto append_boundaries = [this, &mesh](const BoundaryKind kind, const BoundarySide side) {
        for (const TQMesh::Edge* edge : mesh.boundary_edges(kind)) {
            const TQMesh::Facet& cell = mesh.owner(*edge);
            const std::size_t cell_index = static_cast<std::size_t>(cell.index());
            const std::size_t index = boundary_faces_.size();
            boundary_faces_.push_back({cell_index, vector(edge->xy()),
                                       vector(mesh.normal_from_owner(*edge)), edge->length(),
                                       side});
            const FaceReference reference{false, index, true};
            incident_faces_[cell_index].push_back(reference);
            if (side == BoundarySide::surface) {
                surface_faces_.push_back(reference);
            }
        }
    };
    append_boundaries(BoundaryKind::top, BoundarySide::surface);
    append_boundaries(BoundaryKind::bottom, BoundarySide::basal);
    append_boundaries(BoundaryKind::left, BoundarySide::left);
    append_boundaries(BoundaryKind::right, BoundarySide::right);
}

MultipointFlux::Sample MultipointFlux::sample(const std::size_t cell,
                                              const FaceReference& reference,
                                              const std::span<const double> conductivity,
                                              const MultipointBoundaryData& boundary_data) const {
    const Eigen::Vector2d cell_center = cell_centers_[cell];
    if (reference.internal) {
        const InternalFace& face = internal_faces_[reference.index];
        const double owner_conductivity = conductivity[face.owner];
        const double neighbor_conductivity = conductivity[face.neighbor];
        const double denominator = owner_conductivity * face.neighbor_distance +
                                   neighbor_conductivity * face.owner_distance;
        const Eigen::Vector2d harmonic_point =
            ((face.owner_distance * face.neighbor_distance *
              (owner_conductivity - neighbor_conductivity) * face.owner_normal) +
             (owner_conductivity * face.neighbor_distance * cell_centers_[face.owner]) +
             (neighbor_conductivity * face.owner_distance * cell_centers_[face.neighbor])) /
            denominator;
        MultipointFluxExpression value;
        if (reference.owner_side) {
            const double neighbor_weight =
                neighbor_conductivity * face.owner_distance / denominator;
            add_term(value, face.owner, 1.0 - neighbor_weight);
            add_term(value, face.neighbor, neighbor_weight);
            return {harmonic_point - cell_center, std::move(value), face.owner_normal,
                    neighbor_weight};
        }
        const double neighbor_weight = owner_conductivity * face.neighbor_distance / denominator;
        add_term(value, face.neighbor, 1.0 - neighbor_weight);
        add_term(value, face.owner, neighbor_weight);
        return {harmonic_point - cell_center, std::move(value), -face.owner_normal,
                neighbor_weight};
    }

    const BoundaryFace& face = boundary_faces_[reference.index];
    const Eigen::Vector2d displacement = face.center - cell_center;
    const double normal_distance = face.outward_normal.dot(displacement);
    if (!(normal_distance > 0.0)) {
        throw std::runtime_error("Invalid boundary face for multipoint flux");
    }
    if (face.side == BoundarySide::surface) {
        MultipointFluxExpression value;
        value.constant = boundary_data.surface_temperature(point(face.center));
        return {displacement, std::move(value), face.outward_normal, 0.0};
    }

    double inward_flux = 0.0;
    switch (face.side) {
    case BoundarySide::surface:
        break;
    case BoundarySide::basal:
        inward_flux = boundary_data.basal_inward_flux(point(face.center));
        break;
    case BoundarySide::left:
        inward_flux = boundary_data.left_inward_flux(point(face.center));
        break;
    case BoundarySide::right:
        inward_flux = boundary_data.right_inward_flux(point(face.center));
        break;
    }
    MultipointFluxExpression value;
    add_term(value, cell, 1.0);
    value.constant = normal_distance * inward_flux / conductivity[cell];
    return {normal_distance * face.outward_normal, std::move(value), face.outward_normal, 0.0};
}

MultipointFlux::OneSidedFlux
MultipointFlux::one_sided_flux(const std::size_t cell, const FaceReference& target,
                               const std::span<const double> conductivity,
                               const MultipointBoundaryData& boundary_data) const {
    const Sample target_sample = sample(cell, target, conductivity, boundary_data);
    const Eigen::Vector2d conormal = conductivity[cell] * target_sample.outward_normal;
    const auto& incident = incident_faces_[cell];
    double best_score = std::numeric_limits<double>::infinity();
    double best_condition = std::numeric_limits<double>::infinity();
    double best_target_coefficient = 0.0;
    double best_other_coefficient = 0.0;
    const FaceReference* best_other = nullptr;

    for (const FaceReference& other : incident) {
        if (other.internal == target.internal && other.index == target.index) {
            continue;
        }
        const Sample other_sample = sample(cell, other, conductivity, boundary_data);
        const Eigen::Vector2d a = target_sample.displacement;
        const Eigen::Vector2d b = other_sample.displacement;
        const double determinant = (a.x() * b.y()) - (a.y() * b.x());
        const double scale = a.norm() * b.norm();
        if (!(scale > 0.0) || std::abs(determinant) <= 1.0e-12 * scale) {
            continue;
        }
        double target_coefficient = ((conormal.x() * b.y()) - (conormal.y() * b.x())) / determinant;
        double other_coefficient = ((a.x() * conormal.y()) - (a.y() * conormal.x())) / determinant;
        const double coefficient_scale =
            std::max({1.0, std::abs(target_coefficient), std::abs(other_coefficient)});
        if (target_coefficient < -1.0e-10 * coefficient_scale ||
            other_coefficient < -1.0e-10 * coefficient_scale) {
            continue;
        }
        target_coefficient = std::max(0.0, target_coefficient);
        other_coefficient = std::max(0.0, other_coefficient);
        const double score = target_coefficient + other_coefficient;
        const double condition = scale / std::abs(determinant);
        if (best_other == nullptr || score < best_score - 1.0e-12 * std::max(1.0, best_score) ||
            (std::abs(score - best_score) <= 1.0e-12 * std::max(1.0, best_score) &&
             condition < best_condition)) {
            best_score = score;
            best_condition = condition;
            best_target_coefficient = target_coefficient;
            best_other_coefficient = other_coefficient;
            best_other = &other;
        }
    }
    if (best_other == nullptr) {
        std::ostringstream message;
        message << "No positive two-face stencil for multipoint flux at cell " << cell
                << ", center=(" << cell_centers_[cell].x() << ',' << cell_centers_[cell].y()
                << "), conormal=(" << conormal.x() << ',' << conormal.y() << "), directions=";
        for (const FaceReference& reference : incident) {
            const Sample incident_sample = sample(cell, reference, conductivity, boundary_data);
            message << " (" << incident_sample.displacement.x() << ','
                    << incident_sample.displacement.y() << ')';
        }
        throw std::runtime_error(message.str());
    }

    const Sample other_sample = sample(cell, *best_other, conductivity, boundary_data);
    MultipointFluxExpression expression =
        scaled(difference(cell, target_sample.value), best_target_coefficient);
    add_scaled(expression, difference(cell, other_sample.value), best_other_coefficient);
    return {std::move(expression), best_target_coefficient * target_sample.neighbor_weight};
}

MultipointFluxes MultipointFlux::compute(const MultipointFluxScheme scheme,
                                         const std::span<const double> conductivity,
                                         const std::span<const double> solution,
                                         const MultipointBoundaryData& boundary_data) const {
    if (conductivity.size() != cell_centers_.size() || solution.size() != cell_centers_.size()) {
        throw std::invalid_argument("Multipoint flux field size does not match the mesh");
    }
    MultipointFluxes result;
    result.internal.reserve(internal_faces_.size());
    for (std::size_t index = 0; index < internal_faces_.size(); ++index) {
        const InternalFace& face = internal_faces_[index];
        const FaceReference owner_reference{true, index, true};
        const FaceReference neighbor_reference{true, index, false};
        OneSidedFlux owner =
            one_sided_flux(face.owner, owner_reference, conductivity, boundary_data);
        OneSidedFlux neighbor =
            one_sided_flux(face.neighbor, neighbor_reference, conductivity, boundary_data);
        MultipointFluxExpression owner_flux;
        MultipointFluxExpression neighbor_flux;
        if (scheme == MultipointFluxScheme::linear) {
            add_scaled(owner_flux, owner.expression, 0.5 * face.length);
            add_scaled(owner_flux, neighbor.expression, -0.5 * face.length);
            neighbor_flux = owner_flux;
        } else {
            // Book equations (2.44)--(2.55).  The conservative variant freezes one common
            // face flux in both rows.  The DMP variant freezes the two equivalent one-sided
            // representations separately; they become equal only at nonlinear convergence.
            MultipointFluxExpression conservative_flux;
            MultipointFluxExpression dmp_owner_flux;
            MultipointFluxExpression dmp_neighbor_flux;
            const double owner_value = owner.expression.value(solution);
            const double neighbor_value = neighbor.expression.value(solution);
            const double epsilon = std::numeric_limits<double>::epsilon() *
                                   std::max({1.0, std::abs(owner_value), std::abs(neighbor_value)});
            if (owner_value * neighbor_value <= 0.0) {
                const double denominator =
                    std::abs(owner_value) + std::abs(neighbor_value) + (2.0 * epsilon);
                const double owner_weight = (std::abs(neighbor_value) + epsilon) / denominator;
                const double neighbor_weight = (std::abs(owner_value) + epsilon) / denominator;
                add_scaled(conservative_flux, owner.expression, face.length * owner_weight);
                add_scaled(conservative_flux, neighbor.expression, -face.length * neighbor_weight);
                add_scaled(dmp_owner_flux, owner.expression, 2.0 * face.length * owner_weight);
                add_scaled(dmp_neighbor_flux, neighbor.expression,
                           -2.0 * face.length * neighbor_weight);
            } else {
                MultipointFluxExpression owner_remainder = owner.expression;
                add_term(owner_remainder, face.owner, -owner.direct_coefficient);
                add_term(owner_remainder, face.neighbor, owner.direct_coefficient);
                MultipointFluxExpression neighbor_remainder = neighbor.expression;
                add_term(neighbor_remainder, face.neighbor, -neighbor.direct_coefficient);
                add_term(neighbor_remainder, face.owner, neighbor.direct_coefficient);
                const double owner_remainder_value = owner_remainder.value(solution);
                const double neighbor_remainder_value = neighbor_remainder.value(solution);
                const double remainder_epsilon = std::numeric_limits<double>::epsilon() *
                                                 std::max({1.0, std::abs(owner_remainder_value),
                                                           std::abs(neighbor_remainder_value)});
                const double denominator = std::abs(owner_remainder_value) +
                                           std::abs(neighbor_remainder_value) +
                                           (2.0 * remainder_epsilon);
                const double owner_weight =
                    (std::abs(neighbor_remainder_value) + remainder_epsilon) / denominator;
                const double neighbor_weight =
                    (std::abs(owner_remainder_value) + remainder_epsilon) / denominator;
                const double direct = owner_weight * owner.direct_coefficient +
                                      neighbor_weight * neighbor.direct_coefficient;
                add_term(conservative_flux, face.owner, face.length * direct);
                add_term(conservative_flux, face.neighbor, -face.length * direct);
                dmp_owner_flux = conservative_flux;
                dmp_neighbor_flux = conservative_flux;
                if (owner_remainder_value * neighbor_remainder_value <= 0.0) {
                    add_scaled(conservative_flux, owner_remainder, face.length * owner_weight);
                    add_scaled(conservative_flux, neighbor_remainder,
                               -face.length * neighbor_weight);
                    add_scaled(dmp_owner_flux, owner_remainder, 2.0 * face.length * owner_weight);
                    add_scaled(dmp_neighbor_flux, neighbor_remainder,
                               -2.0 * face.length * neighbor_weight);
                }
            }
            if (scheme == MultipointFluxScheme::nonlinear_conservative) {
                owner_flux = conservative_flux;
                neighbor_flux = std::move(conservative_flux);
            } else {
                owner_flux = std::move(dmp_owner_flux);
                neighbor_flux = std::move(dmp_neighbor_flux);
            }
        }
        result.maximum_stencil_size = std::max(
            {result.maximum_stencil_size, stencil_size(owner_flux), stencil_size(neighbor_flux)});
        result.internal.push_back(
            {face.owner, face.neighbor, std::move(owner_flux), std::move(neighbor_flux)});
    }

    result.surface.reserve(surface_faces_.size());
    for (const FaceReference reference : surface_faces_) {
        const BoundaryFace& face = boundary_faces_[reference.index];
        OneSidedFlux flux = one_sided_flux(face.cell, reference, conductivity, boundary_data);
        flux.expression = scaled(std::move(flux.expression), face.length);
        result.maximum_stencil_size =
            std::max(result.maximum_stencil_size, stencil_size(flux.expression));
        result.surface.push_back({face.cell, std::move(flux.expression)});
    }
    return result;
}
