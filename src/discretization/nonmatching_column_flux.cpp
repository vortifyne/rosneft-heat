#include "discretization/nonmatching_column_flux.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

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

struct NonmatchingColumnFlux::Sample {
    Point2D displacement{};
    MultipointFluxExpression value;
    double neighbor_weight = 0.0;
};

struct NonmatchingColumnFlux::OneSidedFlux {
    MultipointFluxExpression expression;
    double direct_coefficient = 0.0;
};

NonmatchingColumnFlux::NonmatchingColumnFlux(const NonmatchingColumnGrid& grid) : grid_(&grid) {}

NonmatchingColumnFlux::Sample
NonmatchingColumnFlux::target_sample(const std::size_t cell, const std::size_t neighbor,
                                     const double interface_x,
                                     const std::span<const double> conductivity) const {
    const NonmatchingColumnCell& current = grid_->cells.at(cell);
    const NonmatchingColumnCell& other = grid_->cells.at(neighbor);
    const Point2D current_center = current.center();
    const Point2D other_center = other.center();
    const double current_distance = std::abs(interface_x - current_center.x);
    const double other_distance = std::abs(other_center.x - interface_x);
    const double denominator =
        conductivity[cell] * other_distance + conductivity[neighbor] * current_distance;
    if (!(current_distance > 0.0) || !(other_distance > 0.0) || !(denominator > 0.0)) {
        throw std::runtime_error("Invalid nonmatching vertical-interface geometry");
    }
    const double neighbor_weight = conductivity[neighbor] * current_distance / denominator;
    MultipointFluxExpression value;
    add_term(value, cell, 1.0 - neighbor_weight);
    add_term(value, neighbor, neighbor_weight);
    const double harmonic_z = (conductivity[cell] * other_distance * current_center.z +
                               conductivity[neighbor] * current_distance * other_center.z) /
                              denominator;
    return {{interface_x - current_center.x, harmonic_z - current_center.z},
            std::move(value),
            neighbor_weight};
}

NonmatchingColumnFlux::Sample
NonmatchingColumnFlux::vertical_sample(const std::size_t cell, const bool upward,
                                       const std::span<const double> conductivity,
                                       const NonmatchingColumnBoundaryData& boundary_data) const {
    const NonmatchingColumnCell& current = grid_->cells.at(cell);
    const Point2D center = current.center();
    const std::optional<std::size_t> neighbor = upward ? current.upward : current.downward;
    const double face_z = upward ? current.z_top : current.z_bottom;
    const double distance = std::abs(face_z - center.z);
    if (neighbor) {
        const NonmatchingColumnCell& other = grid_->cells.at(*neighbor);
        const double other_distance = std::abs(other.center().z - face_z);
        const double denominator =
            conductivity[cell] * other_distance + conductivity[*neighbor] * distance;
        const double neighbor_weight = conductivity[*neighbor] * distance / denominator;
        MultipointFluxExpression value;
        add_term(value, cell, 1.0 - neighbor_weight);
        add_term(value, *neighbor, neighbor_weight);
        return {{0.0, face_z - center.z}, std::move(value), neighbor_weight};
    }
    MultipointFluxExpression value;
    if (upward) {
        value.constant = boundary_data.surface_temperature({center.x, face_z});
    } else {
        add_term(value, cell, 1.0);
        value.constant =
            distance * boundary_data.basal_inward_flux({center.x, face_z}) / conductivity[cell];
    }
    return {{0.0, face_z - center.z}, std::move(value), 0.0};
}

NonmatchingColumnFlux::OneSidedFlux
NonmatchingColumnFlux::one_sided_flux(const std::size_t cell, const std::size_t neighbor,
                                      const double interface_x,
                                      const std::span<const double> conductivity,
                                      const NonmatchingColumnBoundaryData& boundary_data) const {
    const Sample target = target_sample(cell, neighbor, interface_x, conductivity);
    const double target_coefficient = conductivity[cell] / std::abs(target.displacement.x);
    const bool use_upward = target.displacement.z >= 0.0;
    const Sample vertical = vertical_sample(cell, use_upward, conductivity, boundary_data);
    double vertical_coefficient =
        -target_coefficient * target.displacement.z / vertical.displacement.z;
    const double scale = std::max(1.0, target_coefficient);
    if (vertical_coefficient < -1.0e-12 * scale) {
        throw std::runtime_error("Nonmatching column has no positive vertical flux stencil");
    }
    vertical_coefficient = std::max(0.0, vertical_coefficient);
    MultipointFluxExpression expression = difference(cell, target.value);
    for (MultipointFluxTerm& term : expression.terms) {
        term.coefficient *= target_coefficient;
    }
    expression.constant *= target_coefficient;
    add_scaled(expression, difference(cell, vertical.value), vertical_coefficient);
    return {std::move(expression), target_coefficient * target.neighbor_weight};
}

MultipointFluxExpression NonmatchingColumnFlux::surface_boundary_flux(
    const NonmatchingVerticalBoundaryFace& face, const std::span<const double> conductivity,
    const NonmatchingColumnBoundaryData& boundary_data) const {
    const NonmatchingColumnCell& cell = grid_->cells.at(face.cell);
    const Point2D center = cell.center();
    const Point2D boundary_point{face.x, face.center_z()};
    const Point2D displacement{boundary_point.x - center.x, boundary_point.z - center.z};
    if (!(displacement.x * face.outward_x > 0.0)) {
        throw std::runtime_error("Invalid vertical surface-step geometry");
    }
    MultipointFluxExpression boundary_value;
    boundary_value.constant = boundary_data.surface_temperature(boundary_point);
    const double target_coefficient = conductivity[face.cell] / std::abs(displacement.x);
    MultipointFluxExpression expression = difference(face.cell, boundary_value);
    for (MultipointFluxTerm& term : expression.terms) {
        term.coefficient *= target_coefficient;
    }
    expression.constant *= target_coefficient;

    const double vertical_tolerance =
        1.0e-14 * std::max({1.0, cell.height(), std::abs(displacement.z)});
    if (std::abs(displacement.z) > vertical_tolerance) {
        const Sample vertical =
            vertical_sample(face.cell, displacement.z > 0.0, conductivity, boundary_data);
        double vertical_coefficient =
            -target_coefficient * displacement.z / vertical.displacement.z;
        const double coefficient_scale = std::max(1.0, target_coefficient);
        if (vertical_coefficient < -1.0e-12 * coefficient_scale) {
            throw std::runtime_error("Surface step has no positive vertical flux stencil");
        }
        vertical_coefficient = std::max(0.0, vertical_coefficient);
        add_scaled(expression, difference(face.cell, vertical.value), vertical_coefficient);
    }
    for (MultipointFluxTerm& term : expression.terms) {
        term.coefficient *= face.length();
    }
    expression.constant *= face.length();
    return expression;
}

NonmatchingColumnFluxes
NonmatchingColumnFlux::compute(const MultipointFluxScheme scheme,
                               const std::span<const double> conductivity,
                               const std::span<const double> solution,
                               const NonmatchingColumnBoundaryData& boundary_data) const {
    if (conductivity.size() != grid_->cells.size() || solution.size() != grid_->cells.size()) {
        throw std::invalid_argument("Nonmatching-column field size does not match the grid");
    }
    if (!boundary_data.surface_temperature || !boundary_data.basal_inward_flux) {
        throw std::invalid_argument("Nonmatching-column boundary functions must be set");
    }
    for (const double value : conductivity) {
        if (!std::isfinite(value) || !(value > 0.0)) {
            throw std::invalid_argument("Nonmatching-column conductivity must be positive");
        }
    }

    NonmatchingColumnFluxes result;
    result.vertical.reserve(grid_->vertical_subfaces.size());
    for (const NonmatchingVerticalSubface& face : grid_->vertical_subfaces) {
        OneSidedFlux left =
            one_sided_flux(face.left_cell, face.right_cell, face.x, conductivity, boundary_data);
        OneSidedFlux right =
            one_sided_flux(face.right_cell, face.left_cell, face.x, conductivity, boundary_data);
        MultipointFluxExpression left_flux;
        MultipointFluxExpression right_flux;
        if (scheme == MultipointFluxScheme::linear) {
            // Conductance-dependent convex weights from the harmonic-point
            // linearity-preserving construction; 1/2 is valid only for equal
            // normal conductances.
            const double left_distance = std::abs(face.x - grid_->cells[face.left_cell].center().x);
            const double right_distance =
                std::abs(grid_->cells[face.right_cell].center().x - face.x);
            const double left_conductance = conductivity[face.left_cell] / left_distance;
            const double right_conductance = conductivity[face.right_cell] / right_distance;
            const double conductance_sum = left_conductance + right_conductance;
            const double left_weight = right_conductance / conductance_sum;
            const double right_weight = left_conductance / conductance_sum;
            add_scaled(left_flux, left.expression, left_weight * face.length());
            add_scaled(left_flux, right.expression, -right_weight * face.length());
            right_flux = left_flux;
        } else {
            MultipointFluxExpression conservative;
            const double left_value = left.expression.value(solution);
            const double right_value = right.expression.value(solution);
            const double epsilon = std::numeric_limits<double>::epsilon() *
                                   std::max({1.0, std::abs(left_value), std::abs(right_value)});
            if (left_value * right_value <= 0.0) {
                const double denominator =
                    std::abs(left_value) + std::abs(right_value) + 2.0 * epsilon;
                const double left_weight = (std::abs(right_value) + epsilon) / denominator;
                const double right_weight = (std::abs(left_value) + epsilon) / denominator;
                add_scaled(conservative, left.expression, face.length() * left_weight);
                add_scaled(conservative, right.expression, -face.length() * right_weight);
            } else {
                MultipointFluxExpression left_remainder = left.expression;
                add_term(left_remainder, face.left_cell, -left.direct_coefficient);
                add_term(left_remainder, face.right_cell, left.direct_coefficient);
                MultipointFluxExpression right_remainder = right.expression;
                add_term(right_remainder, face.right_cell, -right.direct_coefficient);
                add_term(right_remainder, face.left_cell, right.direct_coefficient);
                const double left_remainder_value = left_remainder.value(solution);
                const double right_remainder_value = right_remainder.value(solution);
                const double remainder_epsilon = std::numeric_limits<double>::epsilon() *
                                                 std::max({1.0, std::abs(left_remainder_value),
                                                           std::abs(right_remainder_value)});
                const double denominator = std::abs(left_remainder_value) +
                                           std::abs(right_remainder_value) +
                                           2.0 * remainder_epsilon;
                const double left_weight =
                    (std::abs(right_remainder_value) + remainder_epsilon) / denominator;
                const double right_weight =
                    (std::abs(left_remainder_value) + remainder_epsilon) / denominator;
                const double direct =
                    left_weight * left.direct_coefficient + right_weight * right.direct_coefficient;
                add_term(conservative, face.left_cell, face.length() * direct);
                add_term(conservative, face.right_cell, -face.length() * direct);
                if (left_remainder_value * right_remainder_value <= 0.0) {
                    add_scaled(conservative, left_remainder, face.length() * left_weight);
                    add_scaled(conservative, right_remainder, -face.length() * right_weight);
                }
            }
            left_flux = conservative;
            right_flux = std::move(conservative);
        }
        result.maximum_stencil_size = std::max(
            {result.maximum_stencil_size, stencil_size(left_flux), stencil_size(right_flux)});
        result.vertical.push_back(
            {face.left_cell, face.right_cell, std::move(left_flux), std::move(right_flux)});
    }
    result.surface.reserve(grid_->surface_vertical_faces.size());
    for (const NonmatchingVerticalBoundaryFace& face : grid_->surface_vertical_faces) {
        MultipointFluxExpression expression =
            surface_boundary_flux(face, conductivity, boundary_data);
        result.maximum_stencil_size =
            std::max(result.maximum_stencil_size, stencil_size(expression));
        result.surface.push_back({face.cell, std::move(expression)});
    }
    return result;
}
