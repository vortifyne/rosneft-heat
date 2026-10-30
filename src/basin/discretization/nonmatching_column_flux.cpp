#include "basin/discretization/nonmatching_column_flux.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

void add_term(NonmatchingColumnFluxStencil& stencil, const std::size_t cell,
              const double coefficient) {
    if (coefficient == 0.0) {
        return;
    }
    for (std::size_t i = 0; i < stencil.size; ++i) {
        if (stencil.cells[i] == cell) {
            stencil.coefficients[i] += coefficient;
            return;
        }
    }
    stencil.cells[stencil.size] = cell;
    stencil.coefficients[stencil.size++] = coefficient;
}

void add_scaled(NonmatchingColumnFluxStencil& target, const NonmatchingColumnFluxStencil& source,
                const double scale) {
    for (std::size_t i = 0; i < source.size; ++i) {
        add_term(target, source.cells[i], scale * source.coefficients[i]);
    }
    target.constant += scale * source.constant;
}

NonmatchingColumnFluxStencil difference(const std::size_t cell,
                                        const NonmatchingColumnFluxStencil& sample) {
    NonmatchingColumnFluxStencil result;
    add_term(result, cell, 1.0);
    add_scaled(result, sample, -1.0);
    return result;
}

double vertical_sample(const NonmatchingColumnGrid& grid, const std::size_t cell, const bool upward,
                       const std::span<const double> conductivity,
                       const NonmatchingColumnBoundaryData& boundary_data,
                       NonmatchingColumnFluxStencil& sample) {
    const NonmatchingColumnCell& current = grid.cells[cell];
    const Point2D center = current.center();
    const auto neighbor = upward ? current.upward : current.downward;
    const double face_z = upward ? current.z_top : current.z_bottom;
    const double distance = std::abs(face_z - center.z);
    if (neighbor) {
        const NonmatchingColumnCell& other = grid.cells[*neighbor];
        const double other_distance = std::abs(other.center().z - face_z);
        const double denominator =
            conductivity[cell] * other_distance + conductivity[*neighbor] * distance;
        const double neighbor_weight = conductivity[*neighbor] * distance / denominator;
        add_term(sample, cell, 1.0 - neighbor_weight);
        add_term(sample, *neighbor, neighbor_weight);
    } else if (upward) {
        sample.constant = boundary_data.surface_temperature({center.x, face_z});
    } else {
        add_term(sample, cell, 1.0);
        sample.constant =
            distance * boundary_data.basal_inward_flux({center.x, face_z}) / conductivity[cell];
    }
    return face_z - center.z;
}

double one_sided_flux(const NonmatchingColumnGrid& grid, const std::size_t cell,
                      const std::size_t neighbor, const double interface_x,
                      const std::span<const double> conductivity,
                      const NonmatchingColumnBoundaryData& boundary_data,
                      NonmatchingColumnFluxStencil& stencil) {
    const Point2D current_center = grid.cells[cell].center();
    const Point2D other_center = grid.cells[neighbor].center();
    const double current_distance = std::abs(interface_x - current_center.x);
    const double other_distance = std::abs(other_center.x - interface_x);
    const double denominator =
        conductivity[cell] * other_distance + conductivity[neighbor] * current_distance;
    if (!(current_distance > 0.0) || !(other_distance > 0.0) || !(denominator > 0.0)) {
        throw std::runtime_error("Invalid nonmatching vertical-interface geometry");
    }
    const double neighbor_weight = conductivity[neighbor] * current_distance / denominator;
    NonmatchingColumnFluxStencil target;
    add_term(target, cell, 1.0 - neighbor_weight);
    add_term(target, neighbor, neighbor_weight);
    const double harmonic_z = (conductivity[cell] * other_distance * current_center.z +
                               conductivity[neighbor] * current_distance * other_center.z) /
                              denominator;
    const Point2D displacement{interface_x - current_center.x, harmonic_z - current_center.z};
    const double target_coefficient = conductivity[cell] / std::abs(displacement.x);
    NonmatchingColumnFluxStencil vertical;
    const double vertical_displacement =
        vertical_sample(grid, cell, displacement.z >= 0.0, conductivity, boundary_data, vertical);
    double vertical_coefficient = -target_coefficient * displacement.z / vertical_displacement;
    const double scale = std::max(1.0, target_coefficient);
    if (vertical_coefficient < -1.0e-12 * scale) {
        throw std::runtime_error("Nonmatching column has no positive vertical flux stencil");
    }
    vertical_coefficient = std::max(0.0, vertical_coefficient);
    stencil = difference(cell, target);
    for (std::size_t i = 0; i < stencil.size; ++i) {
        stencil.coefficients[i] *= target_coefficient;
    }
    stencil.constant *= target_coefficient;
    add_scaled(stencil, difference(cell, vertical), vertical_coefficient);
    return target_coefficient * neighbor_weight;
}

} // namespace

double evaluate_flux(const NonmatchingColumnFluxStencil& stencil,
                     const std::span<const double> temperature) {
    double result = stencil.constant;
    for (std::size_t i = 0; i < stencil.size; ++i) {
        result += stencil.coefficients[i] * temperature[stencil.cells[i]];
    }
    return result;
}

void compute_internal_flux(const NonmatchingColumnGrid& grid,
                           const NonmatchingVerticalSubface& face,
                           const MultipointFluxScheme scheme,
                           const std::span<const double> conductivity,
                           const std::span<const double> temperature,
                           const NonmatchingColumnBoundaryData& boundary_data,
                           NonmatchingColumnFluxStencil& stencil) {
    stencil.size = 0;
    stencil.constant = 0.0;
    NonmatchingColumnFluxStencil left;
    NonmatchingColumnFluxStencil right;
    const double left_direct = one_sided_flux(grid, face.left_cell, face.right_cell, face.x,
                                              conductivity, boundary_data, left);
    const double right_direct = one_sided_flux(grid, face.right_cell, face.left_cell, face.x,
                                               conductivity, boundary_data, right);
    if (scheme == MultipointFluxScheme::linear) {
        // Harmonic-point weights; 1/2 is valid only for equal normal conductances.
        const double left_distance = std::abs(face.x - grid.cells[face.left_cell].center().x);
        const double right_distance = std::abs(grid.cells[face.right_cell].center().x - face.x);
        const double left_conductance = conductivity[face.left_cell] / left_distance;
        const double right_conductance = conductivity[face.right_cell] / right_distance;
        const double conductance_sum = left_conductance + right_conductance;
        const double left_weight = right_conductance / conductance_sum;
        const double right_weight = left_conductance / conductance_sum;
        add_scaled(stencil, left, left_weight * face.length());
        add_scaled(stencil, right, -right_weight * face.length());
    } else {
        const double left_value = evaluate_flux(left, temperature);
        const double right_value = evaluate_flux(right, temperature);
        const double epsilon = std::numeric_limits<double>::epsilon() *
                               std::max({1.0, std::abs(left_value), std::abs(right_value)});
        if (left_value * right_value <= 0.0) {
            const double denominator = std::abs(left_value) + std::abs(right_value) + 2.0 * epsilon;
            const double left_weight = (std::abs(right_value) + epsilon) / denominator;
            const double right_weight = (std::abs(left_value) + epsilon) / denominator;
            add_scaled(stencil, left, face.length() * left_weight);
            add_scaled(stencil, right, -face.length() * right_weight);
        } else {
            NonmatchingColumnFluxStencil left_remainder = left;
            add_term(left_remainder, face.left_cell, -left_direct);
            add_term(left_remainder, face.right_cell, left_direct);
            NonmatchingColumnFluxStencil right_remainder = right;
            add_term(right_remainder, face.right_cell, -right_direct);
            add_term(right_remainder, face.left_cell, right_direct);
            const double left_remainder_value = evaluate_flux(left_remainder, temperature);
            const double right_remainder_value = evaluate_flux(right_remainder, temperature);
            const double remainder_epsilon =
                std::numeric_limits<double>::epsilon() *
                std::max({1.0, std::abs(left_remainder_value), std::abs(right_remainder_value)});
            const double denominator = std::abs(left_remainder_value) +
                                       std::abs(right_remainder_value) + 2.0 * remainder_epsilon;
            const double left_weight =
                (std::abs(right_remainder_value) + remainder_epsilon) / denominator;
            const double right_weight =
                (std::abs(left_remainder_value) + remainder_epsilon) / denominator;
            const double direct = left_weight * left_direct + right_weight * right_direct;
            add_term(stencil, face.left_cell, face.length() * direct);
            add_term(stencil, face.right_cell, -face.length() * direct);
            if (left_remainder_value * right_remainder_value <= 0.0) {
                add_scaled(stencil, left_remainder, face.length() * left_weight);
                add_scaled(stencil, right_remainder, -face.length() * right_weight);
            }
        }
    }
}

void compute_surface_flux(const NonmatchingColumnGrid& grid,
                          const NonmatchingVerticalBoundaryFace& face,
                          const std::span<const double> conductivity,
                          const NonmatchingColumnBoundaryData& boundary_data,
                          NonmatchingColumnFluxStencil& stencil) {
    const NonmatchingColumnCell& cell = grid.cells[face.cell];
    const Point2D center = cell.center();
    const Point2D boundary_point{face.x, face.center_z()};
    const Point2D displacement{boundary_point.x - center.x, boundary_point.z - center.z};
    if (!(displacement.x * face.outward_x > 0.0)) {
        throw std::runtime_error("Invalid vertical surface-step geometry");
    }
    NonmatchingColumnFluxStencil boundary_value;
    boundary_value.constant = boundary_data.surface_temperature(boundary_point);
    const double target_coefficient = conductivity[face.cell] / std::abs(displacement.x);
    stencil = difference(face.cell, boundary_value);
    for (std::size_t i = 0; i < stencil.size; ++i) {
        stencil.coefficients[i] *= target_coefficient;
    }
    stencil.constant *= target_coefficient;

    const double vertical_tolerance =
        1.0e-14 * std::max({1.0, cell.height(), std::abs(displacement.z)});
    if (std::abs(displacement.z) > vertical_tolerance) {
        NonmatchingColumnFluxStencil vertical;
        const double vertical_displacement = vertical_sample(grid, face.cell, displacement.z > 0.0,
                                                             conductivity, boundary_data, vertical);
        double vertical_coefficient = -target_coefficient * displacement.z / vertical_displacement;
        const double coefficient_scale = std::max(1.0, target_coefficient);
        if (vertical_coefficient < -1.0e-12 * coefficient_scale) {
            throw std::runtime_error("Surface step has no positive vertical flux stencil");
        }
        vertical_coefficient = std::max(0.0, vertical_coefficient);
        add_scaled(stencil, difference(face.cell, vertical), vertical_coefficient);
    }
    for (std::size_t i = 0; i < stencil.size; ++i) {
        stencil.coefficients[i] *= face.length();
    }
    stencil.constant *= face.length();
}
