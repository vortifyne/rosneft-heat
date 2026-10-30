#include "basin/discretization/nonmatching_column_heat_system.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

std::size_t matrix_triplet_capacity(const NonmatchingColumnGrid& grid) {
    return 2 * grid.cells.size() + 8 * grid.vertical_subfaces.size() +
           2 * grid.surface_vertical_faces.size() + 4 * grid.horizontal_faces.size();
}

} // namespace

NonmatchingColumnHeatSystem::NonmatchingColumnHeatSystem(
    const NonmatchingColumnGrid& grid, const std::span<const double> thermal_conductivity,
    const std::span<const double> heat_production, NonmatchingColumnBoundaryData boundary_data,
    const std::span<const double> volumetric_heat_capacity)
    : grid_(&grid), thermal_conductivity_(thermal_conductivity.begin(), thermal_conductivity.end()),
      heat_production_(heat_production.begin(), heat_production.end()),
      volumetric_heat_capacity_(volumetric_heat_capacity.begin(), volumetric_heat_capacity.end()),
      boundary_data_(std::move(boundary_data)) {
    if (thermal_conductivity.size() != grid.cells.size() ||
        heat_production.size() != grid.cells.size() ||
        (!volumetric_heat_capacity.empty() &&
         volumetric_heat_capacity.size() != grid.cells.size())) {
        throw std::invalid_argument("Nonmatching-column field sizes do not match the grid");
    }
    if (!boundary_data_.surface_temperature || !boundary_data_.basal_inward_flux) {
        throw std::invalid_argument("Nonmatching-column boundary functions must be set");
    }
    triplets_.reserve(matrix_triplet_capacity(grid));
}

NonmatchingColumnHeatSystem::NonmatchingColumnHeatSystem(
    const NonmatchingColumnGrid& grid, const std::span<const double> porosity,
    const std::span<const LithotypeThermophysicalProperties* const> lithotypes,
    const FluidThermophysicalProperties fluid, NonmatchingColumnBoundaryData boundary_data)
    : grid_(&grid), porosity_(porosity), lithotypes_(lithotypes), fluid_(fluid),
      thermal_conductivity_(grid.cells.size()), heat_production_(grid.cells.size()),
      volumetric_heat_capacity_(grid.cells.size()), boundary_data_(std::move(boundary_data)) {
    if (porosity.size() != grid.cells.size() || lithotypes.size() != grid.cells.size()) {
        throw std::invalid_argument("Nonmatching-column material sizes do not match the grid");
    }
    if (!boundary_data_.surface_temperature || !boundary_data_.basal_inward_flux) {
        throw std::invalid_argument("Nonmatching-column boundary functions must be set");
    }
    triplets_.reserve(matrix_triplet_capacity(grid));
}

void NonmatchingColumnHeatSystem::set_flux_scheme(const MultipointFluxScheme scheme) noexcept {
    flux_scheme_ = scheme;
}

Vector::Index NonmatchingColumnHeatSystem::size() const noexcept {
    return static_cast<Vector::Index>(grid_->cells.size());
}

void NonmatchingColumnHeatSystem::update_coefficients(const Vector& solution) const {
    if (solution.size() != size()) {
        throw std::invalid_argument("Nonmatching-column solution size does not match the grid");
    }
    if (!lithotypes_.empty()) {
        evaluate_thermophysical_property_fields(
            {solution.native().data(), static_cast<std::size_t>(solution.size())}, porosity_,
            lithotypes_, fluid_, thermal_conductivity_, volumetric_heat_capacity_,
            heat_production_);
    }
}

double NonmatchingColumnHeatSystem::horizontal_transmissibility(
    const NonmatchingHorizontalFace& face) const {
    const NonmatchingColumnCell& upper = grid_->cells[face.upper_cell];
    const NonmatchingColumnCell& lower = grid_->cells[face.lower_cell];
    const double upper_distance = face.z - upper.center().z;
    const double lower_distance = lower.center().z - face.z;
    return face.length / ((upper_distance / thermal_conductivity_[face.upper_cell]) +
                          (lower_distance / thermal_conductivity_[face.lower_cell]));
}

void NonmatchingColumnHeatSystem::assemble_residual(const Vector& solution,
                                                    Vector& residual) const {
    update_coefficients(solution);
    residual.resize(size());
    residual.set_zero();
    add_vertical_flux_residual(solution, residual);
    add_surface_step_residual(solution, residual);
    add_horizontal_flux_residual(solution, residual);
    add_boundary_residual(solution, residual);
    add_source_residual(residual);
}

void NonmatchingColumnHeatSystem::assemble_residual(double, const Vector& solution,
                                                    const Vector& solution_derivative,
                                                    Vector& residual) const {
    if (solution_derivative.size() != size()) {
        throw std::invalid_argument("Column-grid temperature derivative size does not match");
    }
    assemble_residual(solution, residual);
    add_accumulation_residual(solution_derivative, residual);
}

void NonmatchingColumnHeatSystem::assemble_matrix(const Vector& solution,
                                                  SparseMatrix& matrix) const {
    assemble_matrix(solution, 0.0, matrix);
}

void NonmatchingColumnHeatSystem::assemble_matrix(NonlinearMethod method, double,
                                                  const Vector& solution,
                                                  const Vector& solution_derivative,
                                                  double derivative_shift,
                                                  SparseMatrix& matrix) const {
    if (method != NonlinearMethod::picard) {
        throw std::invalid_argument("Column-grid heat system currently supports Picard only");
    }
    if (solution_derivative.size() != size() || !std::isfinite(derivative_shift) ||
        derivative_shift < 0.0) {
        throw std::invalid_argument("Invalid column-grid derivative or BDF shift");
    }
    assemble_matrix(solution, derivative_shift, matrix);
}

void NonmatchingColumnHeatSystem::assemble_matrix(const Vector& solution,
                                                  const double derivative_shift,
                                                  SparseMatrix& matrix) const {
    update_coefficients(solution);
    triplets_.clear();
    add_vertical_flux_matrix(solution);
    add_surface_step_matrix();
    add_horizontal_flux_matrix();
    add_boundary_matrix();
    // Prescribed sources and Neumann data have no matrix contribution.
    add_accumulation_matrix(derivative_shift);
    matrix.resize(size(), size());
    matrix.set_from_triplets(triplets_);
}

void NonmatchingColumnHeatSystem::add_vertical_flux_residual(const Vector& solution,
                                                             Vector& residual) const {
    const std::span<const double> values{solution.native().data(),
                                         static_cast<std::size_t>(solution.size())};
    NonmatchingColumnFluxStencil stencil;
    for (const auto& face : grid_->vertical_subfaces) {
        compute_internal_flux(*grid_, face, flux_scheme_, thermal_conductivity_, values,
                              boundary_data_, stencil);
        const double flux = evaluate_flux(stencil, values);
        residual[static_cast<Vector::Index>(face.left_cell)] += flux;
        residual[static_cast<Vector::Index>(face.right_cell)] -= flux;
    }
}

void NonmatchingColumnHeatSystem::add_vertical_flux_matrix(const Vector& solution) const {
    const std::span<const double> values{solution.native().data(),
                                         static_cast<std::size_t>(solution.size())};
    NonmatchingColumnFluxStencil stencil;
    for (const auto& face : grid_->vertical_subfaces) {
        compute_internal_flux(*grid_, face, flux_scheme_, thermal_conductivity_, values,
                              boundary_data_, stencil);
        for (std::size_t i = 0; i < stencil.size; ++i) {
            triplets_.emplace_back(static_cast<Vector::Index>(face.left_cell),
                                   static_cast<Vector::Index>(stencil.cells[i]),
                                   stencil.coefficients[i]);
        }
        for (std::size_t i = 0; i < stencil.size; ++i) {
            triplets_.emplace_back(static_cast<Vector::Index>(face.right_cell),
                                   static_cast<Vector::Index>(stencil.cells[i]),
                                   -stencil.coefficients[i]);
        }
    }
}

void NonmatchingColumnHeatSystem::add_surface_step_residual(const Vector& solution,
                                                            Vector& residual) const {
    const std::span<const double> values{solution.native().data(),
                                         static_cast<std::size_t>(solution.size())};
    NonmatchingColumnFluxStencil stencil;
    for (const auto& face : grid_->surface_vertical_faces) {
        compute_surface_flux(*grid_, face, thermal_conductivity_, boundary_data_, stencil);
        residual[static_cast<Vector::Index>(face.cell)] += evaluate_flux(stencil, values);
    }
}

void NonmatchingColumnHeatSystem::add_surface_step_matrix() const {
    NonmatchingColumnFluxStencil stencil;
    for (const auto& face : grid_->surface_vertical_faces) {
        compute_surface_flux(*grid_, face, thermal_conductivity_, boundary_data_, stencil);
        for (std::size_t i = 0; i < stencil.size; ++i) {
            triplets_.emplace_back(static_cast<Vector::Index>(face.cell),
                                   static_cast<Vector::Index>(stencil.cells[i]),
                                   stencil.coefficients[i]);
        }
    }
}

void NonmatchingColumnHeatSystem::add_horizontal_flux_residual(const Vector& solution,
                                                               Vector& residual) const {
    for (const auto& face : grid_->horizontal_faces) {
        const double flux = horizontal_transmissibility(face) *
                            (solution[static_cast<Vector::Index>(face.upper_cell)] -
                             solution[static_cast<Vector::Index>(face.lower_cell)]);
        residual[static_cast<Vector::Index>(face.upper_cell)] += flux;
        residual[static_cast<Vector::Index>(face.lower_cell)] -= flux;
    }
}

void NonmatchingColumnHeatSystem::add_horizontal_flux_matrix() const {
    for (const auto& face : grid_->horizontal_faces) {
        const double value = horizontal_transmissibility(face);
        triplets_.emplace_back(static_cast<Vector::Index>(face.upper_cell),
                               static_cast<Vector::Index>(face.upper_cell), value);
        triplets_.emplace_back(static_cast<Vector::Index>(face.upper_cell),
                               static_cast<Vector::Index>(face.lower_cell), -value);
        triplets_.emplace_back(static_cast<Vector::Index>(face.lower_cell),
                               static_cast<Vector::Index>(face.upper_cell), -value);
        triplets_.emplace_back(static_cast<Vector::Index>(face.lower_cell),
                               static_cast<Vector::Index>(face.lower_cell), value);
    }
}

void NonmatchingColumnHeatSystem::add_boundary_residual(const Vector& solution,
                                                        Vector& residual) const {
    for (const std::size_t cell_index : grid_->top_cells) {
        const NonmatchingColumnCell& cell = grid_->cells[cell_index];
        const double transmissibility =
            thermal_conductivity_[cell_index] * cell.width() / (0.5 * cell.height());
        residual[static_cast<Vector::Index>(cell_index)] +=
            transmissibility * (solution[static_cast<Vector::Index>(cell_index)] -
                                boundary_data_.surface_temperature({cell.center().x, cell.z_top}));
    }
    for (const std::size_t cell_index : grid_->bottom_cells) {
        const NonmatchingColumnCell& cell = grid_->cells[cell_index];
        residual[static_cast<Vector::Index>(cell_index)] -=
            boundary_data_.basal_inward_flux({cell.center().x, cell.z_bottom}) * cell.width();
    }
    const auto add_lateral = [&residual](const auto& flux_function, const auto& column,
                                         const auto& cells) {
        if (!flux_function) {
            return;
        }
        for (const std::size_t cell_index : column) {
            const NonmatchingColumnCell& cell = cells[cell_index];
            residual[static_cast<Vector::Index>(cell_index)] -=
                flux_function(cell.center()) * cell.height();
        }
    };
    add_lateral(boundary_data_.left_inward_flux, grid_->columns.front(), grid_->cells);
    add_lateral(boundary_data_.right_inward_flux, grid_->columns.back(), grid_->cells);
}

void NonmatchingColumnHeatSystem::add_boundary_matrix() const {
    for (const std::size_t cell_index : grid_->top_cells) {
        const NonmatchingColumnCell& cell = grid_->cells[cell_index];
        const double value =
            thermal_conductivity_[cell_index] * cell.width() / (0.5 * cell.height());
        triplets_.emplace_back(static_cast<Vector::Index>(cell_index),
                               static_cast<Vector::Index>(cell_index), value);
    }
}

void NonmatchingColumnHeatSystem::add_source_residual(Vector& residual) const {
    for (const auto& cell : grid_->cells) {
        residual[static_cast<Vector::Index>(cell.index)] -=
            heat_production_[cell.index] * cell.area();
    }
}

void NonmatchingColumnHeatSystem::add_accumulation_residual(const Vector& derivative,
                                                            Vector& residual) const {
    if (volumetric_heat_capacity_.empty()) {
        return;
    }
    for (const auto& cell : grid_->cells) {
        const auto index = static_cast<Vector::Index>(cell.index);
        residual[index] += cell.area() * volumetric_heat_capacity_[cell.index] * derivative[index];
    }
}

void NonmatchingColumnHeatSystem::add_accumulation_matrix(const double derivative_shift) const {
    if (derivative_shift == 0.0 || volumetric_heat_capacity_.empty()) {
        return;
    }
    for (const auto& cell : grid_->cells) {
        const auto index = static_cast<Vector::Index>(cell.index);
        triplets_.emplace_back(
            index, index, derivative_shift * cell.area() * volumetric_heat_capacity_[cell.index]);
    }
}
