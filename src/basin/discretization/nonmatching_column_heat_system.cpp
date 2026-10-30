#include "basin/discretization/nonmatching_column_heat_system.hpp"

#include <stdexcept>
#include <utility>

NonmatchingColumnHeatSystem::NonmatchingColumnHeatSystem(
    const NonmatchingColumnGrid& grid, const std::span<const double> thermal_conductivity,
    const std::span<const double> heat_production, NonmatchingColumnBoundaryData boundary_data,
    PropertyUpdater property_updater)
    : grid_(&grid), thermal_conductivity_(thermal_conductivity.begin(), thermal_conductivity.end()),
      heat_production_(heat_production.begin(), heat_production.end()),
      boundary_data_(std::move(boundary_data)), property_updater_(std::move(property_updater)),
      vertical_flux_(grid) {
    if (thermal_conductivity.size() != grid.cells.size() ||
        heat_production.size() != grid.cells.size()) {
        throw std::invalid_argument("Nonmatching-column field sizes do not match the grid");
    }
    if (!boundary_data_.surface_temperature || !boundary_data_.basal_inward_flux) {
        throw std::invalid_argument("Nonmatching-column boundary functions must be set");
    }
}

void NonmatchingColumnHeatSystem::set_flux_scheme(const MultipointFluxScheme scheme) noexcept {
    flux_scheme_ = scheme;
}

Vector::Index NonmatchingColumnHeatSystem::size() const noexcept {
    return static_cast<Vector::Index>(grid_->cells.size());
}

void NonmatchingColumnHeatSystem::update_properties(const Vector& solution) const {
    if (solution.size() != size()) {
        throw std::invalid_argument("Nonmatching-column solution size does not match the grid");
    }
    if (property_updater_) {
        property_updater_({solution.native().data(), static_cast<std::size_t>(solution.size())},
                          thermal_conductivity_, heat_production_);
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
    update_properties(solution);
    residual.resize(size());
    residual.set_zero();
    const NonmatchingColumnFluxes vertical = vertical_flux_.compute(
        flux_scheme_, thermal_conductivity_,
        {solution.native().data(), static_cast<std::size_t>(solution.size())}, boundary_data_);
    const auto values = std::span<const double>(solution.native().data(),
                                                static_cast<std::size_t>(solution.size()));
    for (const NonmatchingColumnInternalFlux& flux : vertical.vertical) {
        const double value = flux.expression.value(values);
        residual[static_cast<Vector::Index>(flux.left_cell)] += value;
        residual[static_cast<Vector::Index>(flux.right_cell)] -= value;
    }
    for (const NonmatchingColumnBoundaryFlux& flux : vertical.surface) {
        residual[static_cast<Vector::Index>(flux.cell)] += flux.expression.value(values);
    }
    for (const NonmatchingHorizontalFace& face : grid_->horizontal_faces) {
        const double flux = horizontal_transmissibility(face) *
                            (solution[static_cast<Vector::Index>(face.upper_cell)] -
                             solution[static_cast<Vector::Index>(face.lower_cell)]);
        residual[static_cast<Vector::Index>(face.upper_cell)] += flux;
        residual[static_cast<Vector::Index>(face.lower_cell)] -= flux;
    }
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
    for (const NonmatchingColumnCell& cell : grid_->cells) {
        residual[static_cast<Vector::Index>(cell.index)] -=
            heat_production_[cell.index] * cell.area();
    }
}

void NonmatchingColumnHeatSystem::assemble_matrix(const Vector& solution,
                                                  SparseMatrix& matrix) const {
    update_properties(solution);
    const NonmatchingColumnFluxes vertical = vertical_flux_.compute(
        flux_scheme_, thermal_conductivity_,
        {solution.native().data(), static_cast<std::size_t>(solution.size())}, boundary_data_);
    std::vector<MatrixTriplet> triplets;
    triplets.reserve(grid_->cells.size() + 4 * grid_->horizontal_faces.size() +
                     10 * vertical.vertical.size());
    for (const NonmatchingColumnInternalFlux& flux : vertical.vertical) {
        for (const MultipointFluxTerm term : flux.expression.terms) {
            triplets.emplace_back(static_cast<Vector::Index>(flux.left_cell),
                                  static_cast<Vector::Index>(term.cell), term.coefficient);
        }
        for (const MultipointFluxTerm term : flux.expression.terms) {
            triplets.emplace_back(static_cast<Vector::Index>(flux.right_cell),
                                  static_cast<Vector::Index>(term.cell), -term.coefficient);
        }
    }
    for (const NonmatchingColumnBoundaryFlux& flux : vertical.surface) {
        for (const MultipointFluxTerm term : flux.expression.terms) {
            triplets.emplace_back(static_cast<Vector::Index>(flux.cell),
                                  static_cast<Vector::Index>(term.cell), term.coefficient);
        }
    }
    for (const NonmatchingHorizontalFace& face : grid_->horizontal_faces) {
        const double value = horizontal_transmissibility(face);
        triplets.emplace_back(static_cast<Vector::Index>(face.upper_cell),
                              static_cast<Vector::Index>(face.upper_cell), value);
        triplets.emplace_back(static_cast<Vector::Index>(face.upper_cell),
                              static_cast<Vector::Index>(face.lower_cell), -value);
        triplets.emplace_back(static_cast<Vector::Index>(face.lower_cell),
                              static_cast<Vector::Index>(face.upper_cell), -value);
        triplets.emplace_back(static_cast<Vector::Index>(face.lower_cell),
                              static_cast<Vector::Index>(face.lower_cell), value);
    }
    for (const std::size_t cell_index : grid_->top_cells) {
        const NonmatchingColumnCell& cell = grid_->cells[cell_index];
        const double value =
            thermal_conductivity_[cell_index] * cell.width() / (0.5 * cell.height());
        triplets.emplace_back(static_cast<Vector::Index>(cell_index),
                              static_cast<Vector::Index>(cell_index), value);
    }
    matrix.resize(size(), size());
    matrix.set_from_triplets(triplets);
}
