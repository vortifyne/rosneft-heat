#include "discretization/nonmatching_column_transient_system.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

void validate_capacity(std::span<const double> capacity, std::size_t cells) {
    if (capacity.size() != cells) {
        throw std::invalid_argument("C-grid heat capacity size does not match the grid");
    }
    for (double value : capacity) {
        if (!std::isfinite(value) || !(value > 0.0)) {
            throw std::invalid_argument("C-grid volumetric heat capacity must be positive");
        }
    }
}

} // namespace

NonmatchingColumnTransientSystem::NonmatchingColumnTransientSystem(
    const NonmatchingColumnGrid& grid, std::span<const double> thermal_conductivity,
    std::span<const double> heat_production, std::span<const double> volumetric_heat_capacity,
    NonmatchingColumnBoundaryData boundary_data,
    NonmatchingColumnHeatSystem::PropertyUpdater property_updater, CapacityUpdater capacity_updater)
    : grid_(&grid), spatial_(grid, thermal_conductivity, heat_production, std::move(boundary_data),
                             std::move(property_updater)),
      volumetric_heat_capacity_(volumetric_heat_capacity.begin(), volumetric_heat_capacity.end()),
      capacity_updater_(std::move(capacity_updater)) {
    validate_capacity(volumetric_heat_capacity_, grid.cells.size());
}

void NonmatchingColumnTransientSystem::set_flux_scheme(MultipointFluxScheme scheme) {
    spatial_.set_flux_scheme(scheme);
}

void NonmatchingColumnTransientSystem::update_capacity(double time, const Vector& solution) const {
    if (solution.size() != static_cast<Vector::Index>(grid_->cells.size())) {
        throw std::invalid_argument("C-grid solution size does not match the grid");
    }
    if (capacity_updater_) {
        capacity_updater_(time,
                          {solution.native().data(), static_cast<std::size_t>(solution.size())},
                          volumetric_heat_capacity_);
        validate_capacity(volumetric_heat_capacity_, grid_->cells.size());
    }
}

void NonmatchingColumnTransientSystem::assemble_residual(double time, const Vector& solution,
                                                         const Vector& solution_derivative,
                                                         Vector& residual) const {
    update_capacity(time, solution);
    if (solution_derivative.size() != solution.size()) {
        throw std::invalid_argument("C-grid temperature derivative size does not match");
    }
    spatial_.assemble_residual(solution, residual);
    for (const NonmatchingColumnCell& cell : grid_->cells) {
        const auto index = static_cast<Vector::Index>(cell.index);
        residual[index] +=
            cell.area() * volumetric_heat_capacity_[cell.index] * solution_derivative[index];
    }
}

void NonmatchingColumnTransientSystem::assemble_matrix(NonlinearMethod method, double time,
                                                       const Vector& solution,
                                                       const Vector& solution_derivative,
                                                       double derivative_shift,
                                                       SparseMatrix& matrix) const {
    if (method != NonlinearMethod::picard) {
        throw std::invalid_argument("C-grid transient system currently supports Picard only");
    }
    if (solution_derivative.size() != solution.size() || !std::isfinite(derivative_shift) ||
        derivative_shift < 0.0) {
        throw std::invalid_argument("Invalid C-grid derivative or BDF shift");
    }
    update_capacity(time, solution);
    spatial_.assemble_matrix(solution, matrix);
    if (matrix.storage_order() == SparseStorageOrder::csr) {
        auto& entries = matrix.native_csr();
        for (const NonmatchingColumnCell& cell : grid_->cells) {
            const auto index = static_cast<Vector::Index>(cell.index);
            entries.coeffRef(index, index) +=
                derivative_shift * cell.area() * volumetric_heat_capacity_[cell.index];
        }
        entries.makeCompressed();
    } else {
        auto& entries = matrix.native_csc();
        for (const NonmatchingColumnCell& cell : grid_->cells) {
            const auto index = static_cast<Vector::Index>(cell.index);
            entries.coeffRef(index, index) +=
                derivative_shift * cell.area() * volumetric_heat_capacity_[cell.index];
        }
        entries.makeCompressed();
    }
}

const NonmatchingColumnHeatSystem&
NonmatchingColumnTransientSystem::spatial_system() const noexcept {
    return spatial_;
}
