#pragma once

#include "basin/discretization/nonmatching_column_flux.hpp"
#include "numerics/algebra/sparse_matrix.hpp"
#include "numerics/algebra/vector.hpp"

#include <functional>
#include <span>
#include <vector>

class NonmatchingColumnHeatSystem {
public:
    using PropertyUpdater =
        std::function<void(std::span<const double>, std::span<double>, std::span<double>)>;

    NonmatchingColumnHeatSystem(const NonmatchingColumnGrid& grid,
                                std::span<const double> thermal_conductivity,
                                std::span<const double> heat_production,
                                NonmatchingColumnBoundaryData boundary_data,
                                PropertyUpdater property_updater = {});

    void set_flux_scheme(MultipointFluxScheme scheme) noexcept;
    [[nodiscard]] Vector::Index size() const noexcept;

    void assemble_residual(const Vector& solution, Vector& residual) const;
    void assemble_matrix(const Vector& solution, SparseMatrix& matrix) const;

private:
    void update_properties(const Vector& solution) const;
    [[nodiscard]] double horizontal_transmissibility(const NonmatchingHorizontalFace& face) const;

    const NonmatchingColumnGrid* grid_ = nullptr;
    mutable std::vector<double> thermal_conductivity_;
    mutable std::vector<double> heat_production_;
    NonmatchingColumnBoundaryData boundary_data_;
    PropertyUpdater property_updater_;
    NonmatchingColumnFlux vertical_flux_;
    MultipointFluxScheme flux_scheme_ = MultipointFluxScheme::linear;
};
