#pragma once

#include "discretization/nonmatching_column_flux.hpp"
#include "linear/sparse_matrix.hpp"
#include "linear/vector.hpp"

#include <functional>
#include <span>
#include <vector>

struct NonmatchingColumnEnergyRates {
    double surface_outflow = 0.0;
    double basal_inflow = 0.0;
    double lateral_inflow = 0.0;
    double heat_production = 0.0;
};

struct NonmatchingColumnFluxDiagnostics {
    std::size_t maximum_stencil_size = 0;
    double internal_conservation_defect_linf = 0.0;
};

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
    [[nodiscard]] std::span<const double> thermal_conductivity() const noexcept;
    [[nodiscard]] std::span<const double> heat_production() const noexcept;

    void assemble_residual(const Vector& solution, Vector& residual) const;
    void assemble_matrix(const Vector& solution, SparseMatrix& matrix) const;

    [[nodiscard]] NonmatchingColumnEnergyRates energy_rates(const Vector& solution) const;
    [[nodiscard]] NonmatchingColumnFluxDiagnostics flux_diagnostics(const Vector& solution) const;

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
