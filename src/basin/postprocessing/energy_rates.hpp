#pragma once

#include "basin/discretization/multipoint_flux_types.hpp"

#include <span>

struct NonmatchingColumnGrid;
struct NonmatchingColumnBoundaryData;

struct NonmatchingColumnEnergyRates {
    double surface_outflow = 0.0;
    double basal_inflow = 0.0;
    double lateral_inflow = 0.0;
    double heat_production = 0.0;
};

// Coefficient fields are evaluated at the supplied temperature by the caller.
[[nodiscard]] NonmatchingColumnEnergyRates nonmatching_column_energy_rates(
    const NonmatchingColumnGrid& grid, std::span<const double> temperature,
    std::span<const double> thermal_conductivity, std::span<const double> heat_production,
    const NonmatchingColumnBoundaryData& boundary_data,
    MultipointFluxScheme scheme = MultipointFluxScheme::linear);
