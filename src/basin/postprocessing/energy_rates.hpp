#pragma once

#include <span>

struct ColGrid;
struct HeatBoundaryData;

// Rates per unit out-of-plane width [W/m].
struct EnergyRates {
    double surface_outflow = 0.0;
    double basal_inflow = 0.0;
    double lateral_inflow = 0.0;
    double source = 0.0;
};

// Coefficient fields are evaluated at the supplied temperature by the caller.
[[nodiscard]] EnergyRates energy_rates(const ColGrid& grid, std::span<const double> T,
                                       std::span<const double> lambda, std::span<const double> A,
                                       const HeatBoundaryData& bc);
