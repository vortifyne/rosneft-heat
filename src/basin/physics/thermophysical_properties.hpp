#pragma once

#include <span>

struct LithotypeThermophysicalProperties {
    int code;
    double solid_density;
    double conductivity_at_20_celsius;
    double specific_heat_at_20_celsius;
    double solid_heat_production;
};

struct FluidThermophysicalProperties {
    double thermal_conductivity;
    double density;
    double specific_heat;
};

struct EffectiveThermophysicalProperties {
    double thermal_conductivity;
    double volumetric_heat_capacity;
    double heat_production;
};

[[nodiscard]] constexpr double celsius_to_kelvin(const double temperature) noexcept {
    return temperature + 273.15;
}

[[nodiscard]] constexpr double porosity_percent_to_fraction(const double porosity) noexcept {
    return 0.01 * porosity;
}

[[nodiscard]] constexpr double
microwatts_per_cubic_meter_to_watts_per_cubic_meter(const double heat_production) noexcept {
    return 1.0e-6 * heat_production;
}

[[nodiscard]] double waples_specific_heat(double temperature, double specific_heat_at_20_celsius);

[[nodiscard]] double sekiguchi_thermal_conductivity(double temperature,
                                                    double conductivity_at_20_celsius);

[[nodiscard]] EffectiveThermophysicalProperties
evaluate_thermophysical_properties(double temperature, double porosity,
                                   const LithotypeThermophysicalProperties& lithotype,
                                   const FluidThermophysicalProperties& fluid);

void evaluate_thermophysical_property_fields(
    std::span<const double> temperatures, std::span<const double> porosities,
    std::span<const LithotypeThermophysicalProperties* const> lithotypes,
    const FluidThermophysicalProperties& fluid, std::span<double> thermal_conductivities,
    std::span<double> volumetric_heat_capacities, std::span<double> heat_productions);
