#include "physics/thermophysical_properties.hpp"

#include <cmath>
#include <stdexcept>

namespace {

void require_positive_finite(const double value, const char* message) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::invalid_argument(message);
    }
}

void require_nonnegative_finite(const double value, const char* message) {
    if (value < 0.0 || !std::isfinite(value)) {
        throw std::invalid_argument(message);
    }
}

void check_lithotype(const LithotypeThermophysicalProperties& lithotype) {
    require_positive_finite(lithotype.solid_density,
                            "Lithotype density must be finite and positive");
    require_positive_finite(lithotype.conductivity_at_20_celsius,
                            "Lithotype conductivity must be finite and positive");
    require_positive_finite(lithotype.specific_heat_at_20_celsius,
                            "Lithotype specific heat must be finite and positive");
    require_nonnegative_finite(lithotype.solid_heat_production,
                               "Lithotype heat production must be finite and non-negative");
}

void check_fluid(const FluidThermophysicalProperties& fluid) {
    require_positive_finite(fluid.thermal_conductivity,
                            "Fluid conductivity must be finite and positive");
    require_positive_finite(fluid.density, "Fluid density must be finite and positive");
    require_positive_finite(fluid.specific_heat, "Fluid specific heat must be finite and positive");
}

} // namespace

double waples_specific_heat(const double temperature, const double specific_heat_at_20_celsius) {
    require_positive_finite(temperature, "Temperature must be finite and positive");
    require_positive_finite(specific_heat_at_20_celsius,
                            "Reference specific heat must be finite and positive");
    const double temperature_celsius = temperature - 273.15;
    const double factor =
        0.953 + (2.29e-3 * temperature_celsius) -
        (2.835e-6 * temperature_celsius * temperature_celsius) +
        (1.191e-9 * temperature_celsius * temperature_celsius * temperature_celsius);
    const double result = specific_heat_at_20_celsius * factor;
    require_positive_finite(result, "Waples specific heat is not finite and positive");
    return result;
}

double sekiguchi_thermal_conductivity(const double temperature,
                                      const double conductivity_at_20_celsius) {
    require_positive_finite(temperature, "Temperature must be finite and positive");
    require_positive_finite(conductivity_at_20_celsius,
                            "Reference conductivity must be finite and positive");
    const double result = 1.84 + (358.0 * ((1.0227 * conductivity_at_20_celsius) - 1.882) *
                                  ((1.0 / temperature) - 0.00068));
    require_positive_finite(result, "Sekiguchi conductivity is not finite and positive");
    return result;
}

EffectiveThermophysicalProperties
evaluate_thermophysical_properties(const double temperature, const double porosity,
                                   const LithotypeThermophysicalProperties& lithotype,
                                   const FluidThermophysicalProperties& fluid) {
    if (!std::isfinite(porosity) || porosity < 0.0 || porosity > 1.0) {
        throw std::invalid_argument("Porosity must be finite and lie in [0, 1]");
    }
    check_lithotype(lithotype);
    check_fluid(fluid);

    const double solid_conductivity =
        sekiguchi_thermal_conductivity(temperature, lithotype.conductivity_at_20_celsius);
    const double solid_specific_heat =
        waples_specific_heat(temperature, lithotype.specific_heat_at_20_celsius);
    const double solid_fraction = 1.0 - porosity;

    return {
        .thermal_conductivity = std::pow(solid_conductivity, solid_fraction) *
                                std::pow(fluid.thermal_conductivity, porosity),
        .volumetric_heat_capacity =
            (solid_fraction * lithotype.solid_density * solid_specific_heat) +
            (porosity * fluid.density * fluid.specific_heat),
        .heat_production = solid_fraction * lithotype.solid_heat_production,
    };
}

void evaluate_thermophysical_property_fields(
    const std::span<const double> temperatures, const std::span<const double> porosities,
    const std::span<const LithotypeThermophysicalProperties* const> lithotypes,
    const FluidThermophysicalProperties& fluid, const std::span<double> thermal_conductivities,
    const std::span<double> volumetric_heat_capacities, const std::span<double> heat_productions) {
    const std::size_t size = temperatures.size();
    if (porosities.size() != size || lithotypes.size() != size ||
        thermal_conductivities.size() != size || volumetric_heat_capacities.size() != size ||
        heat_productions.size() != size) {
        throw std::invalid_argument("Thermophysical field sizes must match");
    }

    for (std::size_t index = 0; index < size; ++index) {
        if (lithotypes[index] == nullptr) {
            throw std::invalid_argument("Cell lithotype must not be null");
        }
        const EffectiveThermophysicalProperties properties = evaluate_thermophysical_properties(
            temperatures[index], porosities[index], *lithotypes[index], fluid);
        thermal_conductivities[index] = properties.thermal_conductivity;
        volumetric_heat_capacities[index] = properties.volumetric_heat_capacity;
        heat_productions[index] = properties.heat_production;
    }
}
