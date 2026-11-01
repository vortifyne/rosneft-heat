#pragma once

#include <span>

struct LithotypeProperties {
    int code;
    double rho_s;    // Solid density [kg/m^3].
    double lambda20; // Solid conductivity at 20 C [W/(m*K)].
    double c20;      // Solid specific heat at 20 C [J/(kg*K)].
    double A_s;      // Solid heat production [W/m^3].
};

struct FluidProperties {
    double lambda; // Thermal conductivity [W/(m*K)].
    double rho;    // Density [kg/m^3].
    double c;      // Specific heat [J/(kg*K)].
};

struct ThermalProperties {
    double lambda; // Thermal conductivity [W/(m*K)].
    double rho_c;  // Volumetric heat capacity [J/(m^3*K)].
    double A;      // Radiogenic heat production [W/m^3].
};

[[nodiscard]] double waples_heat_capacity(double T, double c20);

[[nodiscard]] double sekiguchi_conductivity(double T, double lambda20);

[[nodiscard]] ThermalProperties thermal_properties(double T, double phi,
                                                   const LithotypeProperties& lithotype,
                                                   const FluidProperties& fluid);

void update_thermal_fields(std::span<const double> T, std::span<const double> phi,
                           std::span<const LithotypeProperties* const> lithotypes,
                           const FluidProperties& fluid, std::span<double> lambda,
                           std::span<double> rho_c, std::span<double> A);
