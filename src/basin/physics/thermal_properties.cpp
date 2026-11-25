#include "basin/physics/thermal_properties.hpp"

#include <cmath>

double waples_heat_capacity(double T, double c20) {
    const double T_C = T - 273.15;
    const double factor =
        0.953 + (2.29e-3 * T_C) - (2.835e-6 * T_C * T_C) + (1.191e-9 * T_C * T_C * T_C);
    return c20 * factor;
}

double sekiguchi_conductivity(double T, double lambda20) {
    // Sekiguchi (1984), Eq. (8), referenced to the input conductivity at 20 °C.
    constexpr double T_m = 1473.0;
    constexpr double lambda_m = 4.4 * 0.419; // mcal/(cm·s·°C) to W/(m·K).
    return lambda_m + (lambda20 - lambda_m) * (1.0 / T - 1.0 / T_m) / (1.0 / 293.15 - 1.0 / T_m);
}

ThermalProperties thermal_properties(double T, double phi, const LithotypeProperties& lithotype,
                                     const FluidProperties& fluid) {
    const double c_s = waples_heat_capacity(T, lithotype.c20);
    const double f_s = 1.0 - phi;
    const double bulk_lambda20 = std::pow(lithotype.lambda20, f_s) * std::pow(fluid.lambda, phi);

    return {
        .lambda = sekiguchi_conductivity(T, bulk_lambda20),
        .rho_c = (f_s * lithotype.rho_s * c_s) + (phi * fluid.rho * fluid.c),
        .A = f_s * lithotype.A_s,
    };
}

void update_thermal_fields(std::span<const double> T, std::span<const double> phi,
                           std::span<const LithotypeProperties* const> lithotypes,
                           const FluidProperties& fluid, std::span<double> lambda,
                           std::span<double> rho_c, std::span<double> A) {
    const std::size_t size = T.size();
    for (std::size_t index = 0; index < size; ++index) {
        const ThermalProperties properties =
            thermal_properties(T[index], phi[index], *lithotypes[index], fluid);
        lambda[index] = properties.lambda;
        rho_c[index] = properties.rho_c;
        A[index] = properties.A;
    }
}
