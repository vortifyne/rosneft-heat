#include "basin/physics/thermal_properties.hpp"

#include <cmath>

double waples_heat_capacity(double T, double c20) {
    const double T_C = T - 273.15;
    const double factor =
        0.953 + (2.29e-3 * T_C) - (2.835e-6 * T_C * T_C) + (1.191e-9 * T_C * T_C * T_C);
    return c20 * factor;
}

double sekiguchi_conductivity(double T, double lambda20) {
    return 1.84 + (358.0 * ((1.0227 * lambda20) - 1.882) * ((1.0 / T) - 0.00068));
}

ThermalProperties thermal_properties(double T, double phi, const LithotypeProperties& lithotype,
                                     const FluidProperties& fluid) {
    const double lambda_s = sekiguchi_conductivity(T, lithotype.lambda20);
    const double c_s = waples_heat_capacity(T, lithotype.c20);
    const double f_s = 1.0 - phi;

    return {
        .lambda = std::pow(lambda_s, f_s) * std::pow(fluid.lambda, phi),
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
