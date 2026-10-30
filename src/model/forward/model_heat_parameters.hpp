#pragma once

struct ModelHeatParameters {
    double surface_temperature; // Temperature at the upper boundary, K.
    // Basal heat flux, W/m^2: lambda * dT/dz = basal_heat_flux.
    double basal_heat_flux;
};
