#pragma once

#include <vector>

struct Lithotype {
    double thermal_conductivity; // Thermal conductivity, W/(m*K).
    double density;              // Density, kg/m^3.
    double specific_heat;        // Specific heat capacity, J/(kg*K).
    double heat_production;      // Volumetric heat production, W/m^3.
};

struct InitialCondition {
    double initial_temperature; // Initial temperature throughout the domain, K.
};

struct ObservationPoint {
    double x;
    double z;
};

struct ModelHeatProblem {
    Lithotype lithotype;
    InitialCondition initial_condition;
    std::vector<ObservationPoint> observation_points;
};
