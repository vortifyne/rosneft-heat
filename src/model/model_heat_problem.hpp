#pragma once

#include <vector>

struct Lithotype {
    double thermal_conductivity;  // Теплопроводность, Вт/(м·К).
    double density;               // Плотность, кг/м³.
    double specific_heat;         // Удельная теплоёмкость, Дж/(кг·К).
    double heat_production;       // Объёмное тепловыделение, Вт/м³.
};

struct InitialBoundaryConditions {
    double initial_temperature;  // Начальная температура во всей области, К.
    double surface_temperature;  // Температура на верхней границе, К.
    // Базальный тепловой поток, Вт/м²: lambda * dT/dz = basal_heat_flux.
    double basal_heat_flux;
    // На боковых границах задано условие непротекания: dT/dn = 0.
};

struct ObservationPoint {
    double x;
    double z;
};

struct ModelHeatProblem {
    Lithotype lithotype;
    InitialBoundaryConditions initial_boundary_conditions;
    std::vector<ObservationPoint> observation_points;
};
