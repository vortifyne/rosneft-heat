#pragma once

#include <vector>

struct Lithotype {
    double thermal_conductivity; // Теплопроводность, Вт/(м·К).
    double density;              // Плотность, кг/м³.
    double specific_heat;        // Удельная теплоёмкость, Дж/(кг·К).
    double heat_production;      // Объёмное тепловыделение, Вт/м³.
};

struct InitialCondition {
    double initial_temperature; // Начальная температура во всей области, К.
};

struct ModelHeatParameters {
    double surface_temperature; // Температура на верхней границе, К.
    // Базальный тепловой поток, Вт/м²: lambda * dT/dz = basal_heat_flux.
    double basal_heat_flux;
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
