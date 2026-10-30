#pragma once

#include <string>
#include <vector>

struct ScalarProfile {
    std::vector<double> x;
    std::vector<double> values;

    [[nodiscard]] double at(double coordinate) const;
};

struct BasinLayerProfile {
    int id = 0;
    std::string name;
    std::vector<double> x;
    std::vector<double> top;
    std::vector<double> bottom;
    std::vector<int> lithotype;
    std::vector<double> porosity;

    [[nodiscard]] double top_at(double coordinate) const;
    [[nodiscard]] double bottom_at(double coordinate) const;
    [[nodiscard]] double porosity_at(double coordinate) const;
    [[nodiscard]] int lithotype_at(double coordinate) const;
};

struct BasinConfiguration {
    double age_ma = 0.0;
    std::vector<BasinLayerProfile> layers;
    ScalarProfile surface_temperature;
    ScalarProfile basal_heat_flux;

    [[nodiscard]] const BasinLayerProfile& layer(int id) const;
};

struct EasyRoInput {
    double preexponential = 0.0;
    double gas_constant = 0.0;
    std::vector<double> activation_energies;
    std::vector<double> weights;
};
