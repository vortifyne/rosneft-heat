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
    std::vector<double> phi_top;    // Porosity at the layer top [-].
    std::vector<double> phi_bottom; // Porosity at the layer bottom [-].

    [[nodiscard]] double top_at(double coordinate) const;
    [[nodiscard]] double bottom_at(double coordinate) const;
    [[nodiscard]] double phi_at(double coordinate, double relative_depth) const;
    [[nodiscard]] int lithotype_at(double coordinate) const;
};

struct BasinConfig {
    double age_ma = 0.0;
    std::vector<BasinLayerProfile> layers;
    ScalarProfile T_surf;
    ScalarProfile q_basal;

    [[nodiscard]] const BasinLayerProfile& layer(int id) const;
};
