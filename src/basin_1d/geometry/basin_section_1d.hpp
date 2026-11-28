#pragma once

#include "basin/basin_problem.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

struct SectionLayer1D {
    int id;
    int lithotype;
    double phi_top;
    double phi_bottom;
};

struct SectionConfig1D {
    double age_ma;
    double T_surf;
    std::vector<SectionLayer1D> layers;
    std::vector<double> contacts; // Depths [m], top to bottom; absent layers have zero length.
};

struct SectionEpoch1D {
    std::vector<double> start_contacts, end_contacts;
    std::vector<std::size_t> rows; // Number of intervals per layer in the ending configuration.
    std::optional<int> born_layer_id;
    double dz_min;

    [[nodiscard]] std::vector<double> faces_at(double theta) const;
};

struct BasinSection1D {
    std::vector<SectionConfig1D> configs;
    std::vector<SectionEpoch1D> epochs;
};

[[nodiscard]] BasinSection1D make_basin_section_1d(std::span<const BasinConfig> configs, double x,
                                                   double dz, double dz_min);
