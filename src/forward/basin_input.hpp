#pragma once

#include "mesh/mesh2d.hpp"
#include "physics/thermophysical_properties.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <unordered_map>
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
    std::vector<Mesh2D::Region> regions;
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

class BasinInput {
public:
    static BasinInput read(const std::filesystem::path& directory);

    [[nodiscard]] const BasinConfiguration& configuration(double age_ma) const;
    [[nodiscard]] const LithotypeThermophysicalProperties& lithotype(int code) const;
    [[nodiscard]] std::span<const BasinConfiguration> configurations() const noexcept;
    [[nodiscard]] const EasyRoInput& easy_ro() const noexcept;

private:
    std::vector<BasinConfiguration> configurations_;
    std::unordered_map<int, LithotypeThermophysicalProperties> lithotypes_;
    EasyRoInput easy_ro_;
};
