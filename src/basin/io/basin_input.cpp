#include "basin/io/basin_input.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

template <typename Consume>
void read_csv(const std::filesystem::path& path, const std::string& expected_header,
              Consume consume) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open input file: " + path.string());
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Input file is empty: " + path.string());
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    if (line != expected_header) {
        throw std::runtime_error("Unexpected header in " + path.string());
    }
    std::size_t row = 1;
    while (std::getline(input, line)) {
        ++row;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            try {
                consume(split_csv_line(line));
            } catch (const std::exception& error) {
                throw std::runtime_error(path.string() + ":" + std::to_string(row) + ": " +
                                         error.what());
            }
        }
    }
}

BasinConfiguration& find_configuration(std::vector<BasinConfiguration>& configurations,
                                       const double age) {
    const auto found =
        std::find_if(configurations.begin(), configurations.end(),
                     [age](const auto& c) { return std::abs(c.age_ma - age) < 1.0e-10; });
    if (found == configurations.end()) {
        throw std::runtime_error("Input row refers to an unknown configuration");
    }
    return *found;
}

} // namespace

BasinInput BasinInput::read(const std::filesystem::path& directory) {
    BasinInput result;
    read_csv(directory / "configurations.csv", "age_ma", [&](const auto& fields) {
        if (fields.size() != 1) {
            throw std::runtime_error("Invalid configuration row");
        }
        result.configurations_.push_back({.age_ma = std::stod(fields[0]),
                                          .layers = {},
                                          .surface_temperature = {},
                                          .basal_heat_flux = {}});
    });

    read_csv(
        directory / "layers.csv", "age_ma,layer_id,layer_name,x,z_top,z_bottom,lithotype,porosity",
        [&](const auto& fields) {
            if (fields.size() != 8) {
                throw std::runtime_error("Invalid layer profile row");
            }
            auto& configuration = find_configuration(result.configurations_, std::stod(fields[0]));
            const int layer_id = std::stoi(fields[1]);
            auto found =
                std::find_if(configuration.layers.begin(), configuration.layers.end(),
                             [layer_id](const auto& layer) { return layer.id == layer_id; });
            if (found == configuration.layers.end()) {
                configuration.layers.push_back({.id = layer_id,
                                                .name = fields[2],
                                                .x = {},
                                                .top = {},
                                                .bottom = {},
                                                .lithotype = {},
                                                .porosity = {}});
                found = std::prev(configuration.layers.end());
            }
            found->x.push_back(std::stod(fields[3]));
            found->top.push_back(std::stod(fields[4]));
            found->bottom.push_back(std::stod(fields[5]));
            found->lithotype.push_back(std::stoi(fields[6]));
            found->porosity.push_back(std::stod(fields[7]));
        });

    read_csv(directory / "boundaries.csv", "age_ma,kind,x,value_si", [&](const auto& fields) {
        if (fields.size() != 4) {
            throw std::runtime_error("Invalid boundary profile row");
        }
        auto& configuration = find_configuration(result.configurations_, std::stod(fields[0]));
        ScalarProfile* profile = nullptr;
        if (fields[1] == "surface_temperature") {
            profile = &configuration.surface_temperature;
        } else if (fields[1] == "basal_heat_flux") {
            profile = &configuration.basal_heat_flux;
        } else {
            throw std::runtime_error("Unknown boundary profile kind");
        }
        profile->x.push_back(std::stod(fields[2]));
        profile->values.push_back(std::stod(fields[3]));
    });

    read_csv(directory / "lithotypes.csv",
             "code,name,solid_density,conductivity_20,specific_heat_20,heat_production",
             [&](const auto& fields) {
                 if (fields.size() != 6) {
                     throw std::runtime_error("Invalid lithotype row");
                 }
                 const int code = std::stoi(fields[0]);
                 result.lithotypes_.emplace(code,
                                            LithotypeThermophysicalProperties{
                                                .code = code,
                                                .solid_density = std::stod(fields[2]),
                                                .conductivity_at_20_celsius = std::stod(fields[3]),
                                                .specific_heat_at_20_celsius = std::stod(fields[4]),
                                                .solid_heat_production = std::stod(fields[5]),
                                            });
             });

    read_csv(directory / "kinetics.csv", "preexponential,activation_energy,weight,gas_constant",
             [&](const auto& fields) {
                 if (fields.size() != 4) {
                     throw std::runtime_error("Invalid kinetics row");
                 }
                 const double preexponential = std::stod(fields[0]);
                 const double gas_constant = std::stod(fields[3]);
                 if (result.easy_ro_.activation_energies.empty()) {
                     result.easy_ro_.preexponential = preexponential;
                     result.easy_ro_.gas_constant = gas_constant;
                 } else if (preexponential != result.easy_ro_.preexponential ||
                            gas_constant != result.easy_ro_.gas_constant) {
                     throw std::runtime_error("Kinetics constants must be common to all reactions");
                 }
                 result.easy_ro_.activation_energies.push_back(std::stod(fields[1]));
                 result.easy_ro_.weights.push_back(std::stod(fields[2]));
             });
    return result;
}

const LithotypeThermophysicalProperties& BasinInput::lithotype(const int code) const {
    return lithotypes_.at(code);
}

std::span<const BasinConfiguration> BasinInput::configurations() const noexcept {
    return configurations_;
}

const EasyRoInput& BasinInput::easy_ro() const noexcept {
    return easy_ro_;
}
