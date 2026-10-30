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

BasinConfig& find_configuration(std::vector<BasinConfig>& configs, const double age) {
    const auto found = std::find_if(configs.begin(), configs.end(), [age](const auto& c) {
        return std::abs(c.age_ma - age) < 1.0e-10;
    });
    if (found == configs.end()) {
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
        result.configs_.push_back(
            {.age_ma = std::stod(fields[0]), .layers = {}, .T_surf = {}, .q_basal = {}});
    });

    read_csv(directory / "layers.csv",
             "age_ma,layer_id,layer_name,x,z_top,z_bottom,lithotype,porosity",
             [&](const auto& fields) {
                 if (fields.size() != 8) {
                     throw std::runtime_error("Invalid layer profile row");
                 }
                 auto& cfg = find_configuration(result.configs_, std::stod(fields[0]));
                 const int layer_id = std::stoi(fields[1]);
                 auto found =
                     std::find_if(cfg.layers.begin(), cfg.layers.end(),
                                  [layer_id](const auto& layer) { return layer.id == layer_id; });
                 if (found == cfg.layers.end()) {
                     cfg.layers.push_back({.id = layer_id,
                                           .name = fields[2],
                                           .x = {},
                                           .top = {},
                                           .bottom = {},
                                           .lithotype = {},
                                           .phi = {}});
                     found = std::prev(cfg.layers.end());
                 }
                 found->x.push_back(std::stod(fields[3]));
                 found->top.push_back(std::stod(fields[4]));
                 found->bottom.push_back(std::stod(fields[5]));
                 found->lithotype.push_back(std::stoi(fields[6]));
                 found->phi.push_back(std::stod(fields[7]));
             });

    read_csv(directory / "boundaries.csv", "age_ma,kind,x,value_si", [&](const auto& fields) {
        if (fields.size() != 4) {
            throw std::runtime_error("Invalid boundary profile row");
        }
        auto& cfg = find_configuration(result.configs_, std::stod(fields[0]));
        ScalarProfile* profile = nullptr;
        if (fields[1] == "surface_temperature") {
            profile = &cfg.T_surf;
        } else if (fields[1] == "basal_heat_flux") {
            profile = &cfg.q_basal;
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
                 result.lithotypes_.emplace(code, LithotypeProperties{
                                                      .code = code,
                                                      .rho_s = std::stod(fields[2]),
                                                      .lambda20 = std::stod(fields[3]),
                                                      .c20 = std::stod(fields[4]),
                                                      .A_s = std::stod(fields[5]),
                                                  });
             });

    read_csv(directory / "kinetics.csv", "preexponential,activation_energy,weight,gas_constant",
             [&](const auto& fields) {
                 if (fields.size() != 4) {
                     throw std::runtime_error("Invalid kinetics row");
                 }
                 const double A = std::stod(fields[0]);
                 const double Rg = std::stod(fields[3]);
                 if (result.easy_ro_.E.empty()) {
                     result.easy_ro_.A = A;
                     result.easy_ro_.Rg = Rg;
                 } else if (A != result.easy_ro_.A || Rg != result.easy_ro_.Rg) {
                     throw std::runtime_error("Kinetics constants must be common to all reactions");
                 }
                 result.easy_ro_.E.push_back(std::stod(fields[1]));
                 result.easy_ro_.f.push_back(std::stod(fields[2]));
             });
    return result;
}

const LithotypeProperties& BasinInput::lithotype(const int code) const {
    return lithotypes_.at(code);
}

std::span<const BasinConfig> BasinInput::configs() const noexcept {
    return configs_;
}

const EasyRoParameters& BasinInput::easy_ro() const noexcept {
    return easy_ro_;
}
