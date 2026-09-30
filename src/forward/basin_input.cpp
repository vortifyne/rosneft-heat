#include "forward/basin_input.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        if (!field.empty() && field.back() == '\r') {
            field.pop_back();
        }
        fields.push_back(field);
    }
    return fields;
}

std::vector<std::vector<std::string>> read_csv(const std::filesystem::path& path,
                                               const std::string& expected_header) {
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
    std::vector<std::vector<std::string>> rows;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            rows.push_back(split_csv_line(line));
        }
    }
    return rows;
}

BoundaryKind boundary_kind(const int value) {
    switch (value) {
    case 1:
        return BoundaryKind::top;
    case 2:
        return BoundaryKind::bottom;
    case 3:
        return BoundaryKind::left;
    case 4:
        return BoundaryKind::right;
    case 5:
        return BoundaryKind::interface;
    default:
        throw std::runtime_error("Unknown boundary kind in region file");
    }
}

std::vector<Mesh2D::Region> read_regions(const std::filesystem::path& path) {
    const auto rows = read_csv(path, "region_id,layer_id,x,z,edge_kind");
    std::vector<Mesh2D::Region> result;
    std::unordered_map<int, std::size_t> positions;
    for (const auto& fields : rows) {
        if (fields.size() != 5) {
            throw std::runtime_error("Invalid region row in " + path.string());
        }
        const int region_id = std::stoi(fields[0]);
        const int layer_id = std::stoi(fields[1]);
        const auto [position, inserted] = positions.emplace(region_id, result.size());
        if (inserted) {
            result.push_back({.vertices = {}, .edge_kinds = {}, .id = layer_id});
        }
        auto& region = result.at(position->second);
        if (region.id != layer_id) {
            throw std::runtime_error("Layer identifier changes inside a region");
        }
        region.vertices.push_back({std::stod(fields[2]), std::stod(fields[3])});
        region.edge_kinds.push_back(boundary_kind(std::stoi(fields[4])));
    }
    return result;
}

double interpolate(const std::span<const double> x, const std::span<const double> values,
                   const double coordinate) {
    if (x.empty() || x.size() != values.size()) {
        throw std::logic_error("Invalid scalar profile");
    }
    if (coordinate <= x.front()) {
        return values.front();
    }
    if (coordinate >= x.back()) {
        return values.back();
    }
    const auto upper = std::upper_bound(x.begin(), x.end(), coordinate);
    const std::size_t right = static_cast<std::size_t>(upper - x.begin());
    const std::size_t left = right - 1;
    const double fraction = (coordinate - x[left]) / (x[right] - x[left]);
    return ((1.0 - fraction) * values[left]) + (fraction * values[right]);
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

double ScalarProfile::at(const double coordinate) const {
    return interpolate(x, values, coordinate);
}

double BasinLayerProfile::top_at(const double coordinate) const {
    return interpolate(x, top, coordinate);
}

double BasinLayerProfile::bottom_at(const double coordinate) const {
    return interpolate(x, bottom, coordinate);
}

double BasinLayerProfile::porosity_at(const double coordinate) const {
    return interpolate(x, porosity, coordinate);
}

int BasinLayerProfile::lithotype_at(const double coordinate) const {
    if (x.empty() || x.size() != lithotype.size()) {
        throw std::logic_error("Invalid lithotype profile");
    }
    const auto found = std::lower_bound(x.begin(), x.end(), coordinate);
    if (found == x.begin()) {
        return lithotype.front();
    }
    if (found == x.end()) {
        return lithotype.back();
    }
    const std::size_t right = static_cast<std::size_t>(found - x.begin());
    const std::size_t left = right - 1;
    return coordinate - x[left] <= x[right] - coordinate ? lithotype[left] : lithotype[right];
}

const BasinLayerProfile& BasinConfiguration::layer(const int id) const {
    const auto found = std::find_if(layers.begin(), layers.end(),
                                    [id](const auto& value) { return value.id == id; });
    if (found == layers.end()) {
        throw std::out_of_range("Layer is absent from basin configuration");
    }
    return *found;
}

BasinInput BasinInput::read(const std::filesystem::path& directory) {
    BasinInput result;
    const auto configurations = read_csv(directory / "configurations.csv", "age_ma,regions_file");
    for (const auto& fields : configurations) {
        if (fields.size() != 2) {
            throw std::runtime_error("Invalid configuration row");
        }
        result.configurations_.push_back({.age_ma = std::stod(fields[0]),
                                          .regions = read_regions(directory / fields[1]),
                                          .layers = {},
                                          .surface_temperature = {},
                                          .basal_heat_flux = {}});
    }

    const auto layer_rows = read_csv(
        directory / "layers.csv", "age_ma,layer_id,layer_name,x,z_top,z_bottom,lithotype,porosity");
    for (const auto& fields : layer_rows) {
        if (fields.size() != 8) {
            throw std::runtime_error("Invalid layer profile row");
        }
        auto& configuration = find_configuration(result.configurations_, std::stod(fields[0]));
        const int layer_id = std::stoi(fields[1]);
        auto found = std::find_if(configuration.layers.begin(), configuration.layers.end(),
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
    }

    const auto boundary_rows = read_csv(directory / "boundaries.csv", "age_ma,kind,x,value_si");
    for (const auto& fields : boundary_rows) {
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
    }

    const auto lithotype_rows =
        read_csv(directory / "lithotypes.csv",
                 "code,name,solid_density,conductivity_20,specific_heat_20,heat_production");
    for (const auto& fields : lithotype_rows) {
        if (fields.size() != 6) {
            throw std::runtime_error("Invalid lithotype row");
        }
        const int code = std::stoi(fields[0]);
        result.lithotypes_.emplace(code, LithotypeThermophysicalProperties{
                                             .code = code,
                                             .solid_density = std::stod(fields[2]),
                                             .conductivity_at_20_celsius = std::stod(fields[3]),
                                             .specific_heat_at_20_celsius = std::stod(fields[4]),
                                             .solid_heat_production = std::stod(fields[5]),
                                         });
    }

    const auto kinetics = read_csv(directory / "kinetics.csv",
                                   "preexponential,activation_energy,weight,gas_constant");
    for (const auto& fields : kinetics) {
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
    }
    return result;
}

const BasinConfiguration& BasinInput::configuration(const double age_ma) const {
    const auto found =
        std::find_if(configurations_.begin(), configurations_.end(), [age_ma](const auto& value) {
            return std::abs(value.age_ma - age_ma) < 1.0e-10;
        });
    if (found == configurations_.end()) {
        throw std::out_of_range("Requested basin configuration is absent");
    }
    return *found;
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
