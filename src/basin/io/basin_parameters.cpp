#include "basin/io/basin_parameters.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

Vector read_basin_parameters(const std::filesystem::path& path,
                             std::span<const BasinConfig> configs) {
    std::ifstream input(path);
    std::string line;
    if (!std::getline(input, line) || line != "age_ma,basal_heat_flux_mw_m2")
        throw std::runtime_error("Cannot read parameter header: " + path.string());
    Vector parameters(static_cast<Vector::Index>(configs.size()));
    std::size_t index = 0;
    while (std::getline(input, line)) {
        if (line.empty())
            continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        double age = 0.0, flux = 0.0;
        if (!(row >> age >> flux) || index >= configs.size() ||
            std::abs(age - configs[index].age_ma) > 1e-8)
            throw std::runtime_error("Parameters must follow the configuration ages");
        parameters[static_cast<Vector::Index>(index++)] = flux * 1e-3;
    }
    if (index != configs.size())
        throw std::runtime_error("Expected one basal flux per configuration");
    return parameters;
}

void write_basin_parameters(const std::filesystem::path& path, std::span<const BasinConfig> configs,
                            const Vector& parameters) {
    std::ofstream output(path);
    output << "age_ma,basal_heat_flux_mw_m2\n" << std::setprecision(17);
    for (std::size_t i = 0; i < configs.size(); ++i)
        output << configs[i].age_ma << ',' << parameters[static_cast<Vector::Index>(i)] * 1e3
               << '\n';
    output.close();
    if (!output)
        throw std::runtime_error("Cannot write parameters: " + path.string());
}
