#include "basin/io/basin_parameters.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

BasinHeatParameterSpace read_basin_parameters(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::string line;
    if (!std::getline(input, line) ||
        (line != "age_ma,basal_heat_flux_mw_m2" && line != "age_ma,basal_heat_flux_mw_m2\r"))
        throw std::runtime_error("Cannot read parameter header: " + path.string());
    std::vector<double> ages, fluxes;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        row >> std::ws;
        if (row.eof())
            continue;
        char separator;
        double age, flux;
        if (!(row >> age >> separator >> flux) || separator != ',' || !(row >> std::ws).eof())
            throw std::runtime_error("Expected age_ma,basal_heat_flux_mw_m2: " + line);
        ages.push_back(age);
        fluxes.push_back(flux / 1000.0);
    }
    if (!input.eof())
        throw std::runtime_error("Cannot read parameters: " + path.string());
    return {std::move(ages), Vector(Eigen::Map<const Eigen::VectorXd>(
                                 fluxes.data(), static_cast<Vector::Index>(fluxes.size())))};
}

void write_basin_parameters(const std::filesystem::path& path,
                            const BasinHeatParameterSpace& space) {
    std::ofstream output(path);
    output << "age_ma,basal_heat_flux_mw_m2\n" << std::setprecision(17);
    for (Vector::Index i = 0; i < space.values().size(); ++i)
        output << space.ages_ma()[static_cast<std::size_t>(i)] << ',' << space.values()[i] * 1e3
               << '\n';
    output.close();
    if (!output)
        throw std::runtime_error("Cannot write parameters: " + path.string());
}
