#include "basin/inverse/basin_heat_observation_data.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

BasinHeatObservationData::BasinHeatObservationData(std::vector<BasinObservation> points)
    : points_(std::move(points)) {
    if (points_.empty())
        throw std::invalid_argument("At least one observation is required");
    for (const auto& point : points_) {
        if (!std::isfinite(point.x) || !std::isfinite(point.z) || !std::isfinite(point.value) ||
            !std::isfinite(point.error) || !(point.error > 0.0))
            throw std::invalid_argument(
                "Observation coordinates, values and positive errors must be finite");
    }
}

BasinHeatObservationData BasinHeatObservationData::read(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::string line;
    if (!std::getline(input, line) || line != "kind,well,x,z,value,error")
        throw std::runtime_error("Cannot read observation header: " + path.string());
    std::vector<BasinObservation> points;
    while (std::getline(input, line)) {
        if (line.empty())
            continue;
        std::istringstream row(line);
        std::string kind, well;
        std::getline(row, kind, ',');
        std::getline(row, well, ',');
        std::string values;
        std::getline(row, values);
        std::replace(values.begin(), values.end(), ',', ' ');
        std::istringstream numbers(values);
        BasinObservation point{.kind = BasinObservable::temperature,
                               .well = std::move(well),
                               .x = 0.0,
                               .z = 0.0,
                               .value = 0.0,
                               .error = 0.0};
        if (!(numbers >> point.x >> point.z >> point.value >> point.error))
            throw std::runtime_error("Invalid observation row: " + line);
        if (kind == "temperature")
            point.value += 273.15; // Input temperatures and their 2% errors are in °C.
        else if (kind == "reflectance")
            point.kind = BasinObservable::reflectance;
        else
            throw std::runtime_error("Unknown observable: " + kind);
        points.push_back(std::move(point));
    }
    return BasinHeatObservationData(std::move(points));
}

Vector::Index BasinHeatObservationData::size() const noexcept {
    return static_cast<Vector::Index>(points_.size());
}

std::span<const BasinObservation> BasinHeatObservationData::points() const noexcept {
    return points_;
}

Vector BasinHeatObservationData::normalized_residuals(const Vector& calculated) const {
    Vector residuals(size());
    for (Vector::Index i = 0; i < size(); ++i) {
        const auto& point = points_[static_cast<std::size_t>(i)];
        residuals[i] = (calculated[i] - point.value) / point.error;
    }
    return residuals;
}
