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
    std::size_t temperatures = 0;
    for (const auto& point : points_) {
        temperatures += point.kind == BasinObservable::temperature;
        if (!std::isfinite(point.x) || !std::isfinite(point.z) || !std::isfinite(point.value) ||
            !std::isfinite(point.error) || !(point.error > 0.0))
            throw std::invalid_argument(
                "Observation coordinates, values and positive errors must be finite");
    }
    temperature_scale_ = std::sqrt(static_cast<double>(temperatures));
    reflectance_scale_ = std::sqrt(static_cast<double>(points_.size() - temperatures));
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

double BasinHeatObservationData::group_scale(BasinObservable kind) const noexcept {
    return kind == BasinObservable::temperature ? temperature_scale_ : reflectance_scale_;
}

Vector BasinHeatObservationData::residuals_in_error_units(Vector weighted) const {
    if (weighted.size() < size())
        throw std::invalid_argument("Missing observation residuals");
    weighted.native().conservativeResize(size());
    for (Vector::Index i = 0; i < weighted.size(); ++i) {
        const auto& point = points()[static_cast<std::size_t>(i)];
        weighted[i] *= group_scale(point.kind);
    }
    return weighted;
}

LeastSquaresEvaluation
BasinHeatObservationData::evaluate_predictions(Vector residuals, double tolerance,
                                               Eigen::MatrixXd* jacobian) const {
    if (residuals.size() != size() || (jacobian && jacobian->rows() != residuals.size()))
        throw std::runtime_error("Unexpected basin observation or Jacobian dimensions");
    bool within_errors = tolerance > 0.0;
    for (Vector::Index i = 0; i < residuals.size(); ++i) {
        const auto& point = points()[static_cast<std::size_t>(i)];
        const double residual = residuals[i] - point.value;
        within_errors = within_errors && std::abs(residual) <= tolerance * point.error;
        const double scale = point.error * group_scale(point.kind);
        residuals[i] = residual / scale;
        if (jacobian)
            jacobian->row(i) /= scale;
    }
    return {std::move(residuals), within_errors};
}
