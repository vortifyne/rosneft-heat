#include "inverse/model_heat_observation_data.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

ModelHeatObservationData::ModelHeatObservationData(Vector observed_temperatures,
                                                   Vector temperature_error_bounds)
    : observed_temperatures_(std::move(observed_temperatures)),
      temperature_error_bounds_(std::move(temperature_error_bounds)) {
    if (observed_temperatures_.empty()) {
        throw std::invalid_argument("Model heat observations must not be empty");
    }
    if (observed_temperatures_.size() != temperature_error_bounds_.size()) {
        throw std::invalid_argument(
            "Observed temperatures and temperature error bounds must have the same size");
    }
    if (!observed_temperatures_.all_finite()) {
        throw std::invalid_argument("Observed temperatures must be finite");
    }
    for (Vector::Index index = 0; index < temperature_error_bounds_.size(); ++index) {
        const double error_bound = temperature_error_bounds_[index];
        if (!(error_bound > 0.0) || !std::isfinite(error_bound)) {
            throw std::invalid_argument("Temperature error bounds must be finite and positive");
        }
    }
}

Vector::Index ModelHeatObservationData::size() const noexcept {
    return observed_temperatures_.size();
}

const Vector& ModelHeatObservationData::observed_temperatures() const noexcept {
    return observed_temperatures_;
}

const Vector& ModelHeatObservationData::temperature_error_bounds() const noexcept {
    return temperature_error_bounds_;
}

Vector ModelHeatObservationData::normalized_residuals(const Vector& calculated_temperatures) const {
    if (calculated_temperatures.size() != size()) {
        throw std::invalid_argument(
            "Calculated and observed temperature vectors must have the same size");
    }
    if (!calculated_temperatures.all_finite()) {
        throw std::invalid_argument("Calculated temperatures must be finite");
    }

    Vector residuals(size());
    for (Vector::Index index = 0; index < size(); ++index) {
        residuals[index] = (calculated_temperatures[index] - observed_temperatures_[index]) /
                           temperature_error_bounds_[index];
        if (!std::isfinite(residuals[index])) {
            throw std::overflow_error("Normalized temperature residual must be finite");
        }
    }
    return residuals;
}
