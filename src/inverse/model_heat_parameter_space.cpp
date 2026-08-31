#include "inverse/model_heat_parameter_space.hpp"

#include <cmath>
#include <stdexcept>

namespace {

constexpr Vector::Index kSurfaceTemperatureIndex = 0;
constexpr Vector::Index kBasalHeatFluxIndex = 1;

void validate_bounds(double lower, double upper, const char* message) {
    const double span = upper - lower;
    if (!std::isfinite(lower) || !std::isfinite(upper) || !(span > 0.0) || !std::isfinite(span)) {
        throw std::invalid_argument(message);
    }
}

void validate_physical_value(double value, double lower, double upper, const char* message) {
    if (!std::isfinite(value) || value < lower || value > upper) {
        throw std::invalid_argument(message);
    }
}

double normalize(double value, double lower, double upper) {
    return (value - lower) / (upper - lower);
}

double denormalize(double value, double lower, double upper) {
    return lower + (value * (upper - lower));
}

} // namespace

ModelHeatParameterSpace::ModelHeatParameterSpace(ModelHeatParameterBounds bounds)
    : bounds_(bounds) {
    validate_bounds(bounds_.lower.surface_temperature, bounds_.upper.surface_temperature,
                    "Surface temperature bounds must be finite and strictly increasing");
    validate_bounds(bounds_.lower.basal_heat_flux, bounds_.upper.basal_heat_flux,
                    "Basal heat flux bounds must be finite and strictly increasing");
}

Vector ModelHeatParameterSpace::to_normalized(const ModelHeatParameters& parameters) const {
    validate_physical_value(parameters.surface_temperature, bounds_.lower.surface_temperature,
                            bounds_.upper.surface_temperature,
                            "Surface temperature must lie within its physical bounds");
    validate_physical_value(parameters.basal_heat_flux, bounds_.lower.basal_heat_flux,
                            bounds_.upper.basal_heat_flux,
                            "Basal heat flux must lie within its physical bounds");

    Vector normalized(kParameterCount);
    normalized[kSurfaceTemperatureIndex] =
        normalize(parameters.surface_temperature, bounds_.lower.surface_temperature,
                  bounds_.upper.surface_temperature);
    normalized[kBasalHeatFluxIndex] = normalize(
        parameters.basal_heat_flux, bounds_.lower.basal_heat_flux, bounds_.upper.basal_heat_flux);
    return normalized;
}

ModelHeatParameters ModelHeatParameterSpace::to_physical(const Vector& normalized) const {
    if (normalized.size() != kParameterCount) {
        throw std::invalid_argument("Normalized model heat parameter vector must have size two");
    }
    if (!normalized.all_finite()) {
        throw std::invalid_argument("Normalized model heat parameters must be finite");
    }
    for (Vector::Index index = 0; index < normalized.size(); ++index) {
        if (normalized[index] < 0.0 || normalized[index] > 1.0) {
            throw std::invalid_argument("Normalized model heat parameters must lie in [0, 1]");
        }
    }

    return {
        .surface_temperature =
            denormalize(normalized[kSurfaceTemperatureIndex], bounds_.lower.surface_temperature,
                        bounds_.upper.surface_temperature),
        .basal_heat_flux =
            denormalize(normalized[kBasalHeatFluxIndex], bounds_.lower.basal_heat_flux,
                        bounds_.upper.basal_heat_flux),
    };
}

Vector ModelHeatParameterSpace::normalized_lower_bounds() const {
    return Vector{0.0, 0.0};
}

Vector ModelHeatParameterSpace::normalized_upper_bounds() const {
    return Vector{1.0, 1.0};
}

const ModelHeatParameterBounds& ModelHeatParameterSpace::physical_bounds() const noexcept {
    return bounds_;
}
