#include "basin/inverse/basin_heat_parameter_space.hpp"

#include <stdexcept>

namespace {
constexpr double kMinimumFlux = 0.05; // W/m².
constexpr double kMaximumFlux = 0.10;
} // namespace

Vector BasinHeatParameterSpace::to_normalized(const Vector& parameters) const {
    if (parameters.size() != size_ || !parameters.all_finite())
        throw std::invalid_argument("Expected one finite basal flux per configuration");
    Vector normalized(size_);
    double previous = kMaximumFlux;
    for (Vector::Index i = 0; i < size_; ++i) {
        if (parameters[i] < kMinimumFlux || parameters[i] > previous)
            throw std::invalid_argument("Basal flux must decrease within [50, 100] mW/m2");
        normalized[i] = previous == kMinimumFlux
                            ? 1.0
                            : (parameters[i] - kMinimumFlux) / (previous - kMinimumFlux);
        previous = parameters[i];
    }
    return normalized;
}

Vector BasinHeatParameterSpace::to_physical(const Vector& normalized) const {
    if (normalized.size() != size_ || !normalized.all_finite() ||
        (normalized.native().array() < 0.0).any() || (normalized.native().array() > 1.0).any())
        throw std::invalid_argument("Normalized basal flux parameters must lie in [0, 1]");
    Vector parameters(size_);
    double previous = kMaximumFlux;
    for (Vector::Index i = 0; i < size_; ++i) {
        parameters[i] = kMinimumFlux + (previous - kMinimumFlux) * normalized[i];
        previous = parameters[i];
    }
    return parameters;
}

Vector BasinHeatParameterSpace::normalized_lower_bounds() const {
    Vector bounds(size_);
    bounds.set_zero();
    return bounds;
}

Vector BasinHeatParameterSpace::normalized_upper_bounds() const {
    Vector bounds(size_);
    bounds.set_constant(1.0);
    return bounds;
}
