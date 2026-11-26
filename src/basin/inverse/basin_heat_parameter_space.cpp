#include "basin/inverse/basin_heat_parameter_space.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace {
constexpr double kMinimumFlux = 0.05; // W/m².
constexpr double kMaximumFlux = 0.10;
void validate_flux(const Vector& flux, std::size_t count) {
    if (static_cast<std::size_t>(flux.size()) != count || !flux.all_finite())
        throw std::invalid_argument("Expected one finite basal flux per knot");
    double previous = kMaximumFlux;
    for (Vector::Index i = 0; i < flux.size(); ++i) {
        if (flux[i] < kMinimumFlux || flux[i] > previous)
            throw std::invalid_argument("Basal flux must be nonincreasing within [50, 100] mW/m2");
        previous = flux[i];
    }
}
} // namespace

BasinHeatParameterSpace::BasinHeatParameterSpace(std::vector<double> ages_ma, Vector flux)
    : ages_ma_(std::move(ages_ma)) {
    if (ages_ma_.size() < 2)
        throw std::invalid_argument("Expected at least two basal flux knots");
    for (std::size_t i = 0; i < ages_ma_.size(); ++i)
        if (!std::isfinite(ages_ma_[i]) || (i > 0 && ages_ma_[i] >= ages_ma_[i - 1]))
            throw std::invalid_argument("Basal flux ages must be finite and strictly decreasing");
    set_values(std::move(flux));
}

void BasinHeatParameterSpace::set_values(Vector flux) {
    validate_flux(flux, ages_ma_.size());
    flux_ = std::move(flux);
}

double BasinHeatParameterSpace::at(double age_ma) const {
    if (!(age_ma >= ages_ma_.back() && age_ma <= ages_ma_.front()))
        throw std::out_of_range("Age outside the basal flux history");
    const auto younger =
        std::lower_bound(ages_ma_.begin(), ages_ma_.end(), age_ma, std::greater<>{});
    const auto i = younger - ages_ma_.begin();
    if (*younger == age_ma)
        return flux_[i];
    const double theta = (ages_ma_[i - 1] - age_ma) / (ages_ma_[i - 1] - *younger);
    return std::lerp(flux_[i - 1], flux_[i], theta);
}

Vector BasinHeatParameterSpace::to_normalized(const Vector& parameters) const {
    validate_flux(parameters, ages_ma_.size());
    Vector normalized(flux_.size());
    double previous = kMaximumFlux;
    for (Vector::Index i = 0; i < flux_.size(); ++i) {
        normalized[i] = previous == kMinimumFlux
                            ? 1.0
                            : (parameters[i] - kMinimumFlux) / (previous - kMinimumFlux);
        previous = parameters[i];
    }
    return normalized;
}

Vector BasinHeatParameterSpace::to_physical(const Vector& normalized) const {
    if (normalized.size() != flux_.size() || !normalized.all_finite() ||
        (normalized.native().array() < 0.0).any() || (normalized.native().array() > 1.0).any())
        throw std::invalid_argument("Normalized basal flux parameters must lie in [0, 1]");
    Vector parameters(flux_.size());
    double previous = kMaximumFlux;
    for (Vector::Index i = 0; i < flux_.size(); ++i) {
        parameters[i] = std::lerp(kMinimumFlux, previous, normalized[i]);
        previous = parameters[i];
    }
    return parameters;
}

Vector BasinHeatParameterSpace::normalized_lower_bounds() const {
    Vector bounds(flux_.size());
    bounds.set_zero();
    return bounds;
}

Vector BasinHeatParameterSpace::normalized_upper_bounds() const {
    Vector bounds(flux_.size());
    bounds.set_constant(1.0);
    return bounds;
}
