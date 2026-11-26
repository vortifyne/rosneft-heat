#include "basin/inverse/basin_heat_parameter_space.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>

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

std::pair<Vector::Index, double> flux_interval(const std::vector<double>& ages, double age) {
    if (!(age >= ages.back() && age <= ages.front()))
        throw std::out_of_range("Age outside the basal flux history");
    const auto younger = std::lower_bound(ages.begin() + 1, ages.end(), age, std::greater<>{});
    const auto older = younger - ages.begin() - 1;
    const double fraction = age == *younger ? 1.0 : (ages[older] - age) / (ages[older] - *younger);
    return {older, fraction};
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
    const auto [i, theta] = flux_interval(ages_ma_, age_ma);
    return std::lerp(flux_[i], flux_[i + 1], theta);
}

double BasinHeatParameterSpace::at(double age_ma, const Vector& normalized, Vector& dq_dz) const {
    const auto [i, theta] = flux_interval(ages_ma_, age_ma);
    const Vector flux = to_physical(normalized);
    Vector gradient(flux.size());
    gradient.set_zero();
    gradient[i + 1] = theta * (flux[i] - kMinimumFlux);
    // q(t) depends on q_i both directly and through q_{i+1}.
    double dq_dnode = std::lerp(1.0, normalized[i + 1], theta);
    for (auto j = i; j >= 0; --j) {
        const double previous_flux = j == 0 ? kMaximumFlux : flux[j - 1];
        gradient[j] = dq_dnode * (previous_flux - kMinimumFlux);
        dq_dnode *= normalized[j];
    }
    dq_dz = std::move(gradient);
    return std::lerp(flux[i], flux[i + 1], theta);
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
