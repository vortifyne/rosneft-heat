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

Vector prefix_products(const Vector& eta, Vector::Index count) {
    if (eta.size() != count || !eta.all_finite() || (eta.native().array() < 0.0).any() ||
        (eta.native().array() > 1.0).any())
        throw std::invalid_argument("Normalized basal flux parameters must lie in [0, 1]");
    Vector prefix(count);
    double product = 1.0;
    for (Vector::Index i = 0; i < count; ++i) {
        product *= eta[i];
        prefix[i] = product;
    }
    return prefix;
}

double knot_flux(double product) {
    return kMinimumFlux + (kMaximumFlux - kMinimumFlux) * product;
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

double BasinHeatParameterSpace::at(double age_ma, const Vector& eta, Vector& dq_deta) const {
    const auto [i, theta] = flux_interval(ages_ma_, age_ma);
    const Vector prefix = prefix_products(eta, flux_.size());
    Vector gradient(eta.size());
    gradient.set_zero();
    constexpr double range = kMaximumFlux - kMinimumFlux;
    gradient[i + 1] = range * theta * prefix[i];
    // Exclude eta[j] by multiplying the factors on either side, including at eta[j] = 0.
    double suffix = std::lerp(1.0, eta[i + 1], theta);
    for (auto j = i; j >= 0; --j) {
        gradient[j] = range * (j == 0 ? 1.0 : prefix[j - 1]) * suffix;
        suffix *= eta[j];
    }
    dq_deta = std::move(gradient);
    return std::lerp(knot_flux(prefix[i]), knot_flux(prefix[i + 1]), theta);
}

Vector BasinHeatParameterSpace::to_normalized(const Vector& parameters) const {
    validate_flux(parameters, ages_ma_.size());
    Vector eta(flux_.size());
    double previous = kMaximumFlux;
    for (Vector::Index i = 0; i < flux_.size(); ++i) {
        eta[i] = previous == kMinimumFlux
                     ? 1.0
                     : (parameters[i] - kMinimumFlux) / (previous - kMinimumFlux);
        previous = parameters[i];
    }
    return eta;
}

Vector BasinHeatParameterSpace::to_physical(const Vector& eta, Eigen::MatrixXd* jacobian) const {
    Vector flux = prefix_products(eta, flux_.size());
    if (jacobian) {
        jacobian->setZero(eta.size(), eta.size());
        for (Vector::Index i = 0; i < eta.size(); ++i) {
            if (i > 0)
                jacobian->row(i) = eta[i] * jacobian->row(i - 1);
            (*jacobian)(i, i) = (kMaximumFlux - kMinimumFlux) * (i == 0 ? 1.0 : flux[i - 1]);
        }
    }
    for (Vector::Index i = 0; i < flux.size(); ++i)
        flux[i] = knot_flux(flux[i]);
    return flux;
}

Vector::Index BasinHeatParameterSpace::gradient_penalty_size(double regularization) const {
    if (!std::isfinite(regularization) || regularization < 0.0)
        throw std::invalid_argument("Flux gradient regularization must be finite and non-negative");
    return regularization == 0.0 ? 0 : flux_.size() - 1;
}

void BasinHeatParameterSpace::append_gradient_penalty(double regularization, const Vector& eta,
                                                      Vector& residuals,
                                                      Eigen::MatrixXd* jacobian) const {
    const auto intervals = gradient_penalty_size(regularization);
    if (intervals == 0)
        return;
    Eigen::MatrixXd dq;
    const Vector q = to_physical(eta, jacobian ? &dq : nullptr);
    const auto offset = residuals.size();
    residuals.native().conservativeResize(offset + intervals);
    if (jacobian)
        jacobian->conservativeResize(offset + intervals, Eigen::NoChange);
    const double span = ages_ma_.front() - ages_ma_.back();
    const double normalization = std::sqrt(regularization) / (kMaximumFlux - kMinimumFlux);
    for (Vector::Index i = 0; i < intervals; ++i) {
        const double dt = ages_ma_[i] - ages_ma_[i + 1];
        const double scale = normalization * std::sqrt(span / dt);
        residuals[offset + i] = scale * (q[i + 1] - q[i]);
        if (jacobian)
            jacobian->row(offset + i) = scale * (dq.row(i + 1) - dq.row(i));
    }
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
