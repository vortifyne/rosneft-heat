#include "basin/physics/easy_ro.hpp"

#include <array>
#include <cmath>
#include <utility>

namespace {

constexpr std::size_t kTableSize = 8192;
constexpr double kXMin = 20.0;
constexpr double kXMax = 200.0;
constexpr double kInverseStep = static_cast<double>(kTableSize - 1) / (kXMax - kXMin);

const std::array<double, kTableSize>& scaled_primitive_table() {
    // Store H(x) = x^2 * exp(x) * G(x), where G(x) = exp(-x)/x + Ei(-x).
    static const auto table = [] {
        std::array<double, kTableSize> result{};
        for (std::size_t i = 0; i < kTableSize; ++i) {
            const double x = kXMin + static_cast<double>(i) / kInverseStep;
            const double G = std::exp(-x) / x + std::expint(-x);
            result[i] = x * x * std::exp(x) * G;
        }
        return result;
    }();
    return table;
}

} // namespace

EasyRoModel::EasyRoModel(EasyRoParameters params) : params_(std::move(params)) {
    beta_.reserve(params_.E.size());
    for (double E_i : params_.E) {
        beta_.push_back(E_i / params_.Rg);
    }
}

std::size_t EasyRoModel::reaction_count() const noexcept {
    return beta_.size();
}

std::vector<double> EasyRoModel::initial_state(std::size_t cell_count) const {
    return std::vector<double>(cell_count * reaction_count(), 0.0);
}

void EasyRoModel::advance(std::span<const double> T0, std::span<const double> T1, double dt,
                          std::span<double> I) const {
    const auto& table = scaled_primitive_table();
    const auto H = [&table](double x) {
        const double position = (x - kXMin) * kInverseStep;
        const auto i = static_cast<std::size_t>(position);
        const double weight = position - static_cast<double>(i);
        return table[i] + weight * (table[i + 1] - table[i]);
    };
    const std::size_t reactions = reaction_count();
    for (std::size_t P = 0; P < T0.size(); ++P) {
        const double T_old = T0[P];
        const double T_new = T1[P];
        const double inverse_T_old = 1.0 / T_old;
        const double inverse_T_new = 1.0 / T_new;
        if (T_old == T_new) {
            for (std::size_t i = 0; i < reactions; ++i) {
                I[P * reactions + i] += params_.A * dt * std::exp(-beta_[i] * inverse_T_old);
            }
            continue;
        }
        const double multiplier = params_.A * dt / (T_new - T_old);
        const double square_old = T_old * T_old;
        const double square_new = T_new * T_new;
        for (std::size_t i = 0; i < reactions; ++i) {
            const double x_old = beta_[i] * inverse_T_old;
            const double x_new = beta_[i] * inverse_T_new;
            // Phi(T) = beta * G(x) = T^2 / beta * exp(-x) * H(x), x = beta / T.
            const double old_value = square_old * std::exp(-x_old) * H(x_old);
            const double new_value = square_new * std::exp(-x_new) * H(x_new);
            I[P * reactions + i] += multiplier / beta_[i] * (new_value - old_value);
        }
    }
}

void EasyRoModel::reflectance(std::span<const double> I, std::span<double> Ro) const {
    for (std::size_t P = 0; P < Ro.size(); ++P) {
        double F = 0.0;
        for (std::size_t i = 0; i < reaction_count(); ++i) {
            const double I_i = I[(P * reaction_count()) + i];
            F += params_.f[i] * (1.0 - std::exp(-I_i));
        }
        Ro[P] = std::exp(-1.6 + (3.7 * F));
    }
}
