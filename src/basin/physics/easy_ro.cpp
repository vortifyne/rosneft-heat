#include "basin/physics/easy_ro.hpp"

#include <cmath>
#include <utility>

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
    constexpr int substeps = 16;
    constexpr double gauss_offset = 0.5 / 1.7320508075688772935;
    for (std::size_t P = 0; P < T0.size(); ++P) {
        const double T_old = T0[P];
        const double T_new = T1[P];
        for (std::size_t i = 0; i < reaction_count(); ++i) {
            double dI = 0.0;
            for (int substep = 0; substep < substeps; ++substep) {
                const double center = (static_cast<double>(substep) + 0.5) / substeps;
                const double offset = gauss_offset / substeps;
                const double theta_a = center - offset;
                const double theta_b = center + offset;
                const double T_a = ((1.0 - theta_a) * T_old) + (theta_a * T_new);
                const double T_b = ((1.0 - theta_b) * T_old) + (theta_b * T_new);
                const double k_a = params_.A * std::exp(-beta_[i] / T_a);
                const double k_b = params_.A * std::exp(-beta_[i] / T_b);
                dI += 0.5 * dt / substeps * (k_a + k_b);
            }
            I[(P * reaction_count()) + i] += dI;
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
