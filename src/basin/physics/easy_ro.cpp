#include "basin/physics/easy_ro.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
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

std::vector<double> EasyRoModel::initial_state(std::size_t point_count) const {
    return std::vector<double>(point_count * reaction_count(), 0.0);
}

void EasyRoModel::advance(std::span<const double> T0, std::span<const double> T1, double t_old,
                          double t_new, std::span<const double> t_dep, std::span<double> I,
                          const EasyRoStepSensitivities* ddm) const {
    const auto reactions = reaction_count();
    if (T1.size() != T0.size() || t_dep.size() != T0.size() || I.size() != T0.size() * reactions)
        throw std::invalid_argument("Incompatible EasyRo temperature and reaction fields");
    if (ddm &&
        (ddm->S0.rows() != static_cast<Eigen::Index>(T0.size()) ||
         ddm->S1.rows() != ddm->S0.rows() || ddm->Z.rows() != static_cast<Eigen::Index>(I.size()) ||
         ddm->S1.cols() != ddm->S0.cols() || ddm->Z.cols() != ddm->S0.cols()))
        throw std::invalid_argument("Incompatible EasyRo sensitivity fields");

    const auto& table = scaled_primitive_table();
    const auto primitive = [&](double T, double x, double& derivative) {
        const double position = (x - kXMin) * kInverseStep;
        const auto i = static_cast<std::size_t>(position);
        const double diff = table[i + 1] - table[i];
        const double H = table[i] + (position - static_cast<double>(i)) * diff;
        const double rate = params_.A * std::exp(-x);
        // Phi(T) = A*T*exp(-x)*H(x)/x, x = beta/T.
        if (ddm)
            derivative = rate * ((1.0 + 2.0 / x) * H - diff * kInverseStep);
        return rate * T / x * H;
    };
    for (std::size_t P = 0; P < T0.size(); ++P) {
        const double begin = std::max(t_old, t_dep[P]);
        if (begin >= t_new)
            continue;
        const double dt = t_new - begin;
        const double theta = (begin - t_old) / (t_new - t_old);
        const double Ta = std::lerp(T0[P], T1[P], theta);
        const double Tb = T1[P];
        const double delta = Tb - Ta;
        const double Tmid = std::midpoint(Ta, Tb);
        const bool nearly_constant = std::abs(delta) <= 1e-6 * Tmid;

        for (std::size_t i = 0; i < reactions; ++i) {
            double increment, ca, cb;
            if (nearly_constant) {
                // The midpoint limit avoids subtracting nearly equal primitive values.
                increment = dt * params_.A * std::exp(-beta_[i] / Tmid);
                if (ddm)
                    ca = cb = 0.5 * increment * beta_[i] / (Tmid * Tmid);
            } else {
                double da, db;
                const double pa = primitive(Ta, beta_[i] / Ta, da);
                const double pb = primitive(Tb, beta_[i] / Tb, db);
                increment = dt * ((pb - pa) / delta);
                if (ddm) {
                    ca = (increment - dt * da) / delta;
                    cb = (dt * db - increment) / delta;
                }
            }
            const auto row = P * reactions + i;
            I[row] += increment;
            if (ddm)
                ddm->Z.row(row) +=
                    (ca * (1.0 - theta)) * ddm->S0.row(P) + (cb + ca * theta) * ddm->S1.row(P);
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

Eigen::MatrixXd EasyRoModel::reflectance_sensitivities(std::span<const double> I,
                                                       std::span<const double> Ro,
                                                       const Eigen::MatrixXd& Z) const {
    if (I.size() != Ro.size() * reaction_count() || Z.rows() != static_cast<Eigen::Index>(I.size()))
        throw std::invalid_argument("Incompatible EasyRo reflectance sensitivity fields");
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(Ro.size(), Z.cols());
    for (std::size_t P = 0; P < Ro.size(); ++P) {
        for (std::size_t i = 0; i < reaction_count(); ++i) {
            const auto row = P * reaction_count() + i;
            result.row(P) += (params_.f[i] * std::exp(-I[row])) * Z.row(row);
        }
        result.row(P) *= 3.7 * Ro[P];
    }
    return result;
}
