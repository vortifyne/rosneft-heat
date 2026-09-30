#include "physics/easy_ro.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

EasyRoModel::EasyRoModel(EasyRoParameters parameters) : parameters_(std::move(parameters)) {
    if (!(parameters_.preexponential > 0.0) || !std::isfinite(parameters_.preexponential) ||
        !(parameters_.gas_constant > 0.0) || !std::isfinite(parameters_.gas_constant) ||
        parameters_.activation_energies.empty() ||
        parameters_.activation_energies.size() != parameters_.weights.size()) {
        throw std::invalid_argument("Invalid EASY%Ro parameters");
    }
    beta_.reserve(parameters_.activation_energies.size());
    double weight_sum = 0.0;
    for (std::size_t index = 0; index < parameters_.activation_energies.size(); ++index) {
        const double energy = parameters_.activation_energies[index];
        const double weight = parameters_.weights[index];
        if (!(energy > 0.0) || !std::isfinite(energy) || weight < 0.0 || !std::isfinite(weight)) {
            throw std::invalid_argument("Invalid EASY%Ro reaction parameter");
        }
        beta_.push_back(energy / parameters_.gas_constant);
        weight_sum += weight;
    }
    if (!(weight_sum > 0.0)) {
        throw std::invalid_argument("EASY%Ro weights must have a positive sum");
    }
}

std::size_t EasyRoModel::reaction_count() const noexcept {
    return beta_.size();
}

std::vector<double> EasyRoModel::initial_state(const std::size_t cell_count) const {
    return std::vector<double>(cell_count * reaction_count(), 0.0);
}

void EasyRoModel::advance(const std::span<const double> old_temperature,
                          const std::span<const double> new_temperature, const double timestep,
                          const std::span<double> integrals) const {
    if (old_temperature.size() != new_temperature.size() ||
        integrals.size() != old_temperature.size() * reaction_count() || !(timestep > 0.0) ||
        !std::isfinite(timestep)) {
        throw std::invalid_argument("Invalid EASY%Ro state sizes or timestep");
    }
    constexpr int substeps = 16;
    constexpr double gauss_offset = 0.5 / 1.7320508075688772935;
    for (std::size_t cell = 0; cell < old_temperature.size(); ++cell) {
        const double old_value = old_temperature[cell];
        const double new_value = new_temperature[cell];
        if (!(old_value > 0.0) || !(new_value > 0.0) || !std::isfinite(old_value) ||
            !std::isfinite(new_value)) {
            throw std::invalid_argument("EASY%Ro temperatures must be finite and positive");
        }
        for (std::size_t reaction = 0; reaction < reaction_count(); ++reaction) {
            double increment = 0.0;
            for (int substep = 0; substep < substeps; ++substep) {
                const double center = (static_cast<double>(substep) + 0.5) / substeps;
                const double offset = gauss_offset / substeps;
                const double first = center - offset;
                const double second = center + offset;
                const double temperature_1 = ((1.0 - first) * old_value) + (first * new_value);
                const double temperature_2 = ((1.0 - second) * old_value) + (second * new_value);
                const double rate_1 =
                    parameters_.preexponential * std::exp(-beta_[reaction] / temperature_1);
                const double rate_2 =
                    parameters_.preexponential * std::exp(-beta_[reaction] / temperature_2);
                increment += 0.5 * timestep / substeps * (rate_1 + rate_2);
            }
            double& value = integrals[(cell * reaction_count()) + reaction];
            value += increment;
            if (!std::isfinite(value) || value < 0.0) {
                throw std::runtime_error("EASY%Ro integral became invalid");
            }
        }
    }
}

void EasyRoModel::reflectance(const std::span<const double> integrals,
                              const std::span<double> reflectance_percent) const {
    if (integrals.size() != reflectance_percent.size() * reaction_count()) {
        throw std::invalid_argument("Invalid EASY%Ro output size");
    }
    for (std::size_t cell = 0; cell < reflectance_percent.size(); ++cell) {
        double transformed_fraction = 0.0;
        for (std::size_t reaction = 0; reaction < reaction_count(); ++reaction) {
            const double integral = integrals[(cell * reaction_count()) + reaction];
            if (integral < 0.0 || !std::isfinite(integral)) {
                throw std::invalid_argument("EASY%Ro integral must be finite and non-negative");
            }
            transformed_fraction += parameters_.weights[reaction] * (1.0 - std::exp(-integral));
        }
        reflectance_percent[cell] = std::exp(-1.6 + (3.7 * transformed_fraction));
    }
}
