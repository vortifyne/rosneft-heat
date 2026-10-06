#pragma once

#include <cstddef>
#include <span>
#include <vector>

struct EasyRoParameters {
    double preexponential;
    double gas_constant;
    std::vector<double> activation_energies;
    std::vector<double> weights;
};

class EasyRoModel {
public:
    explicit EasyRoModel(EasyRoParameters parameters);

    [[nodiscard]] std::size_t reaction_count() const noexcept;
    [[nodiscard]] std::vector<double> initial_state(std::size_t cell_count) const;

    void advance(std::span<const double> old_temperature, std::span<const double> new_temperature,
                 double timestep, std::span<double> integrals) const;

    void reflectance(std::span<const double> integrals,
                     std::span<double> reflectance_percent) const;

private:
    EasyRoParameters parameters_;
    std::vector<double> beta_;
};
