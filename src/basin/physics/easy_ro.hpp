#pragma once

#include <cstddef>
#include <span>
#include <vector>

struct EasyRoParameters {
    double A;              // Arrhenius prefactor [1/s].
    double Rg;             // Gas constant [J/(mol*K)].
    std::vector<double> E; // Activation energies [J/mol].
    std::vector<double> f; // Reaction weights [-].
};

class EasyRoModel {
public:
    explicit EasyRoModel(EasyRoParameters params);

    [[nodiscard]] std::size_t reaction_count() const noexcept;
    [[nodiscard]] std::vector<double> initial_state(std::size_t cell_count) const;

    void advance(std::span<const double> T0, std::span<const double> T1, double dt,
                 std::span<double> I) const;

    void reflectance(std::span<const double> I, std::span<double> Ro) const;

private:
    EasyRoParameters params_;
    std::vector<double> beta_; // E / Rg [K].
};
