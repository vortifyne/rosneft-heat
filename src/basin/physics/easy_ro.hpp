#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <span>
#include <vector>

struct EasyRoParameters {
    double A;              // Arrhenius prefactor [1/s].
    double Rg;             // Gas constant [J/(mol*K)].
    std::vector<double> E; // Activation energies [J/mol].
    std::vector<double> f; // Reaction weights [-].
};

struct EasyRoStepSensitivities {
    const Eigen::MatrixXd& S0; // dT/deta at material points at the start and end of the step.
    const Eigen::MatrixXd& S1;
    Eigen::MatrixXd&
        Z; // dI/deta, initially zero. Rows follow I[point * reaction_count + reaction].
};

class EasyRoModel {
public:
    explicit EasyRoModel(EasyRoParameters params);

    [[nodiscard]] std::size_t reaction_count() const noexcept;
    [[nodiscard]] std::vector<double> initial_state(std::size_t point_count) const;

    // The primitive table covers 20 <= E / (Rg * T) < 200 at both temperatures.
    void advance(std::span<const double> T0, std::span<const double> T1, double t_old, double t_new,
                 std::span<const double> t_dep, std::span<double> I,
                 const EasyRoStepSensitivities* ddm = nullptr) const;

    void reflectance(std::span<const double> I, std::span<double> Ro) const;

    [[nodiscard]] Eigen::MatrixXd reflectance_sensitivities(std::span<const double> I,
                                                            std::span<const double> Ro,
                                                            const Eigen::MatrixXd& Z) const;

private:
    EasyRoParameters params_;
    std::vector<double> beta_; // E / Rg [K].
};
