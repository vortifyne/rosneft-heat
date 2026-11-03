#include "basin/physics/easy_ro.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>

namespace {

EasyRoModel make_model() {
    return EasyRoModel({.A = 1.0e13, .Rg = 8.314, .E = {142351.2, 150724.8}, .f = {0.03, 0.03}});
}

double reaction_rate(const double energy, const double T) {
    return 1.0e13 * std::exp(-energy / (8.314 * T));
}

double refined_increment(const double energy, const double T0, const double T1,
                         const double timestep) {
    constexpr int reference_parts = 20000;
    double result = 0.0;
    for (int part = 0; part < reference_parts; ++part) {
        const double fraction = (static_cast<double>(part) + 0.5) / reference_parts;
        const double T = ((1.0 - fraction) * T0) + (fraction * T1);
        result += reaction_rate(energy, T);
    }
    return result * timestep / reference_parts;
}

} // namespace

TEST(EasyRo, IntegratesConstantTemperatureExactly) {
    const EasyRoModel model = make_model();
    std::vector<double> state = model.initial_state(1);
    constexpr double T = 400.0;
    constexpr double timestep = 1000.0;

    model.advance(std::array{T}, std::array{T}, timestep, state);

    EXPECT_NEAR(state[0], timestep * 1.0e13 * std::exp(-142351.2 / (8.314 * T)), 1.0e-14);
    EXPECT_NEAR(state[1], timestep * 1.0e13 * std::exp(-150724.8 / (8.314 * T)), 1.0e-14);
}

TEST(EasyRo, ReflectanceGrowsAfterHeating) {
    const EasyRoModel model = make_model();
    std::vector<double> state = model.initial_state(1);
    std::array<double, 1> initial{};
    std::array<double, 1> heated{};
    model.reflectance(state, initial);
    model.advance(std::array{450.0}, std::array{500.0}, 1.0e12, state);
    model.reflectance(state, heated);

    EXPECT_GT(heated[0], initial[0]);
}

TEST(EasyRo, IntegralMatchesRefinedIntegrationAndPhysicalBounds) {
    std::vector<double> energies;
    std::vector<double> f;
    for (int index = 0; index < 20; ++index) {
        energies.push_back((34.0 + (2.0 * index)) * 4186.8);
        f.push_back(index < 2 ? 0.03 : 0.04);
    }
    const EasyRoModel model({.A = 1.0e13, .Rg = 8.314, .E = energies, .f = f});
    constexpr double timestep = 0.1e6 * 365.25 * 24.0 * 3600.0;
    for (const auto T :
         {std::array{300.0, 500.0}, std::array{500.0, 300.0}, std::array{350.0, 360.0}}) {
        std::vector<double> state = model.initial_state(1);
        std::vector<double> reference_state = model.initial_state(1);
        model.advance(std::array{T[0]}, std::array{T[1]}, timestep, state);
        for (std::size_t reaction = 0; reaction < energies.size(); ++reaction) {
            const double reference = refined_increment(energies[reaction], T[0], T[1], timestep);
            reference_state[reaction] = reference;
            EXPECT_NEAR(std::exp(-state[reaction]), std::exp(-reference), 3.0e-4);
            const double left = timestep * reaction_rate(energies[reaction], T[0]);
            const double right = timestep * reaction_rate(energies[reaction], T[1]);
            EXPECT_GE(state[reaction], std::min(left, right) - 1.0e-12);
            EXPECT_LE(state[reaction], std::max(left, right) + 1.0e-12);
        }
        std::array<double, 1> Ro{}, reference_Ro{};
        model.reflectance(state, Ro);
        model.reflectance(reference_state, reference_Ro);
        EXPECT_NEAR(Ro[0] / reference_Ro[0], 1.0, 1.0e-3);
    }
}

TEST(EasyRo, TabulatedPrimitiveMatchesHeatingCoolingAndNearlyConstantHistories) {
    const std::array energies{142351.2, 301449.6};
    const EasyRoModel model(
        {.A = 1.0e13, .Rg = 8.314, .E = {energies[0], energies[1]}, .f = {0.03, 0.01}});
    constexpr double timestep = 1.0e6 * 365.25 * 24.0 * 3600.0;
    const std::array T0{200.0, 250.0, 400.0, 400.000001, 750.0, 800.0};
    const std::array T1{250.0, 200.0, 400.000001, 400.0, 800.0, 750.0};
    auto state = model.initial_state(T0.size());

    model.advance(T0, T1, timestep, state);

    for (std::size_t P = 0; P < T0.size(); ++P) {
        for (std::size_t i = 0; i < energies.size(); ++i) {
            const double reference = refined_increment(energies[i], T0[P], T1[P], timestep);
            // Tiny integrals may vanish when subtracting the primitive terms.
            EXPECT_NEAR(state[P * energies.size() + i], reference,
                        2.0e-6 * std::max(1.0, reference));
        }
    }
}

TEST(EasyRo, UsesCorrectedOrganizerFormulaWithoutNormalization) {
    const std::vector<double> f = {0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07,
                                   0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01};
    std::vector<double> energies;
    for (int index = 0; index < 20; ++index) {
        energies.push_back((34.0 + (2.0 * index)) * 4186.8);
    }
    const EasyRoModel model({.A = 1.0e13, .Rg = 8.314, .E = energies, .f = f});
    const std::vector<double> state = model.initial_state(1);
    std::array<double, 1> reflectance{};

    model.reflectance(state, reflectance);

    EXPECT_NEAR(reflectance[0], std::exp(-1.6), 1.0e-14);

    std::vector<double> fully_reacted(model.reaction_count(), 1000.0);
    model.reflectance(fully_reacted, reflectance);
    EXPECT_NEAR(reflectance[0], std::exp(-1.6 + (3.7 * 0.85)), 1.0e-14);
}
