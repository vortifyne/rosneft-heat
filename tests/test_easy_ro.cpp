#include "physics/easy_ro.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>

namespace {

EasyRoModel make_model() {
    return EasyRoModel({.preexponential = 1.0e13,
                        .gas_constant = 8.314,
                        .activation_energies = {142351.2, 150724.8},
                        .weights = {0.03, 0.03}});
}

double reaction_rate(const double energy, const double temperature) {
    return 1.0e13 * std::exp(-energy / (8.314 * temperature));
}

double refined_increment(const double energy, const double old_temperature,
                         const double new_temperature, const double timestep) {
    constexpr int reference_parts = 20000;
    double result = 0.0;
    for (int part = 0; part < reference_parts; ++part) {
        const double fraction = (static_cast<double>(part) + 0.5) / reference_parts;
        const double temperature =
            ((1.0 - fraction) * old_temperature) + (fraction * new_temperature);
        result += reaction_rate(energy, temperature);
    }
    return result * timestep / reference_parts;
}

} // namespace

TEST(EasyRo, IntegratesConstantTemperatureExactly) {
    const EasyRoModel model = make_model();
    std::vector<double> state = model.initial_state(1);
    constexpr double temperature = 400.0;
    constexpr double timestep = 1000.0;

    model.advance(std::array{temperature}, std::array{temperature}, timestep, state);

    EXPECT_NEAR(state[0], timestep * 1.0e13 * std::exp(-142351.2 / (8.314 * temperature)), 1.0e-14);
    EXPECT_NEAR(state[1], timestep * 1.0e13 * std::exp(-150724.8 / (8.314 * temperature)), 1.0e-14);
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

TEST(EasyRo, GaussQuadratureAgreesWithRefinedIntegrationForOrganizerReactions) {
    std::vector<double> energies;
    std::vector<double> weights;
    for (int index = 0; index < 20; ++index) {
        energies.push_back((34.0 + (2.0 * index)) * 4186.8);
        weights.push_back(index < 2 ? 0.03 : 0.04);
    }
    const EasyRoModel model({.preexponential = 1.0e13,
                             .gas_constant = 8.314,
                             .activation_energies = energies,
                             .weights = weights});
    constexpr double timestep = 0.1e6 * 365.25 * 24.0 * 3600.0;
    for (const auto temperatures :
         {std::array{300.0, 500.0}, std::array{500.0, 300.0}, std::array{350.0, 360.0}}) {
        std::vector<double> state = model.initial_state(1);
        model.advance(std::array{temperatures[0]}, std::array{temperatures[1]}, timestep, state);
        for (std::size_t reaction = 0; reaction < energies.size(); ++reaction) {
            const double reference =
                refined_increment(energies[reaction], temperatures[0], temperatures[1], timestep);
            EXPECT_NEAR(std::exp(-state[reaction]), std::exp(-reference), 3.0e-4);
        }
    }
}

TEST(EasyRo, GaussQuadratureIsBoundedByRectanglesAndMoreAccurateThanTrapezoid) {
    std::vector<double> energies;
    std::vector<double> weights;
    for (int index = 0; index < 20; ++index) {
        energies.push_back((34.0 + (2.0 * index)) * 4186.8);
        weights.push_back(index < 2 ? 0.03 : 0.04);
    }
    const EasyRoModel model({.preexponential = 1.0e13,
                             .gas_constant = 8.314,
                             .activation_energies = energies,
                             .weights = weights});
    constexpr double timestep = 0.1e6 * 365.25 * 24.0 * 3600.0;
    for (const auto temperatures :
         {std::array{300.0, 500.0}, std::array{500.0, 300.0}, std::array{350.0, 360.0}}) {
        std::vector<double> state = model.initial_state(1);
        model.advance(std::array{temperatures[0]}, std::array{temperatures[1]}, timestep, state);
        for (std::size_t reaction = 0; reaction < energies.size(); ++reaction) {
            const double left = timestep * reaction_rate(energies[reaction], temperatures[0]);
            const double right = timestep * reaction_rate(energies[reaction], temperatures[1]);
            const double midpoint =
                timestep *
                reaction_rate(energies[reaction], 0.5 * (temperatures[0] + temperatures[1]));
            const double trapezoid = 0.5 * (left + right);
            const double reference =
                refined_increment(energies[reaction], temperatures[0], temperatures[1], timestep);
            const double gauss_error = std::abs(std::exp(-state[reaction]) - std::exp(-reference));
            const double trapezoid_error = std::abs(std::exp(-trapezoid) - std::exp(-reference));

            EXPECT_GE(state[reaction], std::min(left, right));
            EXPECT_LE(state[reaction], std::max(left, right));
            EXPECT_LE(gauss_error, trapezoid_error + 1.0e-14);
            EXPECT_TRUE(std::isfinite(midpoint));
        }
    }
}

TEST(EasyRo, UsesCorrectedOrganizerFormulaWithoutNormalization) {
    const std::vector<double> weights = {0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06,
                                         0.04, 0.04, 0.07, 0.06, 0.06, 0.06, 0.05,
                                         0.05, 0.04, 0.03, 0.02, 0.02, 0.01};
    std::vector<double> energies;
    for (int index = 0; index < 20; ++index) {
        energies.push_back((34.0 + (2.0 * index)) * 4186.8);
    }
    const EasyRoModel model({.preexponential = 1.0e13,
                             .gas_constant = 8.314,
                             .activation_energies = energies,
                             .weights = weights});
    const std::vector<double> state = model.initial_state(1);
    std::array<double, 1> reflectance{};

    model.reflectance(state, reflectance);

    EXPECT_NEAR(reflectance[0], std::exp(-1.6), 1.0e-14);

    std::vector<double> fully_reacted(model.reaction_count(), 1000.0);
    model.reflectance(fully_reacted, reflectance);
    EXPECT_NEAR(reflectance[0], std::exp(-1.6 + (3.7 * 0.85)), 1.0e-14);
}
