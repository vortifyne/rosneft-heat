#include "model_heat_inverse_test_problem.hpp"
#include "petsc_test_session.hpp"

#include <Eigen/SVD>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>

namespace {

using namespace model_heat_inverse_test;

constexpr ModelHeatParameters kHiddenParameters{
    .surface_temperature = 12.0,
    .basal_heat_flux = 0.3,
};

Vector evaluate_normalized_residuals(const ModelHeatObservationData& observations,
                                     const ModelHeatParameters& parameters) {
    const ModelHeatForwardResult result = solve_forward(parameters);
    return observations.normalized_residuals(result.calculated_temperature);
}

TEST(ModelHeatInverseRecoveryTest, HasIndependentNormalizedParameterSensitivities) {
    constexpr double kNormalizedPerturbation = 0.1;
    const ModelHeatObservationData observations = make_synthetic_observations(kHiddenParameters);
    const ModelHeatParameterSpace parameter_space = make_parameter_space();
    const Vector reference = parameter_space.to_normalized(kHiddenParameters);
    const Vector reference_residuals =
        evaluate_normalized_residuals(observations, kHiddenParameters);

    Eigen::MatrixXd sensitivity(observations.size(), ModelHeatParameterSpace::kParameterCount);
    for (Vector::Index parameter_index = 0;
         parameter_index < ModelHeatParameterSpace::kParameterCount; ++parameter_index) {
        Vector perturbed = reference;
        perturbed[parameter_index] += kNormalizedPerturbation;
        const Vector perturbed_residuals =
            evaluate_normalized_residuals(observations, parameter_space.to_physical(perturbed));
        sensitivity.col(parameter_index) =
            (perturbed_residuals - reference_residuals).native() / kNormalizedPerturbation;
    }

    const double surface_sensitivity_norm = sensitivity.col(0).norm();
    const double basal_sensitivity_norm = sensitivity.col(1).norm();
    const double sensitivity_correlation = std::abs(sensitivity.col(0).dot(sensitivity.col(1))) /
                                           (surface_sensitivity_norm * basal_sensitivity_norm);
    const Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(sensitivity);
    const double largest_singular_value = decomposition.singularValues()[0];
    const double smallest_singular_value = decomposition.singularValues()[1];
    const double condition_number = largest_singular_value / smallest_singular_value;

    std::cout << std::setprecision(10)
              << "normalized sensitivity norms: surface=" << surface_sensitivity_norm
              << ", basal=" << basal_sensitivity_norm << ", correlation=" << sensitivity_correlation
              << ", singular values=" << largest_singular_value << ", " << smallest_singular_value
              << ", condition=" << condition_number << '\n';

    EXPECT_GT(surface_sensitivity_norm, 1e-3);
    EXPECT_GT(basal_sensitivity_norm, 1e-3);
    EXPECT_LT(sensitivity_correlation, 0.999);
    EXPECT_GT(smallest_singular_value, 1e-3);
    EXPECT_LT(condition_number, 1e4);
}

TEST(ModelHeatInverseRecoveryTest, RecoversHiddenParametersFromMultipleInitialGuesses) {
    ensure_petsc_test_session();
    const ModelHeatObservationData observations = make_synthetic_observations(kHiddenParameters);
    const std::array<ModelHeatParameters, 5> initial_guesses{
        ModelHeatParameters{.surface_temperature = 6.0, .basal_heat_flux = 0.1},
        ModelHeatParameters{.surface_temperature = 18.0, .basal_heat_flux = 0.8},
        ModelHeatParameters{.surface_temperature = 6.0, .basal_heat_flux = 0.8},
        ModelHeatParameters{.surface_temperature = 18.0, .basal_heat_flux = 0.1},
        ModelHeatParameters{.surface_temperature = 12.5, .basal_heat_flux = 0.5},
    };

    for (const ModelHeatParameters& initial_parameters : initial_guesses) {
        SCOPED_TRACE(::testing::Message()
                     << "initial surface temperature=" << initial_parameters.surface_temperature
                     << ", initial basal heat flux=" << initial_parameters.basal_heat_flux);
        const double initial_objective =
            evaluate_normalized_residuals(observations, initial_parameters).squared_norm();
        ModelHeatInverseSolver solver = make_inverse_solver(observations);

        const ModelHeatInverseResult result = solver.solve({
            .initial_parameters = initial_parameters,
            .optimization = kOptimizationRequest,
        });

        std::cout << std::setprecision(10) << "start=(" << initial_parameters.surface_temperature
                  << ", " << initial_parameters.basal_heat_flux << "), solution=("
                  << result.parameters.surface_temperature << ", "
                  << result.parameters.basal_heat_flux << "), objective: " << initial_objective
                  << " -> " << result.objective
                  << ", evaluations=" << result.optimization_function_evaluations << '\n';

        EXPECT_TRUE(result.converged());
        EXPECT_LT(result.objective, initial_objective * 1e-8);
        EXPECT_LT(result.objective, 1e-10);
        EXPECT_LT(result.normalized_residuals.infinity_norm(), 1e-5);
        EXPECT_NEAR(result.parameters.surface_temperature, kHiddenParameters.surface_temperature,
                    1e-4);
        EXPECT_NEAR(result.parameters.basal_heat_flux, kHiddenParameters.basal_heat_flux, 1e-4);
        EXPECT_EQ(result.forward_evaluations, result.optimization_function_evaluations + 1);
    }
}

} // namespace
