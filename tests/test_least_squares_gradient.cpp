#include "discretization/least_squares_gradient.hpp"

#include <array>
#include <gtest/gtest.h>
#include <vector>

TEST(LeastSquaresGradient, ReconstructsLinearField) {
    const std::array connections = {
        LeastSquaresConnection{0, 1, {1.0, 0.0}},
        LeastSquaresConnection{0, 2, {0.0, 1.0}},
        LeastSquaresConnection{1, 3, {0.0, 1.0}},
        LeastSquaresConnection{2, 3, {1.0, 0.0}},
    };
    const std::array<LeastSquaresBoundarySample, 0> boundary_samples{};
    const LeastSquaresGradient reconstruction(4, connections, boundary_samples);
    const std::array values = {2.0, 5.0, -2.0, 1.0};
    std::vector<LeastSquaresGradient::Gradient> gradients(4);

    reconstruction.reconstruct(values, std::span<const double>{}, gradients);

    for (const auto& gradient : gradients) {
        EXPECT_NEAR(gradient.x(), 3.0, 1.0e-14);
        EXPECT_NEAR(gradient.y(), -4.0, 1.0e-14);
    }
}
