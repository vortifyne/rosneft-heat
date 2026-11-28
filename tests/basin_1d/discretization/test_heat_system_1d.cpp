#include "basin_1d/discretization/heat_system_1d.hpp"
#include "numerics/linear/tdma_linear_solver.hpp"
#include "numerics/time/time_integrator.hpp"

#include <array>
#include <gtest/gtest.h>

TEST(HeatSystem1D, SolvesSteadyAndBdf1BalancesOnUnequalIntervals) {
    constexpr double k = 4.4 * 0.419; // Fixed point of the conductivity correction.
    constexpr double surface = 290.;
    const LithotypeProperties solid{1, 1., k, 1., 0.};
    const std::array faces{0., 1., 3.};
    const std::array phi{1., 1.};
    const std::array materials{&solid, &solid};
    HeatSystem1D system(faces, phi, materials, {k, 1., 1.}, surface, k);
    TimeIntegrator time;
    time.set_linear_solver({.method = LinearMethod::tdma});

    const Vector initial{surface, surface};
    Vector steady = initial;
    ASSERT_TRUE(time.nonlinear_solver().solve(system, steady, {}, {}).converged());
    EXPECT_NEAR(steady[0], surface + 0.5, 1e-10);
    EXPECT_NEAR(steady[1], surface + 2., 1e-10);

    const double dt = 1. / k;
    time.set_initial_solution(0., initial);
    time.set_timestep(dt);
    ASSERT_TRUE(time.advance_to(system, dt, {}, {}).completed());
    const auto& T = time.current_snapshot().u;
    EXPECT_NEAR(T[0], surface + 1. / 14., 1e-10);
    EXPECT_NEAR(T[1], surface + 11. / 28., 1e-10);

    SparseMatrix A(2, 2, MatrixStorageFormat::tridiagonal);
    system.assemble_matrix(NonlinearMethod::picard, dt, T, (T - initial) / dt, 1. / dt, A);
    const Eigen::MatrixXd inverse = TdmaLinearSolver::solve_columns(A, Eigen::Matrix2d::Identity());
    for (Eigen::Index j = 0; j < 2; ++j) {
        const Vector column(inverse.col(j));
        EXPECT_TRUE((A * column).native().isApprox(Eigen::Vector2d::Unit(j), 1e-12));
    }
}
