#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <tdma/tdma.hpp>
#include <vector>

int main() {
    // u_t = kappa*u_xx on [0,1], u(0,t) = u(1,t) = 0, u(x,0) = sin(pi*x).
    constexpr std::size_t interior = 63;
    constexpr std::size_t steps = 1000;
    constexpr double kappa = 1.0;
    constexpr double final_time = 0.1;
    constexpr double pi = std::numbers::pi;
    const double h = 1.0 / static_cast<double>(interior + 1);
    const double dt = final_time / static_cast<double>(steps);
    const double r = kappa * dt / (h * h);

    // Backward Euler: -r*u[i-1] + (1+2r)*u[i] - r*u[i+1] = previous_u[i].
    const std::vector<double> lower(interior - 1, -r);
    const std::vector<double> diagonal(interior, 1.0 + 2.0 * r);
    const std::vector<double> upper(interior - 1, -r);
    tdma::StandardSolver<double> solver;
    solver.compute({lower, diagonal, upper}); // The matrix is the same at every time step.

    std::vector<double> values(interior), next(interior);
    for (std::size_t i = 0; i < interior; ++i) {
        values[i] = std::sin(pi * static_cast<double>(i + 1) * h);
    }
    for (std::size_t step = 0; step < steps; ++step) {
        solver.solve(values, next);
        values.swap(next);
    }

    const double decay = std::exp(-kappa * pi * pi * final_time);
    double max_error = 0.0;
    for (std::size_t i = 0; i < interior; ++i) {
        const double exact = decay * std::sin(pi * static_cast<double>(i + 1) * h);
        max_error = std::max(max_error, std::abs(values[i] - exact));
    }
    std::cout << std::setprecision(8) << "x\tu(x, " << final_time << ")\n";
    for (std::size_t i = 0; i <= interior + 1; i += (interior + 1) / 4) {
        const double value = (i == 0 || i == interior + 1) ? 0.0 : values[i - 1];
        std::cout << static_cast<double>(i) * h << '\t' << value << '\n';
    }
    // This includes time/space discretization error, not just the linear solve error.
    std::cout << "Maximum error against exp(-kappa*pi^2*t)*sin(pi*x): " << max_error << '\n';
}
