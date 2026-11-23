#include <array>
#include <cmath>
#include <iostream>
#include <tdma/tdma.hpp>

int main() {
    // [0 1; 1 0] is nonsingular, but a classical sweep divides by its first zero pivot.
    const std::array lower{1.0};
    const std::array diagonal{0.0, 0.0};
    const std::array upper{1.0};
    const std::array rhs{3.0, 2.0};

    tdma::PivotingSolver<double> solver;
    solver.compute({lower, diagonal, upper});
    const auto solution = solver.solve(rhs); // Expected: (2, 3).
    std::cout << solution[0] << ' ' << solution[1] << '\n';
    return std::abs(solution[0] - 2.0) <= 1.0e-12 && std::abs(solution[1] - 3.0) <= 1.0e-12 ? 0 : 1;
}
