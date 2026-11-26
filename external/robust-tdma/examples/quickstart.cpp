#include <array>
#include <cmath>
#include <iostream>
#include <tdma/tdma.hpp>

int main() {
    const std::array lower{-1.0, -1.0};
    const std::array diagonal{2.0, 2.0, 2.0};
    const std::array upper{-1.0, -1.0};
    const std::array rhs{0.0, 0.0, 4.0};

    tdma::StandardSolver<double> solver;
    solver.compute({lower, diagonal, upper});
    auto solution = solver.solve(rhs); // Allocates the result; expected: (1, 2, 3).
    const std::array expected{1.0, 2.0, 3.0};
    for (std::size_t i = 0; i < solution.size(); ++i) {
        std::cout << solution[i] << ' ';
        if (!(std::abs(solution[i] - expected[i]) <= 1.0e-12)) {
            return 1;
        }
    }
    std::cout << '\n';

    // Reuse both the prepared matrix and the output buffer, without allocation.
    const std::array next_rhs{4.0, 0.0, 0.0};
    solver.solve(next_rhs, solution); // Expected: (3, 2, 1).
    for (std::size_t i = 0; i < solution.size(); ++i) {
        std::cout << solution[i] << ' ';
        if (!(std::abs(solution[i] - expected[expected.size() - 1 - i]) <= 1.0e-12)) {
            return 1;
        }
    }
    std::cout << '\n';
}
