#pragma once

#include <chrono>

using Clock = std::chrono::steady_clock;

inline double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct AssemblyTimings {
    double residual_s = 0.0;
    double matrix_s = 0.0;

    [[nodiscard]] double total_s() const noexcept {
        return residual_s + matrix_s;
    }

    AssemblyTimings& operator+=(const AssemblyTimings& other) noexcept {
        residual_s += other.residual_s;
        matrix_s += other.matrix_s;
        return *this;
    }
};

struct SolverTimings {
    double total_s = 0.0;
    AssemblyTimings assembly{};
    double linear_s = 0.0;

    SolverTimings& operator+=(const SolverTimings& other) noexcept {
        total_s += other.total_s;
        assembly += other.assembly;
        linear_s += other.linear_s;
        return *this;
    }
};
