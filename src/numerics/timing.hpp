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

struct LinearTimings {
    double transfer_s = 0.0;
    double setup_s = 0.0;
    double solve_s = 0.0;
    int matrix_builds = 0;
    int vector_builds = 0;
    int ksp_builds = 0;
    int pc_updates = 0;

    LinearTimings& operator+=(const LinearTimings& other) noexcept {
        transfer_s += other.transfer_s;
        setup_s += other.setup_s;
        solve_s += other.solve_s;
        matrix_builds += other.matrix_builds;
        vector_builds += other.vector_builds;
        ksp_builds += other.ksp_builds;
        pc_updates += other.pc_updates;
        return *this;
    }
};

struct SolverTimings {
    double total_s = 0.0;
    AssemblyTimings assembly{};
    double linear_s = 0.0;
    LinearTimings linear{};

    SolverTimings& operator+=(const SolverTimings& other) noexcept {
        total_s += other.total_s;
        assembly += other.assembly;
        linear_s += other.linear_s;
        linear += other.linear;
        return *this;
    }
};
