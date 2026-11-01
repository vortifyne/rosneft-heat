#pragma once

#include "numerics/timing.hpp"

#include <cstddef>

struct BasinTimings {
    double total_s = 0.0;
    double grid_s = 0.0;
    double transfer_s = 0.0;
    SolverTimings nonlinear{};
    AssemblyTimings properties{};
    double easy_ro_s = 0.0;
    double output_s = 0.0;

    BasinTimings& operator+=(const BasinTimings& other) noexcept {
        total_s += other.total_s;
        grid_s += other.grid_s;
        transfer_s += other.transfer_s;
        nonlinear += other.nonlinear;
        properties += other.properties;
        easy_ro_s += other.easy_ro_s;
        output_s += other.output_s;
        return *this;
    }
};

struct BasinIntervalStats {
    std::size_t epoch = 0;
    double age_begin_ma = 0.0;
    double age_end_ma = 0.0;
    int timesteps = 0;
    int nonlinear_iterations = 0;
    int linear_iterations = 0;
    BasinTimings timings{};
};
