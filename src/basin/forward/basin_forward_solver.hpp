#pragma once

#include "basin/forward/basin_forward_settings.hpp"
#include "basin/forward/basin_statistics.hpp"
#include "basin/io/basin_input.hpp"

#include <filesystem>
#include <functional>
#include <vector>

struct BasinSnapshot;

struct BasinForwardResult {
    int timesteps = 0;
    int nonlinear_iterations = 0;
    int linear_iterations = 0;
    BasinTimings timings;
    std::vector<BasinIntervalStats> epochs;
    std::vector<BasinIntervalStats> steps;
};

using BasinObserver = std::function<void(const BasinSnapshot&)>;

class BasinForwardSolver {
public:
    BasinForwardSolver(BasinInput input, BasinForwardSettings settings = {});

    [[nodiscard]] BasinForwardResult solve(const BasinObserver& observe = {}) const;
    [[nodiscard]] BasinForwardResult solve(const std::filesystem::path& output_directory,
                                           const BasinObserver& observe = {}) const;

private:
    BasinInput input_;
    BasinForwardSettings settings_;
};
