#pragma once

#include "basin/forward/basin_forward_settings.hpp"
#include "basin/io/basin_input.hpp"

#include <filesystem>
#include <functional>

struct BasinForwardSnapshot;

struct BasinForwardResult {
    int timesteps = 0;
    int nonlinear_iterations = 0;
    int linear_iterations = 0;
};

using BasinForwardObserver = std::function<void(const BasinForwardSnapshot&)>;

class BasinForwardSolver {
public:
    BasinForwardSolver(BasinInput input, BasinForwardSettings settings = {});

    [[nodiscard]] BasinForwardResult solve(const BasinForwardObserver& observe = {}) const;
    [[nodiscard]] BasinForwardResult solve(const std::filesystem::path& output_directory,
                                           const BasinForwardObserver& observe = {}) const;

private:
    BasinInput input_;
    BasinForwardSettings settings_;
};
