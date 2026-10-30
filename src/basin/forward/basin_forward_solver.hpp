#pragma once

#include "basin/forward/basin_forward_settings.hpp"
#include "basin/io/basin_input.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>

struct BasinForwardSnapshot;

struct BasinForwardResult {
    int accepted_steps = 0;
    int nonlinear_iterations = 0;
    int linear_iterations = 0;
    std::size_t maximum_cells = 0;
    double maximum_cell_residual = 0.0;
    double maximum_global_balance = 0.0;
    double maximum_internal_flux_defect = 0.0;
    double final_age_ma = 0.0;
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
