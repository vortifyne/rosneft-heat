#pragma once

#include <filesystem>

struct BasinForwardOptions {
    double cell_size = 50.0;
    double timestep_ma = 0.1;
    int save_every = 10;
    bool fixed_mesh = false;
    int steps_after_transition = 3;
};

struct BasinForwardResult {
    int accepted_steps = 0;
    int nonlinear_iterations = 0;
    int linear_iterations = 0;
    double final_age_ma = 0.0;
    double transfer_energy_error = 0.0;
    double wall_seconds = 0.0;
};

[[nodiscard]] BasinForwardResult run_basin_forward(const std::filesystem::path& input_directory,
                                                   const std::filesystem::path& output_directory,
                                                   const BasinForwardOptions& options);
