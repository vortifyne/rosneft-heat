#pragma once

#include <filesystem>
#include <optional>

struct BasinForwardOptions {
    double cell_size = 50.0;
    double timestep_ma = 0.1;
    int saved_states_per_epoch = 5;
    double thin_layer_cell_fraction = 0.0;
    bool fixed_mesh = false;
    std::optional<std::size_t> max_configurations;
    std::optional<std::filesystem::path> comparison_points;
};

struct BasinForwardResult {
    int accepted_steps = 0;
    int nonlinear_iterations = 0;
    int linear_iterations = 0;
    int configurations = 0;
    double final_age_ma = 0.0;
    double transfer_energy_error = 0.0;
    std::size_t minimum_cells = 0;
    std::size_t maximum_cells = 0;
    double maximum_cell_diameter = 0.0;
    double wall_seconds = 0.0;
};

[[nodiscard]] BasinForwardResult run_basin_forward(const std::filesystem::path& input_directory,
                                                   const std::filesystem::path& output_directory,
                                                   const BasinForwardOptions& options);
