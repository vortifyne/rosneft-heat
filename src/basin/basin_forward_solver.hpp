#pragma once

#include "basin/basin_input.hpp"

#include <filesystem>
#include <memory>
#include <vector>

struct BasinForwardSettings {
    double cell_size = 50.0;
    double timestep_ma = 0.25;
};

struct BasinForwardParameters {
    std::vector<ScalarProfile> surface_temperature;
    std::vector<ScalarProfile> basal_heat_flux;
};

struct BasinForwardResult {
    int accepted_steps = 0;
    int nonlinear_iterations = 0;
    int maximum_nonlinear_iterations_per_step = 0;
    int linear_iterations = 0;
    int topology_regularized_epochs = 0;
    int configurations = 0;
    double final_age_ma = 0.0;
    double global_energy_balance = 0.0;
    std::size_t minimum_cells = 0;
    std::size_t maximum_cells = 0;
    double maximum_cell_diameter = 0.0;
    double wall_seconds = 0.0;
};

class BasinForwardSolver {
public:
    BasinForwardSolver(const std::filesystem::path& input_directory, BasinForwardSettings settings);
    ~BasinForwardSolver();

    BasinForwardSolver(const BasinForwardSolver&) = delete;
    BasinForwardSolver& operator=(const BasinForwardSolver&) = delete;
    BasinForwardSolver(BasinForwardSolver&&) noexcept;
    BasinForwardSolver& operator=(BasinForwardSolver&&) noexcept;

    [[nodiscard]] const BasinForwardParameters& default_parameters() const noexcept;
    [[nodiscard]] BasinForwardResult solve(const BasinForwardParameters& parameters,
                                           const std::filesystem::path& output_directory) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
