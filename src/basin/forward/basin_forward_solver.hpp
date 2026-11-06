#pragma once

#include "basin/forward/basin_forward_settings.hpp"
#include "basin/forward/basin_statistics.hpp"
#include "basin/io/basin_input.hpp"
#include "basin/mesh/col_epoch_grid.hpp"
#include "numerics/algebra/vector.hpp"

#include <array>
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

    void set_parameters(const Vector& parameters);
    void set_intermediate_snapshots(bool enabled) noexcept;
    [[nodiscard]] std::span<const BasinConfig> configs() const noexcept;
    [[nodiscard]] const ColGrid& final_grid() const noexcept;
    void final_temperature_nodes(std::span<const double> T, std::span<double> node_T) const;

    [[nodiscard]] BasinForwardResult solve(const BasinObserver& observe = {}) const;
    [[nodiscard]] BasinForwardResult solve(const std::filesystem::path& output_directory,
                                           const BasinObserver& observe = {}) const;

private:
    [[nodiscard]] Vector solve_initial_temperature() const;
    [[nodiscard]] BasinForwardResult solve(const BasinObserver& observe,
                                           bool configuration_output) const;
    void temperature_nodes(const ColGrid& grid, std::size_t configuration,
                           std::span<const double> T, std::span<double> node_T) const;

    struct EpochData {
        ColEpochGrid mesh;
        ColGrid initial_grid;
        std::vector<int> lithotypes;
        std::vector<std::array<double, 2>> phi0;
        std::vector<std::array<double, 2>> phi1;
        std::vector<double> deposition_times;
    };

    BasinInput input_;
    BasinForwardSettings settings_;
    EasyRoModel easy_ro_;
    std::vector<EpochData> epochs_;
    ColGrid final_grid_;
    ColGrid initial_grid_;
    Vector parameters_; // Basal heat flux [W/m²] at configuration ages.
};
