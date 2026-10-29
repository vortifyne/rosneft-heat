#pragma once

#include "basin/basin_input.hpp"
#include "basin/basin_snapshot.hpp"
#include "discretization/multipoint_flux_types.hpp"
#include "time/time_integrator.hpp"

#include <filesystem>
#include <functional>
#include <optional>

struct BasinForwardSettings {
    enum class Pinchout {
        positive_height,
        exact_zero
    };
    double horizontal_step = 50.0;
    double vertical_step = 50.0;
    double minimum_cell_height = 1.0e-3;
    double timestep_ma = 0.25;
    double rdp_tolerance = 0.0;
    // If set, simplify only the column axis; integrate the geometry selected
    // by rdp_tolerance without this additional simplification.
    std::optional<double> axis_rdp_tolerance = std::nullopt;
    Pinchout pinchout = Pinchout::positive_height;
    MultipointFluxScheme scheme = MultipointFluxScheme::linear;
    LinearSolverOptions linear_solver{};
    PreconditionerReuse pc_reuse = PreconditionerReuse::none;
    NonlinearSolveRequest nonlinear{};
    LinearSolveRequest linear{};
};

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
