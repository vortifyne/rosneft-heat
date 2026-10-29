#pragma once

#include "basin/basin_input.hpp"
#include "basin/nonmatching_column_grid.hpp"
#include "discretization/multipoint_flux_types.hpp"
#include "linear/linear_solver.hpp"

#include <functional>
#include <optional>
#include <span>

struct ColumnBasinSettings {
    enum class Pinchout {
        positive_height,
        exact_zero
    };
    double horizontal_step = 50.0;
    double vertical_step = 50.0;
    double minimum_cell_height = 1.0e-3;
    double timestep_ma = 0.25;
    double picard_relative_tolerance = 1.0e-8;
    double rdp_tolerance = 0.0;
    // If set, simplify only the column axis; integrate the geometry selected
    // by rdp_tolerance without this additional simplification.
    std::optional<double> axis_rdp_tolerance = std::nullopt;
    Pinchout pinchout = Pinchout::positive_height;
    MultipointFluxScheme scheme = MultipointFluxScheme::linear;
    LinearSolverKind linear_solver = LinearSolverKind::umfpack_lu;
    LinearSolveRequest linear{};
};

struct ColumnBasinSnapshot {
    enum class Kind {
        initial,
        before_event,
        after_event
    };
    Kind kind = Kind::initial;
    double age_ma = 0.0;
    const NonmatchingColumnGrid& grid;
    std::span<const double> temperature;
    std::span<const double> maturity_integrals;
    std::span<const double> reflectance_percent;
};

struct ColumnBasinResult {
    int accepted_steps = 0;
    int nonlinear_iterations = 0;
    int linear_iterations = 0;
    std::size_t maximum_cells = 0;
    double maximum_cell_residual = 0.0;
    double maximum_global_balance = 0.0;
    double maximum_internal_flux_defect = 0.0;
    double final_age_ma = 0.0;
};

using ColumnBasinObserver = std::function<void(const ColumnBasinSnapshot&)>;

[[nodiscard]] ColumnBasinResult solve_column_basin(const BasinInput& input,
                                                   const ColumnBasinSettings& settings,
                                                   const ColumnBasinObserver& observe = {});
