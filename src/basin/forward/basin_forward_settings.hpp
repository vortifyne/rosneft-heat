#pragma once

#include "basin/discretization/multipoint_flux_types.hpp"
#include "numerics/linear/linear_solver.hpp"
#include "numerics/nonlinear/nonlinear_solver.hpp"
#include "numerics/time/preconditioner_reuse.hpp"

#include <optional>

struct BasinForwardSettings {
    double horizontal_step = 50.0;
    double vertical_step = 50.0;
    double minimum_cell_height = 1.0e-3;
    double timestep_ma = 0.25;
    double rdp_tolerance = 0.0;
    // If set, simplify only the column axis; integrate the geometry selected
    // by rdp_tolerance without this additional simplification.
    std::optional<double> axis_rdp_tolerance = std::nullopt;
    MultipointFluxScheme scheme = MultipointFluxScheme::linear;
    LinearSolverOptions linear_solver{};
    PreconditionerReuse pc_reuse = PreconditionerReuse::none;
    NonlinearSolveRequest nonlinear{};
    LinearSolveRequest linear{};
};
