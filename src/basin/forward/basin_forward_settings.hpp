#pragma once

#include "basin/discretization/multipoint_flux_types.hpp"
#include "numerics/linear/linear_solver.hpp"
#include "numerics/nonlinear/nonlinear_solver.hpp"
#include "numerics/time/pc_reuse.hpp"

#include <optional>

struct BasinForwardSettings {
    double dx = 200.0;      // Horizontal step [m].
    double dz = 50.0;       // Vertical step [m].
    double dz_min = 5.0e-2; // Minimum cell height [m].
    double dt_ma = 1.0;     // Time step [Ma].
    double rdp_tol = 5.0;
    // If set, simplify only the column axis; integrate the geometry selected
    // by rdp_tol without this additional simplification.
    std::optional<double> axis_rdp_tol = std::nullopt;
    bool intermediate_snapshots = true;
    bool reset_final_maturity = false; // Discard the final new layer's reaction history.
    MultipointFluxScheme scheme = MultipointFluxScheme::linear;
    LinearSolverOptions linear_solver{.method = LinearMethod::gmres,
                                      .preconditioner = Preconditioner::ilu2};
    PcReuse pc_reuse = PcReuse::none;
    NonlinearSolveRequest nonlinear{};
    LinearSolveRequest linear{};
};
