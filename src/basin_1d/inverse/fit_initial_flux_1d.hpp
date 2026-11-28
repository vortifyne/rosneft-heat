#pragma once

#include "basin_1d/forward/basin_forward_solver_1d.hpp"
#include "numerics/optimization/bounded_least_squares_problem.hpp"

#include <iosfwd>

// The space supplies the 2D knot ages and the constant 1D starting flux (W/m²).
[[nodiscard]] Vector fit_initial_flux_1d(const BasinForwardSolver1D& section,
                                         const BasinHeatParameterSpace& space,
                                         const BasinHeatObservationData& observations,
                                         std::span<const std::string> wells,
                                         const LeastSquaresSolveRequest& options,
                                         std::ostream& log);
