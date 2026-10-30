#pragma once

#include "basin/discretization/multipoint_flux_types.hpp"
#include "basin/mesh/nonmatching_column_grid.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <span>

struct NonmatchingColumnBoundaryData {
    std::function<double(Point2D)> surface_temperature;
    std::function<double(Point2D)> basal_inward_flux;
    std::function<double(Point2D)> left_inward_flux{};
    std::function<double(Point2D)> right_inward_flux{};
};

// Each side uses the two adjacent cells and at most one vertical neighbour.
// Their union therefore contains at most four cells, also for NMPFA remainders.
struct NonmatchingColumnFluxStencil {
    std::array<std::size_t, 4> cells{};
    std::array<double, 4> coefficients{};
    std::size_t size = 0;
    double constant = 0.0;
};

[[nodiscard]] double evaluate_flux(const NonmatchingColumnFluxStencil& stencil,
                                   std::span<const double> temperature);

void compute_internal_flux(const NonmatchingColumnGrid& grid,
                           const NonmatchingVerticalSubface& face, MultipointFluxScheme scheme,
                           std::span<const double> conductivity,
                           std::span<const double> temperature,
                           const NonmatchingColumnBoundaryData& boundary_data,
                           NonmatchingColumnFluxStencil& stencil);

void compute_surface_flux(const NonmatchingColumnGrid& grid,
                          const NonmatchingVerticalBoundaryFace& face,
                          std::span<const double> conductivity,
                          const NonmatchingColumnBoundaryData& boundary_data,
                          NonmatchingColumnFluxStencil& stencil);
