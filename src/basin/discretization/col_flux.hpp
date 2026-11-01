#pragma once

#include "basin/discretization/multipoint_flux_types.hpp"
#include "basin/mesh/col_grid.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <span>

struct HeatBoundaryData {
    std::function<double(Point2D)> T_surf;  // Surface temperature [K].
    std::function<double(Point2D)> q_basal; // Inward fluxes [W/m^2].
    std::function<double(Point2D)> q_left{};
    std::function<double(Point2D)> q_right{};
};

// sum(a[i] * T[cells[i]]) + b; at most four cells for MPFA and NMPFA.
struct FluxStencil {
    std::array<std::size_t, 4> cells{};
    std::array<double, 4> a{};
    std::size_t size = 0;
    double b = 0.0;
};

[[nodiscard]] double evaluate_flux(const FluxStencil& stencil, std::span<const double> T);

void compute_internal_flux(const ColGrid& grid, const VerticalSubface& face,
                           MultipointFluxScheme scheme, std::span<const double> lambda,
                           std::span<const double> T, const HeatBoundaryData& bc,
                           FluxStencil& stencil);

void compute_surface_flux(const ColGrid& grid, const VerticalBoundaryFace& face,
                          std::span<const double> lambda, const HeatBoundaryData& bc,
                          FluxStencil& stencil);
