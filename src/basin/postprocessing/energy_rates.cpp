#include "basin/postprocessing/energy_rates.hpp"

#include "basin/discretization/col_flux.hpp"
#include "basin/mesh/col_grid.hpp"

#include <cstddef>

EnergyRates energy_rates(const ColGrid& grid, std::span<const double> T,
                         std::span<const double> lambda, std::span<const double> A,
                         const HeatBoundaryData& bc) {
    EnergyRates result;
    FluxStencil stencil;
    for (const auto& face : grid.surface_vertical_faces) {
        compute_surface_flux(grid, face, lambda, bc, stencil);
        result.surface_outflow += evaluate_flux(stencil, T);
    }
    for (const std::size_t cell_index : grid.top_cells) {
        const ColCell& cell = grid.cells[cell_index];
        const double transmissibility = lambda[cell_index] * cell.width() / (0.5 * cell.height());
        result.surface_outflow +=
            transmissibility * (T[cell_index] - bc.T_surf({cell.center().x, cell.z_top}));
    }
    for (const std::size_t cell_index : grid.bottom_cells) {
        const ColCell& cell = grid.cells[cell_index];
        result.basal_inflow += bc.q_basal({cell.center().x, cell.z_bottom}) * cell.width();
    }
    const auto add_lateral = [&result](const auto& flux_function, const auto& column,
                                       const auto& cells) {
        if (!flux_function) {
            return;
        }
        for (const std::size_t cell_index : column) {
            const ColCell& cell = cells[cell_index];
            result.lateral_inflow += flux_function(cell.center()) * cell.height();
        }
    };
    add_lateral(bc.q_left, grid.columns.front(), grid.cells);
    add_lateral(bc.q_right, grid.columns.back(), grid.cells);
    for (const ColCell& cell : grid.cells) {
        result.source += A[cell.index] * cell.area();
    }
    return result;
}
