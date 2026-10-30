#include "basin/postprocessing/energy_rates.hpp"

#include "basin/discretization/nonmatching_column_flux.hpp"
#include "basin/mesh/nonmatching_column_grid.hpp"

#include <cstddef>
#include <stdexcept>

NonmatchingColumnEnergyRates nonmatching_column_energy_rates(
    const NonmatchingColumnGrid& grid, const std::span<const double> temperature,
    const std::span<const double> thermal_conductivity,
    const std::span<const double> heat_production,
    const NonmatchingColumnBoundaryData& boundary_data, const MultipointFluxScheme scheme) {
    if (temperature.size() != grid.cells.size() ||
        thermal_conductivity.size() != grid.cells.size() ||
        heat_production.size() != grid.cells.size()) {
        throw std::invalid_argument("Energy-rate field sizes do not match the grid");
    }
    NonmatchingColumnEnergyRates result;
    const NonmatchingColumnFluxes vertical = NonmatchingColumnFlux(grid).compute(
        scheme, thermal_conductivity, temperature, boundary_data);
    for (const NonmatchingColumnBoundaryFlux& flux : vertical.surface) {
        result.surface_outflow += flux.expression.value(temperature);
    }
    for (const std::size_t cell_index : grid.top_cells) {
        const NonmatchingColumnCell& cell = grid.cells[cell_index];
        const double transmissibility =
            thermal_conductivity[cell_index] * cell.width() / (0.5 * cell.height());
        result.surface_outflow +=
            transmissibility * (temperature[cell_index] -
                                boundary_data.surface_temperature({cell.center().x, cell.z_top}));
    }
    for (const std::size_t cell_index : grid.bottom_cells) {
        const NonmatchingColumnCell& cell = grid.cells[cell_index];
        result.basal_inflow +=
            boundary_data.basal_inward_flux({cell.center().x, cell.z_bottom}) * cell.width();
    }
    const auto add_lateral = [&result](const auto& flux_function, const auto& column,
                                       const auto& cells) {
        if (!flux_function) {
            return;
        }
        for (const std::size_t cell_index : column) {
            const NonmatchingColumnCell& cell = cells[cell_index];
            result.lateral_inflow += flux_function(cell.center()) * cell.height();
        }
    };
    add_lateral(boundary_data.left_inward_flux, grid.columns.front(), grid.cells);
    add_lateral(boundary_data.right_inward_flux, grid.columns.back(), grid.cells);
    for (const NonmatchingColumnCell& cell : grid.cells) {
        result.heat_production += heat_production[cell.index] * cell.area();
    }
    return result;
}
