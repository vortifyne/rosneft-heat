#pragma once

#include "basin/discretization/nonmatching_column_heat_system.hpp"
#include "numerics/time/semi_discrete_system.hpp"

#include <functional>
#include <span>
#include <vector>

// Cell-centred BDF1 operator on a C grid. The grid supplied for a time step is
// the geometry at the end of that step; the previous solution uses the same
// cell indices, even when their geometry has moved.
class NonmatchingColumnTransientSystem final : public SemiDiscreteSystem {
public:
    using CapacityUpdater = std::function<void(std::span<const double>, std::span<double>)>;

    NonmatchingColumnTransientSystem(
        const NonmatchingColumnGrid& grid, std::span<const double> thermal_conductivity,
        std::span<const double> heat_production, std::span<const double> volumetric_heat_capacity,
        NonmatchingColumnBoundaryData boundary_data,
        NonmatchingColumnHeatSystem::PropertyUpdater property_updater = {},
        CapacityUpdater capacity_updater = {});

    void set_flux_scheme(MultipointFluxScheme scheme);

    void assemble_residual(double time, const Vector& solution, const Vector& solution_derivative,
                           Vector& residual) const override;
    void assemble_matrix(NonlinearMethod method, double time, const Vector& solution,
                         const Vector& solution_derivative, double derivative_shift,
                         SparseMatrix& matrix) const override;

private:
    void update_capacity(const Vector& solution) const;

    const NonmatchingColumnGrid* grid_ = nullptr;
    NonmatchingColumnHeatSystem spatial_;
    mutable std::vector<double> volumetric_heat_capacity_;
    CapacityUpdater capacity_updater_;
};
