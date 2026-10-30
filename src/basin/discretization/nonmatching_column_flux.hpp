#pragma once

#include "basin/discretization/multipoint_flux_types.hpp"
#include "basin/mesh/nonmatching_column_grid.hpp"

#include <functional>
#include <span>
#include <vector>

struct NonmatchingColumnBoundaryData {
    std::function<double(Point2D)> surface_temperature;
    std::function<double(Point2D)> basal_inward_flux;
    std::function<double(Point2D)> left_inward_flux{};
    std::function<double(Point2D)> right_inward_flux{};
};

struct NonmatchingColumnInternalFlux {
    std::size_t left_cell = 0;
    std::size_t right_cell = 0;
    MultipointFluxExpression expression;
};

struct NonmatchingColumnBoundaryFlux {
    std::size_t cell = 0;
    MultipointFluxExpression expression;
};

struct NonmatchingColumnFluxes {
    std::vector<NonmatchingColumnInternalFlux> vertical;
    std::vector<NonmatchingColumnBoundaryFlux> surface;
};

class NonmatchingColumnFlux {
public:
    explicit NonmatchingColumnFlux(const NonmatchingColumnGrid& grid);

    [[nodiscard]] NonmatchingColumnFluxes
    compute(MultipointFluxScheme scheme, std::span<const double> conductivity,
            std::span<const double> solution,
            const NonmatchingColumnBoundaryData& boundary_data) const;

private:
    struct Sample;
    struct OneSidedFlux;

    [[nodiscard]] Sample target_sample(std::size_t cell, std::size_t neighbor, double interface_x,
                                       std::span<const double> conductivity) const;
    [[nodiscard]] Sample vertical_sample(std::size_t cell, bool upward,
                                         std::span<const double> conductivity,
                                         const NonmatchingColumnBoundaryData& boundary_data) const;
    [[nodiscard]] OneSidedFlux
    one_sided_flux(std::size_t cell, std::size_t neighbor, double interface_x,
                   std::span<const double> conductivity,
                   const NonmatchingColumnBoundaryData& boundary_data) const;
    [[nodiscard]] MultipointFluxExpression
    surface_boundary_flux(const NonmatchingVerticalBoundaryFace& face,
                          std::span<const double> conductivity,
                          const NonmatchingColumnBoundaryData& boundary_data) const;

    const NonmatchingColumnGrid* grid_ = nullptr;
};
