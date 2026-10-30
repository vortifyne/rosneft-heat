#pragma once

#include "basin/discretization/nonmatching_column_flux.hpp"
#include "basin/physics/thermophysical_properties.hpp"
#include "numerics/time/semi_discrete_system.hpp"

#include <span>
#include <vector>

class NonmatchingColumnHeatSystem final : public SemiDiscreteSystem {
public:
    NonmatchingColumnHeatSystem(const NonmatchingColumnGrid& grid,
                                std::span<const double> thermal_conductivity,
                                std::span<const double> heat_production,
                                NonmatchingColumnBoundaryData boundary_data,
                                std::span<const double> volumetric_heat_capacity = {});

    // Material identities are fixed for an epoch, porosity and geometry for a step.
    // The caller keeps the supplied material arrays alive during assembly.
    NonmatchingColumnHeatSystem(
        const NonmatchingColumnGrid& grid, std::span<const double> porosity,
        std::span<const LithotypeThermophysicalProperties* const> lithotypes,
        FluidThermophysicalProperties fluid, NonmatchingColumnBoundaryData boundary_data);

    void set_flux_scheme(MultipointFluxScheme scheme) noexcept;
    [[nodiscard]] Vector::Index size() const noexcept;

    void assemble_residual(const Vector& solution, Vector& residual) const;
    void assemble_matrix(const Vector& solution, SparseMatrix& matrix) const;

    void assemble_residual(double time, const Vector& solution, const Vector& solution_derivative,
                           Vector& residual) const override;
    void assemble_matrix(NonlinearMethod method, double time, const Vector& solution,
                         const Vector& solution_derivative, double derivative_shift,
                         SparseMatrix& matrix) const override;

private:
    void update_coefficients(const Vector& solution) const;
    [[nodiscard]] double horizontal_transmissibility(const NonmatchingHorizontalFace& face) const;

    void add_vertical_flux_residual(const Vector& solution, Vector& residual) const;
    void add_vertical_flux_matrix(const Vector& solution) const;
    void add_surface_step_residual(const Vector& solution, Vector& residual) const;
    void add_surface_step_matrix() const;
    void add_horizontal_flux_residual(const Vector& solution, Vector& residual) const;
    void add_horizontal_flux_matrix() const;
    void add_boundary_residual(const Vector& solution, Vector& residual) const;
    void add_boundary_matrix() const;
    void add_source_residual(Vector& residual) const;
    void add_accumulation_residual(const Vector& derivative, Vector& residual) const;
    void add_accumulation_matrix(double derivative_shift) const;
    void assemble_matrix(const Vector& solution, double derivative_shift,
                         SparseMatrix& matrix) const;

    const NonmatchingColumnGrid* grid_ = nullptr;
    std::span<const double> porosity_;
    std::span<const LithotypeThermophysicalProperties* const> lithotypes_;
    FluidThermophysicalProperties fluid_{};
    mutable std::vector<double> thermal_conductivity_;
    mutable std::vector<double> heat_production_;
    mutable std::vector<double> volumetric_heat_capacity_;
    NonmatchingColumnBoundaryData boundary_data_;
    MultipointFluxScheme flux_scheme_ = MultipointFluxScheme::linear;
    mutable std::vector<MatrixTriplet> triplets_;
};
