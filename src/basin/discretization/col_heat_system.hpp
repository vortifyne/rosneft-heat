#pragma once

#include "basin/discretization/col_flux.hpp"
#include "basin/physics/thermal_properties.hpp"
#include "numerics/time/semi_discrete_system.hpp"
#include "numerics/timing.hpp"

#include <span>
#include <vector>

class ColHeatSystem final : public SemiDiscreteSystem {
public:
    ColHeatSystem(const ColGrid& grid, std::span<const double> lambda, std::span<const double> A,
                  HeatBoundaryData bc, std::span<const double> rho_c = {});

    // Material identities are fixed for an epoch, porosity and geometry for a step.
    // The caller keeps the supplied material arrays alive during assembly.
    ColHeatSystem(const ColGrid& grid, std::span<const double> phi,
                  std::span<const LithotypeProperties* const> lithotypes, FluidProperties fluid,
                  HeatBoundaryData bc);

    void set_flux_scheme(MultipointFluxScheme scheme) noexcept;
    [[nodiscard]] Vector::Index size() const noexcept;

    [[nodiscard]] const AssemblyTimings& property_timings() const noexcept {
        return property_timings_;
    }

    void assemble_residual(const Vector& T, Vector& r) const;
    void assemble_matrix(const Vector& T, SparseMatrix& M) const;

    void assemble_residual(double t, const Vector& T, const Vector& dT_dt,
                           Vector& r) const override;
    void assemble_matrix(NonlinearMethod method, double t, const Vector& T, const Vector& dT_dt,
                         double alpha, SparseMatrix& M) const override;

private:
    void update_coefficients(const Vector& T, double& property_s) const;
    [[nodiscard]] double horizontal_transmissibility(const HorizontalFace& face) const;

    void add_vertical_flux_residual(const Vector& T, Vector& r) const;
    void add_vertical_flux_matrix(const Vector& T) const;
    void add_surface_step_residual(const Vector& T, Vector& r) const;
    void add_surface_step_matrix() const;
    void add_horizontal_flux_residual(const Vector& T, Vector& r) const;
    void add_horizontal_flux_matrix() const;
    void add_boundary_residual(const Vector& T, Vector& r) const;
    void add_boundary_matrix() const;
    void add_source_residual(Vector& r) const;
    void add_accumulation_residual(const Vector& dT_dt, Vector& r) const;
    void add_accumulation_matrix(double alpha) const;
    void assemble_matrix(const Vector& T, double alpha, SparseMatrix& M) const;

    const ColGrid* grid_ = nullptr;
    std::span<const double> phi_;
    std::span<const LithotypeProperties* const> lithotypes_;
    FluidProperties fluid_{};
    mutable std::vector<double> lambda_;
    mutable std::vector<double> A_;
    mutable std::vector<double> rho_c_;
    HeatBoundaryData bc_;
    MultipointFluxScheme scheme_ = MultipointFluxScheme::linear;
    mutable std::vector<MatrixTriplet> triplets_;
    mutable AssemblyTimings property_timings_;
};
