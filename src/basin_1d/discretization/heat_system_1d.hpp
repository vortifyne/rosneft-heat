#pragma once
#include "basin/physics/thermal_properties.hpp"
#include "numerics/nonlinear/nonlinear_system.hpp"
#include "numerics/time/semi_discrete_system.hpp"

#include <vector>

class HeatSystem1D final : public NonlinearSystem, public SemiDiscreteSystem {
public:
    // Geometry and boundary values are fixed on one step. Material spans must outlive the system.
    HeatSystem1D(std::span<const double> faces, std::span<const double> porosity,
                 std::span<const LithotypeProperties* const> lithotypes, FluidProperties fluid,
                 double surface_temperature, double basal_flux);

    bool has_fixed_matrix_pattern() const noexcept override {
        return true;
    }

    void assemble_residual(const Vector& T, Vector& r) const override;
    void assemble_residual(double t, const Vector& T, const Vector& dT_dt,
                           Vector& r) const override;
    void assemble_matrix(NonlinearMethod method, double t, const Vector& T, const Vector& dT_dt,
                         double alpha, SparseMatrix& A) const override;

    Eigen::MatrixXd assemble_ddm(const Vector& T, double alpha, const Eigen::MatrixXd& previous,
                                 const Vector& dq, SparseMatrix& A) const;
    Vector face_temperatures(const Vector& T, double end_flux, const Eigen::MatrixXd& S,
                             const Vector& dq, Eigen::MatrixXd* face_S) const;

private:
    void assemble_picard_matrix(const Vector& T, SparseMatrix& A) const override;
    void assemble_newton_matrix(const Vector&, SparseMatrix&) const override;
    void update_coefficients(const Vector& T) const;
    void assemble_balance(const Vector& T, const Vector* dT_dt, Vector& r) const;
    void assemble_bands(const Vector& T, double alpha, SparseMatrix& A) const;

    std::span<const double> porosity_;
    std::span<const LithotypeProperties* const> lithotypes_;
    FluidProperties fluid_;
    double surface_temperature_, basal_flux_;
    std::vector<double> dz_;
    mutable std::vector<double> conductance_, mass_, source_, resistance_;
};
