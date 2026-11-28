#include "basin_1d/discretization/heat_system_1d.hpp"

#include <cmath>
#include <stdexcept>

HeatSystem1D::HeatSystem1D(std::span<const double> faces, std::span<const double> porosity,
                           std::span<const LithotypeProperties* const> lithotypes,
                           FluidProperties fluid, double surface_temperature, double basal_flux)
    : porosity_(porosity), lithotypes_(lithotypes), fluid_(fluid),
      surface_temperature_(surface_temperature), basal_flux_(basal_flux), dz_(porosity.size()),
      conductance_(dz_.size()), mass_(dz_.size()), source_(dz_.size()) {
    if (dz_.empty() || faces.size() != dz_.size() + 1 || lithotypes.size() != dz_.size())
        throw std::invalid_argument("Expected one material and porosity per 1D cell");
    for (std::size_t i = 0; i < dz_.size(); ++i) {
        dz_[i] = faces[i + 1] - faces[i];
        if (!std::isfinite(dz_[i]) || dz_[i] <= 0. || !lithotypes[i])
            throw std::invalid_argument("Expected positive 1D cell length and a material");
    }
}

void HeatSystem1D::update_coefficients(const Vector& T) const {
    if (static_cast<std::size_t>(T.size()) != dz_.size())
        throw std::invalid_argument("Expected one temperature per 1D cell");
    double previous_resistance = 0.;
    for (std::size_t i = 0; i < dz_.size(); ++i) {
        const auto p = thermal_properties(T[i], porosity_[i], *lithotypes_[i], fluid_);
        const double resistance = 0.5 * dz_[i] / p.lambda;
        conductance_[i] = 1. / (previous_resistance + resistance);
        previous_resistance = resistance;
        mass_[i] = dz_[i] * p.rho_c;
        source_[i] = dz_[i] * p.A;
    }
}

void HeatSystem1D::assemble_residual(const Vector& T, Vector& r) const {
    assemble_balance(T, nullptr, r);
}

void HeatSystem1D::assemble_residual(double, const Vector& T, const Vector& dT_dt,
                                     Vector& r) const {
    if (dT_dt.size() != T.size())
        throw std::invalid_argument("Invalid 1D temperature derivative size");
    assemble_balance(T, &dT_dt, r);
}

void HeatSystem1D::assemble_balance(const Vector& T, const Vector* dT_dt, Vector& r) const {
    update_coefficients(T);
    r.resize(T.size());
    for (Vector::Index i = 0; i < T.size(); ++i)
        r[i] = -source_[i];
    if (dT_dt)
        for (Vector::Index i = 0; i < T.size(); ++i)
            r[i] += mass_[i] * (*dT_dt)[i];
    r[0] += conductance_[0] * (T[0] - surface_temperature_);
    for (Vector::Index i = 1; i < T.size(); ++i) {
        const double flux = conductance_[i] * (T[i - 1] - T[i]);
        r[i - 1] += flux;
        r[i] -= flux;
    }
    r[T.size() - 1] -= basal_flux_;
}

void HeatSystem1D::assemble_bands(const Vector& T, double alpha, SparseMatrix& A) const {
    update_coefficients(T);
    auto& bands = A.native_tridiagonal();
    bands.resize(T.size(), T.size());
    auto diag = bands.diagonal();
    auto lower = bands.lower();
    auto upper = bands.upper();
    for (Vector::Index i = 0; i < T.size(); ++i)
        diag[i] = conductance_[i] + alpha * mass_[i];
    for (Vector::Index i = 1; i < T.size(); ++i) {
        diag[i - 1] += conductance_[i];
        upper[i - 1] = lower[i - 1] = -conductance_[i];
    }
}

void HeatSystem1D::assemble_picard_matrix(const Vector& T, SparseMatrix& A) const {
    assemble_bands(T, 0., A);
}
void HeatSystem1D::assemble_newton_matrix(const Vector&, SparseMatrix&) const {
    throw std::invalid_argument("1D heat system supports Picard only");
}
void HeatSystem1D::assemble_matrix(NonlinearMethod method, double, const Vector& T, const Vector&,
                                   double alpha, SparseMatrix& A) const {
    if (method != NonlinearMethod::picard)
        throw std::invalid_argument("1D heat system supports Picard only");
    assemble_bands(T, alpha, A);
}
