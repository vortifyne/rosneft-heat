#include "basin/discretization/col_flux.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

void add_term(FluxStencil& stencil, const std::size_t P, const double a) {
    if (a == 0.0) {
        return;
    }
    for (std::size_t i = 0; i < stencil.size; ++i) {
        if (stencil.cells[i] == P) {
            stencil.a[i] += a;
            return;
        }
    }
    stencil.cells[stencil.size] = P;
    stencil.a[stencil.size++] = a;
}

void add_scaled(FluxStencil& target, const FluxStencil& source, const double scale) {
    for (std::size_t i = 0; i < source.size; ++i) {
        add_term(target, source.cells[i], scale * source.a[i]);
    }
    target.b += scale * source.b;
}

FluxStencil difference(const std::size_t P, const FluxStencil& sample) {
    FluxStencil result;
    add_term(result, P, 1.0);
    add_scaled(result, sample, -1.0);
    return result;
}

double vertical_sample(const ColGrid& grid, const std::size_t P, const bool upward,
                       const std::span<const double> lambda, const HeatBoundaryData& bc,
                       FluxStencil& sample) {
    const ColCell& cell_P = grid.cells[P];
    const Point2D p = cell_P.center();
    const auto N = upward ? cell_P.upward : cell_P.downward;
    const double z_f = upward ? cell_P.z_top : cell_P.z_bottom;
    const double d_P = std::abs(z_f - p.z);
    if (N) {
        const ColCell& cell_N = grid.cells[*N];
        const double d_N = std::abs(cell_N.center().z - z_f);
        const double denom = lambda[P] * d_N + lambda[*N] * d_P;
        const double w_N = lambda[*N] * d_P / denom;
        add_term(sample, P, 1.0 - w_N);
        add_term(sample, *N, w_N);
    } else if (upward) {
        sample.b = bc.T_surf({p.x, z_f});
    } else {
        add_term(sample, P, 1.0);
        sample.b = d_P * bc.q_basal({p.x, z_f}) / lambda[P];
    }
    return z_f - p.z;
}

double one_sided_flux(const ColGrid& grid, const std::size_t P, const std::size_t N,
                      const double x_f, const std::span<const double> lambda,
                      const HeatBoundaryData& bc, FluxStencil& stencil) {
    const Point2D p = grid.cells[P].center();
    const Point2D n = grid.cells[N].center();
    const double d_P = std::abs(x_f - p.x);
    const double d_N = std::abs(n.x - x_f);
    const double denom = lambda[P] * d_N + lambda[N] * d_P;
    if (!(d_P > 0.0) || !(d_N > 0.0) || !(denom > 0.0)) {
        throw std::runtime_error("Invalid nonmatching vertical-interface geometry");
    }
    const double w_N = lambda[N] * d_P / denom;
    FluxStencil target;
    add_term(target, P, 1.0 - w_N);
    add_term(target, N, w_N);
    const double z_h = (lambda[P] * d_N * p.z + lambda[N] * d_P * n.z) / denom;
    const Point2D d{x_f - p.x, z_h - p.z};
    const double a_x = lambda[P] / std::abs(d.x);
    FluxStencil vertical;
    const double dz = vertical_sample(grid, P, d.z >= 0.0, lambda, bc, vertical);
    double a_z = -a_x * d.z / dz;
    const double scale = std::max(1.0, a_x);
    if (a_z < -1.0e-12 * scale) {
        throw std::runtime_error("Nonmatching column has no positive vertical flux stencil");
    }
    a_z = std::max(0.0, a_z);
    stencil = difference(P, target);
    for (std::size_t i = 0; i < stencil.size; ++i) {
        stencil.a[i] *= a_x;
    }
    stencil.b *= a_x;
    add_scaled(stencil, difference(P, vertical), a_z);
    return a_x * w_N;
}

} // namespace

double evaluate_flux(const FluxStencil& stencil, const std::span<const double> T) {
    double result = stencil.b;
    for (std::size_t i = 0; i < stencil.size; ++i) {
        result += stencil.a[i] * T[stencil.cells[i]];
    }
    return result;
}

void compute_internal_flux(const ColGrid& grid, const VerticalSubface& face,
                           const MultipointFluxScheme scheme, const std::span<const double> lambda,
                           const std::span<const double> T, const HeatBoundaryData& bc,
                           FluxStencil& stencil) {
    const auto P = face.left_cell;
    const auto N = face.right_cell;
    stencil.size = 0;
    stencil.b = 0.0;
    FluxStencil s_P;
    FluxStencil s_N;
    const double a_P = one_sided_flux(grid, P, N, face.x, lambda, bc, s_P);
    const double a_N = one_sided_flux(grid, N, P, face.x, lambda, bc, s_N);
    if (scheme == MultipointFluxScheme::linear) {
        // Harmonic-point weights; 1/2 is valid only for equal normal conductances.
        const double d_P = std::abs(face.x - grid.cells[P].center().x);
        const double d_N = std::abs(grid.cells[N].center().x - face.x);
        const double g_P = lambda[P] / d_P;
        const double g_N = lambda[N] / d_N;
        const double g_sum = g_P + g_N;
        const double w_P = g_N / g_sum;
        const double w_N = g_P / g_sum;
        add_scaled(stencil, s_P, w_P * face.length());
        add_scaled(stencil, s_N, -w_N * face.length());
    } else {
        const double q_P = evaluate_flux(s_P, T);
        const double q_N = evaluate_flux(s_N, T);
        const double eps =
            std::numeric_limits<double>::epsilon() * std::max({1.0, std::abs(q_P), std::abs(q_N)});
        if (q_P * q_N <= 0.0) {
            const double denom = std::abs(q_P) + std::abs(q_N) + 2.0 * eps;
            const double w_P = (std::abs(q_N) + eps) / denom;
            const double w_N = (std::abs(q_P) + eps) / denom;
            add_scaled(stencil, s_P, face.length() * w_P);
            add_scaled(stencil, s_N, -face.length() * w_N);
        } else {
            FluxStencil r_P = s_P;
            add_term(r_P, P, -a_P);
            add_term(r_P, N, a_P);
            FluxStencil r_N = s_N;
            add_term(r_N, N, -a_N);
            add_term(r_N, P, a_N);
            const double v_P = evaluate_flux(r_P, T);
            const double v_N = evaluate_flux(r_N, T);
            const double eps_r = std::numeric_limits<double>::epsilon() *
                                 std::max({1.0, std::abs(v_P), std::abs(v_N)});
            const double denom = std::abs(v_P) + std::abs(v_N) + 2.0 * eps_r;
            const double w_P = (std::abs(v_N) + eps_r) / denom;
            const double w_N = (std::abs(v_P) + eps_r) / denom;
            const double a_PN = w_P * a_P + w_N * a_N;
            add_term(stencil, P, face.length() * a_PN);
            add_term(stencil, N, -face.length() * a_PN);
            if (v_P * v_N <= 0.0) {
                add_scaled(stencil, r_P, face.length() * w_P);
                add_scaled(stencil, r_N, -face.length() * w_N);
            }
        }
    }
}

void compute_surface_flux(const ColGrid& grid, const VerticalBoundaryFace& face,
                          const std::span<const double> lambda, const HeatBoundaryData& bc,
                          FluxStencil& stencil) {
    const auto P = face.cell;
    const auto& cell_P = grid.cells[P];
    const Point2D p = cell_P.center();
    const Point2D p_f{face.x, face.center_z()};
    const Point2D d{p_f.x - p.x, p_f.z - p.z};
    if (!(d.x * face.outward_x > 0.0)) {
        throw std::runtime_error("Invalid vertical surface-step geometry");
    }
    FluxStencil boundary_value;
    boundary_value.b = bc.T_surf(p_f);
    const double a_x = lambda[P] / std::abs(d.x);
    stencil = difference(P, boundary_value);
    for (std::size_t i = 0; i < stencil.size; ++i) {
        stencil.a[i] *= a_x;
    }
    stencil.b *= a_x;

    const double dz_tol = 1.0e-14 * std::max({1.0, cell_P.height(), std::abs(d.z)});
    if (std::abs(d.z) > dz_tol) {
        FluxStencil vertical;
        const double dz = vertical_sample(grid, P, d.z > 0.0, lambda, bc, vertical);
        double a_z = -a_x * d.z / dz;
        const double scale = std::max(1.0, a_x);
        if (a_z < -1.0e-12 * scale) {
            throw std::runtime_error("Surface step has no positive vertical flux stencil");
        }
        a_z = std::max(0.0, a_z);
        add_scaled(stencil, difference(P, vertical), a_z);
    }
    for (std::size_t i = 0; i < stencil.size; ++i) {
        stencil.a[i] *= face.length();
    }
    stencil.b *= face.length();
}
