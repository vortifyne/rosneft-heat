#include "basin/discretization/col_heat_system.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

std::size_t matrix_triplet_capacity(const ColGrid& grid) {
    return 2 * grid.cells.size() + 8 * grid.vertical_subfaces.size() +
           2 * grid.surface_vertical_faces.size() + 4 * grid.horizontal_faces.size();
}

} // namespace

ColHeatSystem::ColHeatSystem(const ColGrid& grid, std::span<const double> lambda,
                             std::span<const double> A, HeatBoundaryData bc,
                             std::span<const double> rho_c)
    : grid_(&grid), lambda_(lambda.begin(), lambda.end()), A_(A.begin(), A.end()),
      rho_c_(rho_c.begin(), rho_c.end()), bc_(std::move(bc)) {
    if (lambda.size() != grid.cells.size() || A.size() != grid.cells.size() ||
        (!rho_c.empty() && rho_c.size() != grid.cells.size())) {
        throw std::invalid_argument("Nonmatching-column field sizes do not match the grid");
    }
    if (!bc_.T_surf || !bc_.q_basal) {
        throw std::invalid_argument("Nonmatching-column boundary functions must be set");
    }
    triplets_.reserve(matrix_triplet_capacity(grid));
}

ColHeatSystem::ColHeatSystem(const ColGrid& grid, std::span<const double> phi,
                             std::span<const LithotypeProperties* const> lithotypes,
                             FluidProperties fluid, HeatBoundaryData bc)
    : grid_(&grid), phi_(phi), lithotypes_(lithotypes), fluid_(fluid), lambda_(grid.cells.size()),
      A_(grid.cells.size()), rho_c_(grid.cells.size()), bc_(std::move(bc)) {
    if (phi.size() != grid.cells.size() || lithotypes.size() != grid.cells.size()) {
        throw std::invalid_argument("Nonmatching-column material sizes do not match the grid");
    }
    if (!bc_.T_surf || !bc_.q_basal) {
        throw std::invalid_argument("Nonmatching-column boundary functions must be set");
    }
    triplets_.reserve(matrix_triplet_capacity(grid));
}

void ColHeatSystem::assemble_newton_matrix(const Vector&, SparseMatrix&) const {
    throw std::invalid_argument("Column-grid heat system currently supports Picard only");
}

void ColHeatSystem::set_flux_scheme(MultipointFluxScheme scheme) noexcept {
    scheme_ = scheme;
}

void ColHeatSystem::interpolate_horizontal_temperature(const Vector& T,
                                                       const HeatBoundaryData& boundaries,
                                                       std::span<double> node_T) const {
    double seconds = 0.0;
    update_coefficients(T, seconds);
    ::interpolate_horizontal_temperature(*grid_,
                                         {T.native().data(), static_cast<std::size_t>(T.size())},
                                         lambda_, boundaries, node_T);
}

Eigen::MatrixXd ColHeatSystem::horizontal_face_sensitivities(const Vector& T,
                                                             const Eigen::MatrixXd& cell_S,
                                                             const Vector& dq_deta) const {
    double seconds = 0.0;
    update_coefficients(T, seconds);
    return ::horizontal_face_sensitivities(*grid_, lambda_, cell_S, dq_deta);
}

Vector::Index ColHeatSystem::size() const noexcept {
    return static_cast<Vector::Index>(grid_->cells.size());
}

void ColHeatSystem::update_coefficients(const Vector& T, double& property_s) const {
    if (T.size() != size()) {
        throw std::invalid_argument("Nonmatching-column solution size does not match the grid");
    }
    if (!lithotypes_.empty()) {
        const auto start = Clock::now();
        update_thermal_fields({T.native().data(), static_cast<std::size_t>(T.size())}, phi_,
                              lithotypes_, fluid_, lambda_, rho_c_, A_);
        property_s += elapsed_seconds(start);
    }
}

double ColHeatSystem::horizontal_transmissibility(const HorizontalFace& face) const {
    const auto P = face.upper_cell;
    const auto N = face.lower_cell;
    const auto& cell_P = grid_->cells[P];
    const auto& cell_N = grid_->cells[N];
    const double d_P = face.z - cell_P.center().z;
    const double d_N = cell_N.center().z - face.z;
    return face.length / ((d_P / lambda_[P]) + (d_N / lambda_[N]));
}

void ColHeatSystem::assemble_residual(const Vector& T, Vector& r) const {
    update_coefficients(T, property_timings_.residual_s);
    r.resize(size());
    r.set_zero();
    if (scheme_ == MultipointFluxScheme::linear)
        add_mpfa_residual(T, r);
    else
        add_nmpfa_residual(T, r);
    add_surface_step_residual(T, r);
    add_horizontal_flux_residual(T, r);
    add_boundary_residual(T, r);
    add_source_residual(r);
}

void ColHeatSystem::assemble_residual(double, const Vector& T, const Vector& dT_dt,
                                      Vector& r) const {
    if (dT_dt.size() != size()) {
        throw std::invalid_argument("Column-grid temperature derivative size does not match");
    }
    assemble_residual(T, r);
    add_accumulation_residual(dT_dt, r);
}

void ColHeatSystem::assemble_matrix(const Vector& T, SparseMatrix& M) const {
    assemble_matrix(T, 0.0, M);
}

void ColHeatSystem::assemble_matrix(NonlinearMethod method, double, const Vector& T,
                                    const Vector& dT_dt, double alpha, SparseMatrix& M) const {
    if (method != NonlinearMethod::picard) {
        throw std::invalid_argument("Column-grid heat system currently supports Picard only");
    }
    if (dT_dt.size() != size() || !std::isfinite(alpha) || alpha < 0.0) {
        throw std::invalid_argument("Invalid column-grid derivative or BDF shift");
    }
    assemble_matrix(T, alpha, M);
}

void ColHeatSystem::assemble_matrix(const Vector& T, double alpha, SparseMatrix& M) const {
    update_coefficients(T, property_timings_.matrix_s);
    triplets_.clear();
    if (scheme_ == MultipointFluxScheme::linear)
        add_mpfa_matrix();
    else
        add_nmpfa_matrix(T);
    add_surface_step_matrix();
    add_horizontal_flux_matrix();
    add_boundary_matrix();
    // Prescribed sources and Neumann data have no matrix contribution.
    add_accumulation_matrix(alpha);
    M.resize(size(), size());
    M.set_from_triplets(triplets_);
}

Eigen::MatrixXd ColHeatSystem::assemble_ddm_step(const Vector& T, double dt,
                                                 const Eigen::MatrixXd& previous,
                                                 const Vector& dq_deta,
                                                 SparseMatrix& matrix) const {
    if (!(dt > 0.0) || !std::isfinite(dt) || previous.rows() != size() ||
        previous.cols() != dq_deta.size() || dq_deta.empty() || rho_c_.empty())
        throw std::invalid_argument("Invalid BDF1 sensitivity dimensions, capacity or timestep");
    const double alpha = 1.0 / dt;
    const Vector basal = assemble_ddm_matrix(T, alpha, matrix);
    Eigen::MatrixXd rhs(size(), previous.cols());
    for (const auto& cell : grid_->cells) {
        const auto i = cell.index;
        rhs.row(i) = alpha * cell.area() * rho_c_[i] * previous.row(i) +
                     basal[i] * dq_deta.native().transpose();
    }
    return rhs;
}

Vector ColHeatSystem::assemble_ddm_matrix(const Vector& T, double alpha, SparseMatrix& matrix) const {
    // Picard matrix freezes coefficients at T; use Newton matrix for exact derivatives.
    assemble_matrix(T, alpha, matrix);

    Vector basal(size());
    basal.set_zero();
    for (const auto cell : grid_->bottom_cells)
        basal[cell] += grid_->cells[cell].width();

    // Unit basal flux and zero surface temperature isolate the MPFA boundary constant.
    const HeatBoundaryData unit{.T_surf = [](Point2D) { return 0.0; },
                                .q_basal = [](Point2D) { return 1.0; }};
    const std::span<const double> values{T.native().data(), static_cast<std::size_t>(T.size())};
    FluxStencil stencil;
    for (const auto& face : grid_->vertical_subfaces) {
        if (grid_->cells[face.left_cell].downward && grid_->cells[face.right_cell].downward)
            continue;
        double slope;
        if (scheme_ == MultipointFluxScheme::linear) {
            compute_mpfa_flux(*grid_, face, lambda_, unit, stencil);
            slope = stencil.b;
        } else {
            compute_nmpfa_flux(*grid_, face, lambda_, values, bc_, stencil, &slope);
        }
        basal[face.left_cell] -= slope;
        basal[face.right_cell] += slope;
    }
    for (const auto& face : grid_->surface_vertical_faces) {
        if (!grid_->cells[face.cell].downward) {
            compute_surface_flux(*grid_, face, lambda_, unit, stencil);
            basal[face.cell] -= stencil.b;
        }
    }
    return basal;
}

void ColHeatSystem::add_mpfa_residual(const Vector& T, Vector& r) const {
    const std::span<const double> values{T.native().data(), static_cast<std::size_t>(T.size())};
    FluxStencil stencil;
    for (const auto& face : grid_->vertical_subfaces) {
        const auto P = face.left_cell;
        const auto N = face.right_cell;
        compute_mpfa_flux(*grid_, face, lambda_, bc_, stencil);
        const double F = evaluate_flux(stencil, values);
        r[static_cast<Vector::Index>(P)] += F;
        r[static_cast<Vector::Index>(N)] -= F;
    }
}

void ColHeatSystem::add_nmpfa_residual(const Vector& T, Vector& r) const {
    const std::span<const double> values{T.native().data(), static_cast<std::size_t>(T.size())};
    FluxStencil stencil;
    for (const auto& face : grid_->vertical_subfaces) {
        compute_nmpfa_flux(*grid_, face, lambda_, values, bc_, stencil);
        const double F = evaluate_flux(stencil, values);
        r[static_cast<Vector::Index>(face.left_cell)] += F;
        r[static_cast<Vector::Index>(face.right_cell)] -= F;
    }
}

void ColHeatSystem::add_mpfa_matrix() const {
    FluxStencil stencil;
    for (const auto& face : grid_->vertical_subfaces) {
        const auto P = face.left_cell;
        const auto N = face.right_cell;
        compute_mpfa_flux(*grid_, face, lambda_, bc_, stencil);
        for (std::size_t i = 0; i < stencil.size; ++i) {
            triplets_.emplace_back(static_cast<Vector::Index>(P),
                                   static_cast<Vector::Index>(stencil.cells[i]), stencil.a[i]);
        }
        for (std::size_t i = 0; i < stencil.size; ++i) {
            triplets_.emplace_back(static_cast<Vector::Index>(N),
                                   static_cast<Vector::Index>(stencil.cells[i]), -stencil.a[i]);
        }
    }
}

void ColHeatSystem::add_nmpfa_matrix(const Vector& T) const {
    const std::span<const double> values{T.native().data(), static_cast<std::size_t>(T.size())};
    FluxStencil stencil;
    for (const auto& face : grid_->vertical_subfaces) {
        compute_nmpfa_flux(*grid_, face, lambda_, values, bc_, stencil);
        for (std::size_t i = 0; i < stencil.size; ++i) {
            triplets_.emplace_back(static_cast<Vector::Index>(face.left_cell),
                                   static_cast<Vector::Index>(stencil.cells[i]), stencil.a[i]);
            triplets_.emplace_back(static_cast<Vector::Index>(face.right_cell),
                                   static_cast<Vector::Index>(stencil.cells[i]), -stencil.a[i]);
        }
    }
}

void ColHeatSystem::add_surface_step_residual(const Vector& T, Vector& r) const {
    const std::span<const double> values{T.native().data(), static_cast<std::size_t>(T.size())};
    FluxStencil stencil;
    for (const auto& face : grid_->surface_vertical_faces) {
        compute_surface_flux(*grid_, face, lambda_, bc_, stencil);
        r[static_cast<Vector::Index>(face.cell)] += evaluate_flux(stencil, values);
    }
}

void ColHeatSystem::add_surface_step_matrix() const {
    FluxStencil stencil;
    for (const auto& face : grid_->surface_vertical_faces) {
        compute_surface_flux(*grid_, face, lambda_, bc_, stencil);
        for (std::size_t i = 0; i < stencil.size; ++i) {
            triplets_.emplace_back(static_cast<Vector::Index>(face.cell),
                                   static_cast<Vector::Index>(stencil.cells[i]), stencil.a[i]);
        }
    }
}

void ColHeatSystem::add_horizontal_flux_residual(const Vector& T, Vector& r) const {
    for (const auto& face : grid_->horizontal_faces) {
        const auto P = face.upper_cell;
        const auto N = face.lower_cell;
        const double F = horizontal_transmissibility(face) *
                         (T[static_cast<Vector::Index>(P)] - T[static_cast<Vector::Index>(N)]);
        r[static_cast<Vector::Index>(P)] += F;
        r[static_cast<Vector::Index>(N)] -= F;
    }
}

void ColHeatSystem::add_horizontal_flux_matrix() const {
    for (const auto& face : grid_->horizontal_faces) {
        const auto P = face.upper_cell;
        const auto N = face.lower_cell;
        const double a_PN = horizontal_transmissibility(face);
        triplets_.emplace_back(static_cast<Vector::Index>(P), static_cast<Vector::Index>(P), a_PN);
        triplets_.emplace_back(static_cast<Vector::Index>(P), static_cast<Vector::Index>(N), -a_PN);
        triplets_.emplace_back(static_cast<Vector::Index>(N), static_cast<Vector::Index>(P), -a_PN);
        triplets_.emplace_back(static_cast<Vector::Index>(N), static_cast<Vector::Index>(N), a_PN);
    }
}

void ColHeatSystem::add_boundary_residual(const Vector& T, Vector& r) const {
    for (const std::size_t P : grid_->top_cells) {
        const auto& cell = grid_->cells[P];
        const double a_PN = lambda_[P] * cell.width() / (0.5 * cell.height());
        r[static_cast<Vector::Index>(P)] +=
            a_PN * (T[static_cast<Vector::Index>(P)] - bc_.T_surf({cell.center().x, cell.z_top}));
    }
    for (const std::size_t P : grid_->bottom_cells) {
        const auto& cell = grid_->cells[P];
        r[static_cast<Vector::Index>(P)] -=
            bc_.q_basal({cell.center().x, cell.z_bottom}) * cell.width();
    }
    const auto add_lateral = [&r](const auto& flux_function, const auto& column,
                                  const auto& cells) {
        if (!flux_function) {
            return;
        }
        for (const std::size_t P : column) {
            const auto& cell = cells[P];
            r[static_cast<Vector::Index>(P)] -= flux_function(cell.center()) * cell.height();
        }
    };
    add_lateral(bc_.q_left, grid_->columns.front(), grid_->cells);
    add_lateral(bc_.q_right, grid_->columns.back(), grid_->cells);
}

void ColHeatSystem::add_boundary_matrix() const {
    for (const std::size_t P : grid_->top_cells) {
        const auto& cell = grid_->cells[P];
        const double a_PN = lambda_[P] * cell.width() / (0.5 * cell.height());
        triplets_.emplace_back(static_cast<Vector::Index>(P), static_cast<Vector::Index>(P), a_PN);
    }
}

void ColHeatSystem::add_source_residual(Vector& r) const {
    for (const auto& cell : grid_->cells) {
        r[static_cast<Vector::Index>(cell.index)] -= A_[cell.index] * cell.area();
    }
}

void ColHeatSystem::add_accumulation_residual(const Vector& dT_dt, Vector& r) const {
    if (rho_c_.empty()) {
        return;
    }
    for (const auto& cell : grid_->cells) {
        const auto index = static_cast<Vector::Index>(cell.index);
        r[index] += cell.area() * rho_c_[cell.index] * dT_dt[index];
    }
}

void ColHeatSystem::add_accumulation_matrix(double alpha) const {
    if (alpha == 0.0 || rho_c_.empty()) {
        return;
    }
    for (const auto& cell : grid_->cells) {
        const auto index = static_cast<Vector::Index>(cell.index);
        triplets_.emplace_back(index, index, alpha * cell.area() * rho_c_[cell.index]);
    }
}
