#include "discretization/model_heat_system.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void check_positive_finite(double value, const char* message) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::invalid_argument(message);
    }
}

void check_finite(double value, const char* message) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(message);
    }
}

} // namespace

ModelHeatSystem::ModelHeatSystem(const RegularGrid& grid, const ModelHeatProblem& problem)
    : grid_(grid), thermal_conductivity_(problem.lithotype.thermal_conductivity),
      volumetric_heat_capacity_(problem.lithotype.density * problem.lithotype.specific_heat),
      heat_production_(problem.lithotype.heat_production),
      surface_temperature_(problem.initial_boundary_conditions.surface_temperature),
      basal_heat_flux_(problem.initial_boundary_conditions.basal_heat_flux) {
    check_positive_finite(problem.lithotype.thermal_conductivity,
                          "Thermal conductivity must be finite and positive");
    check_positive_finite(problem.lithotype.density, "Density must be finite and positive");
    check_positive_finite(problem.lithotype.specific_heat,
                          "Specific heat must be finite and positive");
    check_finite(volumetric_heat_capacity_, "Volumetric heat capacity must be finite");
    check_finite(problem.lithotype.heat_production, "Heat production must be finite");
    check_finite(problem.initial_boundary_conditions.surface_temperature,
                 "Surface temperature must be finite");
    check_finite(problem.initial_boundary_conditions.basal_heat_flux,
                 "Basal heat flux must be finite");
}

Vector::Index ModelHeatSystem::size() const noexcept {
    return static_cast<Vector::Index>(grid_.cell_count());
}

void ModelHeatSystem::assemble_residual(double time, const Vector& solution,
                                        const Vector& solution_derivative, Vector& residual) const {
    check_finite(time, "Assembly time must be finite");
    check_vector_sizes(solution, solution_derivative);

    residual.resize(size());
    for (Vector::Index index = 0; index < size(); ++index) {
        // The cell balance is divided by the cell area, so every residual entry has units W/m^3:
        //     F_K = rho*c*Tdot_K - div(lambda*grad(T))_K - A.
        residual[index] =
            (volumetric_heat_capacity_ * solution_derivative[index]) - heat_production_;
    }

    // A two-point flux through an internal face is
    //     lambda * face_length * (T_neighbour - T_cell) / centre_distance.
    // Dividing it by the cell area gives lambda/dx^2 for a vertical face and lambda/dz^2
    // for a horizontal face. The geometric form is kept here to expose the FVM balance.
    const double vertical_face_coefficient =
        thermal_conductivity_ * grid_.dz * grid_.inv_dx * grid_.inv_cell_area;
    const double horizontal_face_coefficient =
        thermal_conductivity_ * grid_.dx * grid_.inv_dz * grid_.inv_cell_area;

    // The grid is regular, so direct index loops avoid constructing face and neighbour objects in
    // these hot assembly paths. Each internal face is visited once and contributes conservatively
    // to its two adjacent cells.
    for (int iz = 0; iz < grid_.nz; ++iz) {
        for (int ix = 0; ix + 1 < grid_.nx; ++ix) {
            const Vector::Index left = cell_index(ix, iz);
            const Vector::Index right = cell_index(ix + 1, iz);
            // The residual contains -div(lambda*grad(T)): the same face contribution enters the
            // two adjacent cells with opposite signs and therefore conserves heat locally.
            const double contribution =
                vertical_face_coefficient * (solution[left] - solution[right]);
            residual[left] += contribution;
            residual[right] -= contribution;
        }
    }

    for (int iz = 0; iz + 1 < grid_.nz; ++iz) {
        for (int ix = 0; ix < grid_.nx; ++ix) {
            const Vector::Index top = cell_index(ix, iz);
            const Vector::Index bottom = cell_index(ix, iz + 1);
            // The z-direction uses the same conservative two-point flux as the x-direction.
            const double contribution =
                horizontal_face_coefficient * (solution[top] - solution[bottom]);
            residual[top] += contribution;
            residual[bottom] -= contribution;
        }
    }

    // Cell centres are dz/2 away from the upper Dirichlet boundary. Hence its two-point
    // transmissibility is lambda*dx/(dz/2), which becomes 2*lambda/dz^2 after division by area.
    // This face-gradient reconstruction is only first-order pointwise at the boundary. On this
    // uniform orthogonal cell-centred grid the temperature solution for the current smooth model
    // still converges with second order; the convergence test guards that property separately.
    const double surface_face_coefficient = 2.0 * horizontal_face_coefficient;

    // At the bottom lambda*dT/dz = q_b is prescribed directly, so no one-sided temperature
    // difference is needed. Its normalized balance contribution is -q_b*dx/(dx*dz) = -q_b/dz;
    // the minus sign follows from F = rho*c*Tdot - div(lambda*grad(T)) - A.
    const double basal_flux_contribution = basal_heat_flux_ * grid_.dx * grid_.inv_cell_area;
    for (int ix = 0; ix < grid_.nx; ++ix) {
        const Vector::Index top = cell_index(ix, 0);
        residual[top] += surface_face_coefficient * (solution[top] - surface_temperature_);

        const Vector::Index bottom = cell_index(ix, grid_.nz - 1);
        residual[bottom] -= basal_flux_contribution;
    }

    // The prescribed zero flux on both lateral boundaries contributes nothing to the balance.
}

void ModelHeatSystem::assemble_matrix(NonlinearMethod method, double time, const Vector& solution,
                                      const Vector& solution_derivative, double derivative_shift,
                                      SparseMatrix& matrix) const {
    static_cast<void>(method);
    check_finite(time, "Assembly time must be finite");
    check_finite(derivative_shift, "Derivative shift must be finite");
    check_vector_sizes(solution, solution_derivative);

    // Differentiating the BDF residual gives
    //     J = rho*c*derivative_shift*I + K,
    // where K is the positive-semidefinite diffusion matrix. Constant heat production and the
    // prescribed Neumann flux only affect the residual, not the matrix. With constant material
    // coefficients the Picard and Newton matrices are therefore identical.
    const double vertical_face_coefficient =
        thermal_conductivity_ * grid_.dz * grid_.inv_dx * grid_.inv_cell_area;
    const double horizontal_face_coefficient =
        thermal_conductivity_ * grid_.dx * grid_.inv_dz * grid_.inv_cell_area;

    std::vector<double> diagonal(static_cast<std::size_t>(size()),
                                 volumetric_heat_capacity_ * derivative_shift);
    std::vector<MatrixTriplet> triplets;
    const auto nx = static_cast<std::size_t>(grid_.nx);
    const auto nz = static_cast<std::size_t>(grid_.nz);
    const auto internal_face_count = ((nx - 1) * nz) + (nx * (nz - 1));
    triplets.reserve(static_cast<std::size_t>(size()) + (2 * internal_face_count));

    // As in residual assembly, faces are traversed by regular-grid indices. Accumulating the
    // diagonal separately avoids duplicate diagonal triplets during Eigen assembly.
    for (int iz = 0; iz < grid_.nz; ++iz) {
        for (int ix = 0; ix + 1 < grid_.nx; ++ix) {
            const Vector::Index left = cell_index(ix, iz);
            const Vector::Index right = cell_index(ix + 1, iz);
            diagonal[static_cast<std::size_t>(left)] += vertical_face_coefficient;
            diagonal[static_cast<std::size_t>(right)] += vertical_face_coefficient;
            triplets.emplace_back(left, right, -vertical_face_coefficient);
            triplets.emplace_back(right, left, -vertical_face_coefficient);
        }
    }

    for (int iz = 0; iz + 1 < grid_.nz; ++iz) {
        for (int ix = 0; ix < grid_.nx; ++ix) {
            const Vector::Index top = cell_index(ix, iz);
            const Vector::Index bottom = cell_index(ix, iz + 1);
            diagonal[static_cast<std::size_t>(top)] += horizontal_face_coefficient;
            diagonal[static_cast<std::size_t>(bottom)] += horizontal_face_coefficient;
            triplets.emplace_back(top, bottom, -horizontal_face_coefficient);
            triplets.emplace_back(bottom, top, -horizontal_face_coefficient);
        }
    }

    // The upper Dirichlet face contributes only to the diagonal because its temperature is known.
    // The lower and lateral Neumann faces add no matrix coefficients.
    const double surface_face_coefficient = 2.0 * horizontal_face_coefficient;
    for (int ix = 0; ix < grid_.nx; ++ix) {
        const Vector::Index top = cell_index(ix, 0);
        diagonal[static_cast<std::size_t>(top)] += surface_face_coefficient;
    }

    for (Vector::Index index = 0; index < size(); ++index) {
        triplets.emplace_back(index, index, diagonal[static_cast<std::size_t>(index)]);
    }

    matrix.resize(size(), size());
    matrix.set_from_triplets(triplets);
}

Vector::Index ModelHeatSystem::cell_index(int ix, int iz) const noexcept {
    return (static_cast<Vector::Index>(iz) * static_cast<Vector::Index>(grid_.nx)) +
           static_cast<Vector::Index>(ix);
}

void ModelHeatSystem::check_vector_sizes(const Vector& solution,
                                         const Vector& solution_derivative) const {
    if (solution.size() != size() || solution_derivative.size() != size()) {
        throw std::invalid_argument("Heat system vector sizes must match the grid cell count");
    }
}
