#include "numerics/nonlinear/nonlinear_solver.hpp"

#include "numerics/linear/petsc_krylov_linear_solver.hpp"
#include "numerics/linear/umfpack_linear_solver.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

NonlinearSolver::NonlinearSolver() : linear_solver_(make_linear_solver(linear_solver_options_)) {}

void NonlinearSolver::set_linear_solver(LinearSolverOptions options) {
    if (options == linear_solver_options_)
        return;
    auto linear_solver = make_linear_solver(options);
    linear_solver_ = std::move(linear_solver);
    linear_solver_options_ = options;
}

NonlinearSolveResult NonlinearSolver::solve(const NonlinearSystem& nonlinear_system, Vector& x,
                                            const NonlinearSolveRequest& nonlinear_request,
                                            const LinearSolveRequest& linear_request) {
    if (nonlinear_request.max_iterations < 0 ||
        !std::isfinite(nonlinear_request.relative_tolerance) ||
        nonlinear_request.relative_tolerance < 0.0 || nonlinear_request.relative_tolerance >= 1.0 ||
        !std::isfinite(nonlinear_request.absolute_tolerance) ||
        nonlinear_request.absolute_tolerance < 0.0) {
        throw std::invalid_argument("Invalid nonlinear solver iteration settings");
    }
    SolverTimings timings;
    const auto elapsed_seconds = [](const auto start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    Vector residual;
    auto operation_start = std::chrono::steady_clock::now();
    nonlinear_system.assemble_residual(x, residual);
    timings.assembly_seconds += elapsed_seconds(operation_start);

    if (!residual.all_finite()) {
        return {
            .status = NonlinearSolveStatus::nonfinite_residual,
            .iterations = 0,
            .final_residual_norm = std::numeric_limits<double>::infinity(),
            .linear_iterations = 0,
            .last_linear_status = std::nullopt,
            .timings = timings,
        };
    }

    const double initial_residual_norm = residual.norm();
    if (initial_residual_norm <= nonlinear_request.absolute_tolerance) {
        return {
            .status = NonlinearSolveStatus::converged_residual_absolute,
            .iterations = 0,
            .final_residual_norm = initial_residual_norm,
            .linear_iterations = 0,
            .last_linear_status = std::nullopt,
            .timings = timings,
        };
    }

    int linear_iterations = 0;
    std::optional<LinearSolveStatus> last_linear_status;
    for (int iteration = 0; iteration < nonlinear_request.max_iterations; ++iteration) {
        SparseMatrix matrix(x.size(), x.size(), linear_solver_->required_storage_order());
        operation_start = std::chrono::steady_clock::now();
        nonlinear_system.assemble_matrix(nonlinear_request.nonlinear_method, x, matrix);
        timings.assembly_seconds += elapsed_seconds(operation_start);

        const Vector b = -residual;
        Vector delta_x;
        operation_start = std::chrono::steady_clock::now();
        const LinearSolveResult linear_result =
            linear_solver_->solve(matrix, b, delta_x, linear_request);
        timings.linear_solve_seconds += elapsed_seconds(operation_start);
        linear_iterations += linear_result.iterations;
        last_linear_status = linear_result.status;

        if (linear_result.status != LinearSolveStatus::converged) {
            return {
                .status = NonlinearSolveStatus::linear_solve_failed,
                .iterations = iteration,
                .final_residual_norm = residual.norm(),
                .linear_iterations = linear_iterations,
                .last_linear_status = last_linear_status,
                .timings = timings,
            };
        }

        Vector x_candidate = x + delta_x;
        Vector candidate_residual;
        operation_start = std::chrono::steady_clock::now();
        nonlinear_system.assemble_residual(x_candidate, candidate_residual);
        timings.assembly_seconds += elapsed_seconds(operation_start);

        if (!candidate_residual.all_finite()) {
            return {
                .status = NonlinearSolveStatus::nonfinite_residual,
                .iterations = iteration + 1,
                .final_residual_norm = std::numeric_limits<double>::infinity(),
                .linear_iterations = linear_iterations,
                .last_linear_status = last_linear_status,
                .timings = timings,
            };
        }

        const double residual_norm = candidate_residual.norm();
        const int completed_iterations = iteration + 1;

        x = std::move(x_candidate);
        residual = std::move(candidate_residual);

        if (residual_norm <= nonlinear_request.absolute_tolerance) {
            return {
                .status = NonlinearSolveStatus::converged_residual_absolute,
                .iterations = completed_iterations,
                .final_residual_norm = residual_norm,
                .linear_iterations = linear_iterations,
                .last_linear_status = last_linear_status,
                .timings = timings,
            };
        }

        if (residual_norm <= nonlinear_request.relative_tolerance * initial_residual_norm) {
            return {
                .status = NonlinearSolveStatus::converged_residual_relative,
                .iterations = completed_iterations,
                .final_residual_norm = residual_norm,
                .linear_iterations = linear_iterations,
                .last_linear_status = last_linear_status,
                .timings = timings,
            };
        }
    }

    return {
        .status = NonlinearSolveStatus::max_iterations,
        .iterations = nonlinear_request.max_iterations,
        .final_residual_norm = residual.norm(),
        .linear_iterations = linear_iterations,
        .last_linear_status = last_linear_status,
        .timings = timings,
    };
}

std::unique_ptr<LinearSolver> NonlinearSolver::make_linear_solver(LinearSolverOptions options) {
    switch (options.method) {
    case LinearMethod::lu:
        return std::make_unique<UmfpackLinearSolver>();
    case LinearMethod::gmres:
    case LinearMethod::bicgstab:
        return std::make_unique<PetscKrylovLinearSolver>(options.method, options.preconditioner);
    }
    throw std::invalid_argument("Unknown linear method");
}
