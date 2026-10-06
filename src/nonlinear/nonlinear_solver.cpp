#include "nonlinear/nonlinear_solver.hpp"

#include "linear/umfpack_linear_solver.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

NonlinearSolver::NonlinearSolver() : linear_solver_(make_linear_solver(linear_solver_kind_)) {}

void NonlinearSolver::set_linear_solver(LinearSolverKind kind) {
    auto linear_solver = make_linear_solver(kind);
    linear_solver_ = std::move(linear_solver);
    linear_solver_kind_ = kind;
}

NonlinearSolveResult NonlinearSolver::solve(const NonlinearSystem& nonlinear_system, Vector& x,
                                            const NonlinearSolveRequest& nonlinear_request,
                                            const LinearSolveRequest& linear_request) {
    if (nonlinear_request.max_iterations < 0 || nonlinear_request.max_backtracking_steps < 0 ||
        !(nonlinear_request.backtracking_reduction > 0.0) ||
        !(nonlinear_request.backtracking_reduction < 1.0)) {
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
        LinearSolveRequest current_linear_request = linear_request;
        operation_start = std::chrono::steady_clock::now();
        const LinearSolveResult linear_result =
            linear_solver_->solve(matrix, b, delta_x, current_linear_request);
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

        double step_scale = 1.0;
        Vector scaled_delta = delta_x;
        Vector x_candidate = x + scaled_delta;
        Vector candidate_residual;
        operation_start = std::chrono::steady_clock::now();
        nonlinear_system.assemble_residual(x_candidate, candidate_residual);
        timings.assembly_seconds += elapsed_seconds(operation_start);

        if (nonlinear_request.use_backtracking && candidate_residual.all_finite()) {
            const double current_norm = residual.norm();
            int backtracking_step = 0;
            while (candidate_residual.norm() >= current_norm &&
                   backtracking_step < nonlinear_request.max_backtracking_steps) {
                step_scale *= nonlinear_request.backtracking_reduction;
                scaled_delta = delta_x * step_scale;
                x_candidate = x + scaled_delta;
                operation_start = std::chrono::steady_clock::now();
                nonlinear_system.assemble_residual(x_candidate, candidate_residual);
                timings.assembly_seconds += elapsed_seconds(operation_start);
                ++backtracking_step;
                if (!candidate_residual.all_finite()) {
                    continue;
                }
            }
        }

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
        const double step_norm = scaled_delta.norm();
        const double solution_norm = x_candidate.norm();
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

        if (step_norm < nonlinear_request.step_relative_tolerance * solution_norm) {
            return {
                .status = NonlinearSolveStatus::converged_step,
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

std::unique_ptr<LinearSolver> NonlinearSolver::make_linear_solver(LinearSolverKind kind) {
    switch (kind) {
    case LinearSolverKind::umfpack_lu:
        return std::make_unique<UmfpackLinearSolver>();
    }

    throw std::invalid_argument("Unknown linear solver kind");
}
