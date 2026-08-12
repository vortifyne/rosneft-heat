#include "nonlinear/nonlinear_solver.hpp"

#include "linear/umfpack_linear_solver.hpp"

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
    Vector residual;
    nonlinear_system.assemble_residual(x, residual);

    if (!residual.all_finite()) {
        return {
            .status = NonlinearSolveStatus::nonfinite_residual,
            .iterations = 0,
            .final_residual_norm = std::numeric_limits<double>::infinity(),
            .linear_iterations = 0,
            .last_linear_status = std::nullopt,
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
        };
    }

    int linear_iterations = 0;
    std::optional<LinearSolveStatus> last_linear_status;
    for (int iteration = 0; iteration < nonlinear_request.max_iterations; ++iteration) {
        SparseMatrix matrix(x.size(), x.size(), linear_solver_->required_storage_order());
        nonlinear_system.assemble_matrix(nonlinear_request.nonlinear_method, x, matrix);

        const Vector b = -residual;
        Vector delta_x;
        LinearSolveRequest current_linear_request = linear_request;
        const LinearSolveResult linear_result =
            linear_solver_->solve(matrix, b, delta_x, current_linear_request);
        linear_iterations += linear_result.iterations;
        last_linear_status = linear_result.status;

        if (linear_result.status != LinearSolveStatus::converged) {
            return {
                .status = NonlinearSolveStatus::linear_solve_failed,
                .iterations = iteration,
                .final_residual_norm = residual.norm(),
                .linear_iterations = linear_iterations,
                .last_linear_status = last_linear_status,
            };
        }

        Vector x_candidate = x + delta_x;
        Vector candidate_residual;
        nonlinear_system.assemble_residual(x_candidate, candidate_residual);

        if (!candidate_residual.all_finite()) {
            return {
                .status = NonlinearSolveStatus::nonfinite_residual,
                .iterations = iteration + 1,
                .final_residual_norm = std::numeric_limits<double>::infinity(),
                .linear_iterations = linear_iterations,
                .last_linear_status = last_linear_status,
            };
        }

        const double residual_norm = candidate_residual.norm();
        const double step_norm = delta_x.norm();
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
            };
        }

        if (residual_norm <= nonlinear_request.relative_tolerance * initial_residual_norm) {
            return {
                .status = NonlinearSolveStatus::converged_residual_relative,
                .iterations = completed_iterations,
                .final_residual_norm = residual_norm,
                .linear_iterations = linear_iterations,
                .last_linear_status = last_linear_status,
            };
        }

        if (step_norm < nonlinear_request.step_relative_tolerance * solution_norm) {
            return {
                .status = NonlinearSolveStatus::converged_step,
                .iterations = completed_iterations,
                .final_residual_norm = residual_norm,
                .linear_iterations = linear_iterations,
                .last_linear_status = last_linear_status,
            };
        }
    }

    return {
        .status = NonlinearSolveStatus::max_iterations,
        .iterations = nonlinear_request.max_iterations,
        .final_residual_norm = residual.norm(),
        .linear_iterations = linear_iterations,
        .last_linear_status = last_linear_status,
    };
}

std::unique_ptr<LinearSolver> NonlinearSolver::make_linear_solver(LinearSolverKind kind) {
    switch (kind) {
    case LinearSolverKind::umfpack_lu:
        return std::make_unique<UmfpackLinearSolver>();
    }

    throw std::invalid_argument("Unknown linear solver kind");
}
