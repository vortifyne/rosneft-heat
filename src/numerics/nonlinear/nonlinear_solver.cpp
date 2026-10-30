#include "numerics/nonlinear/nonlinear_solver.hpp"

#include "numerics/linear/petsc_krylov_linear_solver.hpp"
#include "numerics/linear/umfpack_linear_solver.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

NonlinearSolver::NonlinearSolver() : linear_solver_(make_linear_solver(linear_options_)) {}

void NonlinearSolver::set_linear_solver(LinearSolverOptions options) {
    if (options == linear_options_)
        return;
    auto linear_solver = make_linear_solver(options);
    linear_solver_ = std::move(linear_solver);
    linear_options_ = options;
}

NonlinearSolveResult NonlinearSolver::solve(const NonlinearSystem& system, Vector& x,
                                            const NonlinearSolveRequest& request,
                                            const LinearSolveRequest& linear_request) {
    const auto solve_start = Clock::now();
    if (request.maxiter < 0 || !std::isfinite(request.rtol) || request.rtol < 0.0 ||
        request.rtol >= 1.0 || !std::isfinite(request.atol) || request.atol < 0.0) {
        throw std::invalid_argument("Invalid nonlinear solver iteration settings");
    }
    SolverTimings timings;
    int linear_iterations = 0;
    std::optional<LinearSolveStatus> last_linear_status;
    const auto finish = [&](NonlinearSolveStatus status, int iterations, double r_norm) {
        timings.total_s = elapsed_seconds(solve_start);
        return NonlinearSolveResult{.status = status,
                                    .iterations = iterations,
                                    .r_norm = r_norm,
                                    .linear_iterations = linear_iterations,
                                    .last_linear_status = last_linear_status,
                                    .timings = timings};
    };
    Vector r;
    auto start = Clock::now();
    system.assemble_residual(x, r);
    timings.assembly.residual_s += elapsed_seconds(start);

    if (!r.all_finite()) {
        return finish(NonlinearSolveStatus::nonfinite_residual, 0,
                      std::numeric_limits<double>::infinity());
    }

    const double r0_norm = r.norm();
    if (r0_norm <= request.atol) {
        return finish(NonlinearSolveStatus::converged_residual_absolute, 0, r0_norm);
    }

    for (int iteration = 0; iteration < request.maxiter; ++iteration) {
        SparseMatrix A(x.size(), x.size(), linear_solver_->required_storage_order());
        start = Clock::now();
        system.assemble_matrix(request.method, x, A);
        timings.assembly.matrix_s += elapsed_seconds(start);

        const Vector b = -r;
        Vector dx;
        start = Clock::now();
        const LinearSolveResult linear_result = linear_solver_->solve(A, b, dx, linear_request);
        timings.linear_s += elapsed_seconds(start);
        linear_iterations += linear_result.iterations;
        last_linear_status = linear_result.status;

        if (linear_result.status != LinearSolveStatus::converged) {
            return finish(NonlinearSolveStatus::linear_solve_failed, iteration, r.norm());
        }

        Vector x_new = x + dx;
        Vector r_new;
        start = Clock::now();
        system.assemble_residual(x_new, r_new);
        timings.assembly.residual_s += elapsed_seconds(start);

        if (!r_new.all_finite()) {
            return finish(NonlinearSolveStatus::nonfinite_residual, iteration + 1,
                          std::numeric_limits<double>::infinity());
        }

        const double r_norm = r_new.norm();
        const int niter = iteration + 1;

        x = std::move(x_new);
        r = std::move(r_new);

        if (r_norm <= request.atol) {
            return finish(NonlinearSolveStatus::converged_residual_absolute, niter, r_norm);
        }

        if (r_norm <= request.rtol * r0_norm) {
            return finish(NonlinearSolveStatus::converged_residual_relative, niter, r_norm);
        }
    }

    return finish(NonlinearSolveStatus::maxiter, request.maxiter, r.norm());
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
