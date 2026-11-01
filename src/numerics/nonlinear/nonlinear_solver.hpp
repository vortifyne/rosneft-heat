#pragma once

#include "numerics/linear/linear_solver.hpp"
#include "numerics/nonlinear/nonlinear_method.hpp"
#include "numerics/nonlinear/nonlinear_system.hpp"
#include "numerics/timing.hpp"

#include <memory>
#include <optional>

struct NonlinearSolveRequest {
    NonlinearMethod method = NonlinearMethod::picard;
    double rtol = 1.0e-8;
    double atol = 1.0e-8;
    int maxiter = 200;
};

enum class NonlinearSolveStatus {
    converged_residual_absolute,
    converged_residual_relative,
    maxiter,
    linear_solve_failed,
    nonfinite_residual,
};

struct [[nodiscard]] NonlinearSolveResult {
    NonlinearSolveStatus status;
    int iterations;
    double r_norm;
    int linear_iterations = 0;
    std::optional<LinearSolveStatus> last_linear_status;
    SolverTimings timings;

    [[nodiscard]] constexpr bool converged() const noexcept {
        return status == NonlinearSolveStatus::converged_residual_absolute ||
               status == NonlinearSolveStatus::converged_residual_relative;
    }
};

class NonlinearSolver {
public:
    NonlinearSolver();

    void set_linear_solver(LinearSolverOptions options);

    LinearSolver& linear_solver() noexcept {
        return *linear_solver_;
    }

    const LinearSolver& linear_solver() const noexcept {
        return *linear_solver_;
    }

    const LinearSolverOptions& linear_options() const noexcept {
        return linear_options_;
    }

    LinearSolverType linear_solver_type() const noexcept {
        return linear_solver_->type();
    }

    NonlinearSolveResult solve(const NonlinearSystem& system, Vector& x,
                               const NonlinearSolveRequest& request,
                               const LinearSolveRequest& linear_request);

private:
    static std::unique_ptr<LinearSolver> make_linear_solver(LinearSolverOptions options);

    LinearSolverOptions linear_options_{};
    std::unique_ptr<LinearSolver> linear_solver_;
};
