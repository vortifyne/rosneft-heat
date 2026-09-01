#pragma once

#include "linear/linear_solver.hpp"
#include "nonlinear/nonlinear_method.hpp"
#include "nonlinear/nonlinear_system.hpp"

#include <memory>
#include <optional>

struct NonlinearSolveRequest {
    NonlinearMethod nonlinear_method;
    double relative_tolerance;
    double absolute_tolerance;
    double step_relative_tolerance;
    int max_iterations;
};

enum class NonlinearSolveStatus {
    converged_residual_absolute,
    converged_residual_relative,
    converged_step,
    max_iterations,
    linear_solve_failed,
    nonfinite_residual,
};

struct [[nodiscard]] NonlinearSolveResult {
    NonlinearSolveStatus status;
    int iterations;
    double final_residual_norm;
    int linear_iterations = 0;
    std::optional<LinearSolveStatus> last_linear_status;

    [[nodiscard]] constexpr bool converged() const noexcept {
        return status == NonlinearSolveStatus::converged_residual_absolute ||
               status == NonlinearSolveStatus::converged_residual_relative ||
               status == NonlinearSolveStatus::converged_step;
    }
};

class NonlinearSolver {
public:
    NonlinearSolver();

    void set_linear_solver(LinearSolverKind kind);

    LinearSolver& linear_solver() noexcept {
        return *linear_solver_;
    }

    const LinearSolver& linear_solver() const noexcept {
        return *linear_solver_;
    }

    LinearSolverKind linear_solver_kind() const noexcept {
        return linear_solver_kind_;
    }

    LinearSolverType linear_solver_type() const noexcept {
        return linear_solver_->type();
    }

    NonlinearSolveResult solve(const NonlinearSystem& nonlinear_system, Vector& x,
                               const NonlinearSolveRequest& nonlinear_request,
                               const LinearSolveRequest& linear_request);

private:
    static std::unique_ptr<LinearSolver> make_linear_solver(LinearSolverKind kind);

    LinearSolverKind linear_solver_kind_ = LinearSolverKind::umfpack_lu;
    std::unique_ptr<LinearSolver> linear_solver_;
};
