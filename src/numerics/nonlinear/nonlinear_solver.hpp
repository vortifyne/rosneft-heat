#pragma once

#include "numerics/linear/linear_solver.hpp"
#include "numerics/nonlinear/nonlinear_method.hpp"
#include "numerics/nonlinear/nonlinear_system.hpp"

#include <memory>
#include <optional>

struct NonlinearSolveRequest {
    NonlinearMethod nonlinear_method = NonlinearMethod::picard;
    double relative_tolerance = 1.0e-8;
    double absolute_tolerance = 1.0e-8;
    int max_iterations = 200;
};

struct SolverTimings {
    double assembly_seconds = 0.0;
    double linear_solve_seconds = 0.0;

    SolverTimings& operator+=(const SolverTimings& other) noexcept {
        assembly_seconds += other.assembly_seconds;
        linear_solve_seconds += other.linear_solve_seconds;
        return *this;
    }
};

enum class NonlinearSolveStatus {
    converged_residual_absolute,
    converged_residual_relative,
    max_iterations,
    linear_solve_failed,
    nonfinite_residual,
};

struct [[nodiscard]] NonlinearSolveResult {
    NonlinearSolveStatus status;
    int iterations;
    double final_residual_norm;
    double final_step_norm = 0.0;
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

    const LinearSolverOptions& linear_solver_options() const noexcept {
        return linear_solver_options_;
    }

    LinearSolverType linear_solver_type() const noexcept {
        return linear_solver_->type();
    }

    NonlinearSolveResult solve(const NonlinearSystem& nonlinear_system, Vector& x,
                               const NonlinearSolveRequest& nonlinear_request,
                               const LinearSolveRequest& linear_request);

private:
    static std::unique_ptr<LinearSolver> make_linear_solver(LinearSolverOptions options);

    LinearSolverOptions linear_solver_options_{};
    std::unique_ptr<LinearSolver> linear_solver_;
};
