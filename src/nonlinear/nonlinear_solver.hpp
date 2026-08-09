#pragma once

#include "linear/linear_solver.hpp"
#include "nonlinear/nonlinear_method.hpp"
#include "nonlinear/nonlinear_system.hpp"

#include <memory>

struct NonlinearSolveRequest {
    NonlinearMethod nonlinear_method;
    double relative_tolerance;
    double absolute_tolerance;
    int max_iterations;
};

enum class NonlinearSolveStatus {
    converged,
    max_iterations,
    linear_solve_failed,
    nonfinite_residual,
};

struct NonlinearSolveResult {
    NonlinearSolveStatus status;
    int iterations;
    double final_residual_norm;
};

class NonlinearSolver {
public:
    explicit NonlinearSolver() {
        // TODO реализовать линейный решатель (LIN-001)
        // linear_solver_ = std::make_unique<UmfpackLU>();
    }

    NonlinearSolveResult solve(const NonlinearSystem& nonlinear_system, Vector& x,
                               const NonlinearSolveRequest& request) {
        /*
        r = F(x₀)
        r0_norm = ||r||

        пока критерий не выполнен:
        A = линеаризация в x

        решить A · delta_x = -r
        x += delta_x

        r = F(x)
        */
        return NonlinearSolveResult(NonlinearSolveStatus::converged, 1, 0.0);
    }

private:
    std::unique_ptr<LinearSolver> linear_solver_;
};
