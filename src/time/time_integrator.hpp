#pragma once

#include "linear/linear_solver.hpp"
#include "nonlinear/nonlinear_solver.hpp"
#include "time/time_history.hpp"

#include <memory>
#include <optional>

class SemiDiscreteSystem;
class TimeScheme;

enum class TimeIntegrationStatus {
    completed,
    nonlinear_solve_failed,
    time_step_too_small,
};

struct TimeIntegrationResult {
    TimeIntegrationStatus status;
    int accepted_steps;
    int rejected_steps;
    int nonlinear_iterations = 0;
    int linear_iterations = 0;
    std::optional<NonlinearSolveStatus> last_nonlinear_status;
    std::optional<LinearSolveStatus> last_linear_status;

    bool completed() const noexcept {
        return status == TimeIntegrationStatus::completed;
    }
};

class TimeIntegrator {
public:
    TimeIntegrator();
    ~TimeIntegrator();

    TimeIntegrator(const TimeIntegrator&) = delete;
    TimeIntegrator& operator=(const TimeIntegrator&) = delete;
    TimeIntegrator(TimeIntegrator&&) noexcept;
    TimeIntegrator& operator=(TimeIntegrator&&) noexcept;

    void set_initial_solution(double initial_time, const Vector& initial_solution);
    void set_timestep(double dt);
    void set_linear_solver(LinearSolverKind kind);

    NonlinearSolver& nonlinear_solver() noexcept;
    const NonlinearSolver& nonlinear_solver() const noexcept;

    const TimeSnapshot& current_snapshot() const;

    TimeIntegrationResult advance_to(const SemiDiscreteSystem& semi_discrete_system,
                                     double final_time,
                                     const NonlinearSolveRequest& nonlinear_request,
                                     const LinearSolveRequest& linear_request);

private:
    double dt_ = 0.0;
    TimeHistory time_history_;
    std::unique_ptr<TimeScheme> time_scheme_;
    NonlinearSolver nonlinear_solver_;
};
