#pragma once

#include "numerics/linear/linear_solver.hpp"
#include "numerics/nonlinear/nonlinear_solver.hpp"
#include "numerics/time/pc_reuse.hpp"
#include "numerics/time/time_history.hpp"

#include <memory>
#include <optional>

class SemiDiscreteSystem;
class TimeScheme;

enum class TimeIntegrationStatus {
    completed,
    nonlinear_solve_failed,
    time_step_too_small,
};

struct [[nodiscard]] TimeIntegrationResult {
    TimeIntegrationStatus status;
    int accepted_steps;
    int rejected_steps;
    int nonlinear_iterations = 0;
    int linear_iterations = 0;
    double last_r_norm = 0.0;
    std::optional<NonlinearSolveStatus> last_nonlinear_status;
    std::optional<LinearSolveStatus> last_linear_status;
    SolverTimings timings;

    [[nodiscard]] constexpr bool completed() const noexcept {
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

    void set_initial_solution(double t0, const Vector& u0);
    void set_timestep(double dt);
    void set_linear_solver(LinearSolverOptions options);
    void set_preconditioner_reuse(PcReuse reuse) noexcept;
    void reset_linear_solver() noexcept;

    NonlinearSolver& nonlinear_solver() noexcept;
    const NonlinearSolver& nonlinear_solver() const noexcept;

    const TimeSnapshot& current_snapshot() const;

    TimeIntegrationResult advance_to(const SemiDiscreteSystem& system, double t_end,
                                     const NonlinearSolveRequest& nonlinear,
                                     const LinearSolveRequest& linear);

private:
    double dt_ = 0.0;
    PcReuse pc_reuse_ = PcReuse::none;
    TimeHistory history_;
    std::unique_ptr<TimeScheme> scheme_;
    NonlinearSolver nonlinear_solver_;
};
