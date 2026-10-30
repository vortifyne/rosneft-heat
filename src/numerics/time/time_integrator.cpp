#include "numerics/time/time_integrator.hpp"

#include "numerics/time/bdf1.hpp"
#include "numerics/time/semi_discrete_system.hpp"
#include "numerics/time/time_scheme.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

TimeIntegrator::TimeIntegrator() : scheme_(std::make_unique<BDF1>()) {
    history_.set_capacity(scheme_->required_snapshot_count());
}

TimeIntegrator::~TimeIntegrator() = default;
TimeIntegrator::TimeIntegrator(TimeIntegrator&&) noexcept = default;
TimeIntegrator& TimeIntegrator::operator=(TimeIntegrator&&) noexcept = default;

void TimeIntegrator::set_initial_solution(double t0, const Vector& u0) {
    if (!std::isfinite(t0)) {
        throw std::invalid_argument("Initial time must be finite");
    }

    history_.clear();
    history_.accept({.t = t0, .u = u0});
}

void TimeIntegrator::set_timestep(double dt) {
    if (!(dt > 0.0) || !std::isfinite(dt)) {
        throw std::invalid_argument("Time step must be finite and positive");
    }

    dt_ = dt;
}

void TimeIntegrator::set_linear_solver(LinearSolverOptions options) {
    nonlinear_solver_.set_linear_solver(options);
    nonlinear_solver_.linear_solver().set_preconditioner_reuse(pc_reuse_ != PcReuse::none);
}

void TimeIntegrator::set_preconditioner_reuse(PcReuse reuse) noexcept {
    if (reuse != pc_reuse_)
        reset_linear_solver();
    pc_reuse_ = reuse;
    nonlinear_solver_.linear_solver().set_preconditioner_reuse(reuse != PcReuse::none);
}

void TimeIntegrator::reset_linear_solver() noexcept {
    nonlinear_solver_.linear_solver().reset();
}

NonlinearSolver& TimeIntegrator::nonlinear_solver() noexcept {
    return nonlinear_solver_;
}

const NonlinearSolver& TimeIntegrator::nonlinear_solver() const noexcept {
    return nonlinear_solver_;
}

const TimeSnapshot& TimeIntegrator::current_snapshot() const {
    if (history_.empty()) {
        throw std::logic_error("Time integrator has no initial solution");
    }

    return history_.current();
}

TimeIntegrationResult TimeIntegrator::advance_to(const SemiDiscreteSystem& system, double t_end,
                                                 const NonlinearSolveRequest& nonlinear,
                                                 const LinearSolveRequest& linear) {
    if (history_.empty()) {
        throw std::logic_error("Time integrator has no initial solution");
    }
    if (!(dt_ > 0.0) || !std::isfinite(dt_)) {
        throw std::logic_error("Time integrator has no valid time step");
    }
    if (!std::isfinite(t_end) || t_end < history_.current().t) {
        throw std::invalid_argument("Final time must be finite and not less than the current time");
    }

    int accepted_steps = 0;
    int rejected_steps = 0;
    int nonlinear_iterations = 0;
    int linear_iterations = 0;
    double last_r_norm = 0.0;
    std::optional<NonlinearSolveStatus> last_nonlinear_status;
    std::optional<LinearSolveStatus> last_linear_status;
    SolverTimings timings;

    while (history_.current().t < t_end) {
        const double t = history_.current().t;
        const double remaining = t_end - t;
        const double t_tol =
            8.0 * std::numeric_limits<double>::epsilon() * std::max(dt_, remaining);
        const bool last_step = remaining <= dt_ + t_tol;
        const double t_next = last_step ? t_end : t + std::min(dt_, remaining);

        if (!(t_next > t)) {
            return {
                .status = TimeIntegrationStatus::time_step_too_small,
                .accepted_steps = accepted_steps,
                .rejected_steps = rejected_steps,
                .nonlinear_iterations = nonlinear_iterations,
                .linear_iterations = linear_iterations,
                .last_r_norm = last_r_norm,
                .last_nonlinear_status = last_nonlinear_status,
                .last_linear_status = last_linear_status,
                .timings = timings,
            };
        }

        if (pc_reuse_ == PcReuse::step)
            reset_linear_solver();
        Vector u = history_.current().u;
        auto nonlinear_system = scheme_->make_nonlinear_system(system, history_, t_next);
        const NonlinearSolveResult nonlinear_result =
            nonlinear_solver_.solve(*nonlinear_system, u, nonlinear, linear);
        nonlinear_iterations += nonlinear_result.iterations;
        linear_iterations += nonlinear_result.linear_iterations;
        timings += nonlinear_result.timings;
        last_r_norm = nonlinear_result.r_norm;
        last_nonlinear_status = nonlinear_result.status;
        if (nonlinear_result.last_linear_status.has_value()) {
            last_linear_status = nonlinear_result.last_linear_status;
        }
        if (!nonlinear_result.converged()) {
            ++rejected_steps;
            return {
                .status = TimeIntegrationStatus::nonlinear_solve_failed,
                .accepted_steps = accepted_steps,
                .rejected_steps = rejected_steps,
                .nonlinear_iterations = nonlinear_iterations,
                .linear_iterations = linear_iterations,
                .last_r_norm = last_r_norm,
                .last_nonlinear_status = last_nonlinear_status,
                .last_linear_status = last_linear_status,
                .timings = timings,
            };
        }

        nonlinear_system.reset();
        history_.accept({.t = t_next, .u = std::move(u)});
        ++accepted_steps;
    }

    return {
        .status = TimeIntegrationStatus::completed,
        .accepted_steps = accepted_steps,
        .rejected_steps = rejected_steps,
        .nonlinear_iterations = nonlinear_iterations,
        .linear_iterations = linear_iterations,
        .last_r_norm = last_r_norm,
        .last_nonlinear_status = last_nonlinear_status,
        .last_linear_status = last_linear_status,
        .timings = timings,
    };
}
