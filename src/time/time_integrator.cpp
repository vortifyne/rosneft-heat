#include "time/time_integrator.hpp"

#include "discretization/semi_discrete_system.hpp"
#include "time/bdf1.hpp"
#include "time/time_scheme.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

TimeIntegrator::TimeIntegrator()
    : time_scheme_(std::make_unique<BDF1>()),
      nonlinear_solver_(std::make_unique<NonlinearSolver>()) {
    time_history_.set_capacity(time_scheme_->required_snapshot_count());
}

TimeIntegrator::~TimeIntegrator() = default;
TimeIntegrator::TimeIntegrator(TimeIntegrator&&) noexcept = default;
TimeIntegrator& TimeIntegrator::operator=(TimeIntegrator&&) noexcept = default;

void TimeIntegrator::set_initial_solution(double initial_time, const Vector& initial_solution) {
    if (!std::isfinite(initial_time)) {
        throw std::invalid_argument("Initial time must be finite");
    }

    time_history_.clear();
    time_history_.accept({.time = initial_time, .solution = initial_solution});
}

void TimeIntegrator::set_timestep(double dt) {
    if (!(dt > 0.0) || !std::isfinite(dt)) {
        throw std::invalid_argument("Time step must be finite and positive");
    }

    dt_ = dt;
}

void TimeIntegrator::set_linear_solver(LinearSolverKind kind) {
    nonlinear_solver_->set_linear_solver(kind);
}

const TimeSnapshot& TimeIntegrator::current_snapshot() const {
    if (time_history_.empty()) {
        throw std::logic_error("Time integrator has no initial solution");
    }

    return time_history_.current();
}

TimeIntegrationResult TimeIntegrator::advance_to(const SemiDiscreteSystem& semi_discrete_system,
                                                 double final_time,
                                                 const NonlinearSolveRequest& nonlinear_request,
                                                 const LinearSolveRequest& linear_request) {
    if (time_history_.empty()) {
        throw std::logic_error("Time integrator has no initial solution");
    }
    if (!(dt_ > 0.0) || !std::isfinite(dt_)) {
        throw std::logic_error("Time integrator has no valid time step");
    }
    if (!std::isfinite(final_time) || final_time < time_history_.current().time) {
        throw std::invalid_argument("Final time must be finite and not less than the current time");
    }

    int accepted_steps = 0;
    int rejected_steps = 0;
    int nonlinear_iterations = 0;
    int linear_iterations = 0;
    std::optional<NonlinearSolveStatus> last_nonlinear_status;
    std::optional<LinearSolveStatus> last_linear_status;

    while (time_history_.current().time < final_time) {
        const double current_time = time_history_.current().time;
        const double remaining_time = final_time - current_time;
        const bool reaches_final_time = remaining_time <= dt_;
        const double step_end_time =
            reaches_final_time ? final_time : current_time + std::min(dt_, remaining_time);

        if (!(step_end_time > current_time)) {
            return {
                .status = TimeIntegrationStatus::time_step_too_small,
                .accepted_steps = accepted_steps,
                .rejected_steps = rejected_steps,
                .nonlinear_iterations = nonlinear_iterations,
                .linear_iterations = linear_iterations,
                .last_nonlinear_status = last_nonlinear_status,
                .last_linear_status = last_linear_status,
            };
        }

        Vector solution = time_history_.current().solution;
        auto nonlinear_system =
            time_scheme_->make_nonlinear_system(semi_discrete_system, time_history_, step_end_time);
        const NonlinearSolveResult nonlinear_result = nonlinear_solver_->solve(
            *nonlinear_system, solution, nonlinear_request, linear_request);
        nonlinear_iterations += nonlinear_result.iterations;
        linear_iterations += nonlinear_result.linear_iterations;
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
                .last_nonlinear_status = last_nonlinear_status,
                .last_linear_status = last_linear_status,
            };
        }

        nonlinear_system.reset();
        time_history_.accept({.time = step_end_time, .solution = std::move(solution)});
        ++accepted_steps;
    }

    return {
        .status = TimeIntegrationStatus::completed,
        .accepted_steps = accepted_steps,
        .rejected_steps = rejected_steps,
        .nonlinear_iterations = nonlinear_iterations,
        .linear_iterations = linear_iterations,
        .last_nonlinear_status = last_nonlinear_status,
        .last_linear_status = last_linear_status,
    };
}
