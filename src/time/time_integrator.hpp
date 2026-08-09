#pragma once

#include "nonlinear/nonlinear_solver.hpp"
#include "time/bdf1.hpp"
#include "time/time_history.hpp"
#include "time/time_scheme.hpp"

#include <algorithm>

class TimeIntegrator {
public:
    explicit TimeIntegrator() {
        time_scheme_ = std::make_unique<BDF1>();
        time_history_.set_capacity(time_scheme_->required_snapshot_count());
        nonlinear_solver_ = std::make_unique<NonlinearSolver>();
    }

    void set_initial_solution(double t_init, const Vector& solution_init) {
        time_history_.accept({.time = t_init, .solution = solution_init});
    }

    void set_timestep(double dt) {
        dt_ = dt;
    }

    void advance_to(const SemiDiscreteSystem& semi_discrete_system, double t_final,
                    const NonlinearSolveRequest& nonlinear_request) {
        while (time_history_.current().time < t_final) {
            const double t_current = time_history_.current().time;
            const double dt_trial = std::min(dt_, t_final - t_current);

            // TODO у класса Vector должен быть оператор "="
            Vector solution = time_history_.current().solution;

            auto nonlinear_system =
                time_scheme_->make_nonlinear_system(semi_discrete_system, time_history_, dt_trial);
            auto nonlinear_result =
                nonlinear_solver_->solve(*nonlinear_system, solution, nonlinear_request);

            // TODO логика адаптивного шага: точность решения + затраты на вычисления
            const bool step_accepted = (nonlinear_result.status == NonlinearSolveStatus::converged);
            if (step_accepted) {
                time_history_.accept(
                    {.time = t_current + dt_trial, .solution = std::move(solution)});
                // TODO вычисление обновлённого шага по времени
            } else {
                // TODO уменьшение шага по времени из-за недостаточной точности или плохой
                // сходимости
                dt_ *= 0.5;
            }
        }
    }

private:
    double dt_ = 0.0;
    TimeHistory time_history_;
    std::unique_ptr<TimeScheme> time_scheme_;
    std::unique_ptr<NonlinearSolver> nonlinear_solver_;
};
