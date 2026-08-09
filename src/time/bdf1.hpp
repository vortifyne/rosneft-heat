#pragma once

#include "time/time_scheme.hpp"

class BDF1NonlinearSystem final : public NonlinearSystem {
public:
    BDF1NonlinearSystem(const SemiDiscreteSystem& semi_discrete_system,
                        const TimeSnapshot& previous_snapshot, double time, double dt)
        : semi_discrete_system_(semi_discrete_system), previous_snapshot_(previous_snapshot),
          time_(time), inv_dt_(1.0 / dt) {}

    void assemble_residual(const Vector& solution, Vector& residual) const override {

        Vector solution_derivative;
        // solution_derivative =
        //     (solution - previous_snapshot_.solution) * inv_dt_;

        semi_discrete_system_.assemble_residual(time_, solution, solution_derivative, residual);
    }

private:
    void assemble_picard_matrix(const Vector& solution, SparseMatrix& matrix) const override {

        Vector solution_derivative;
        // TODO Собрать производную, как выше

        semi_discrete_system_.assemble_matrix(NonlinearMethod::picard, time_, solution,
                                              solution_derivative, inv_dt_, matrix);
    }

    void assemble_newton_matrix(const Vector& solution, SparseMatrix& matrix) const override {

        Vector solution_derivative;
        // TODO Собрать производную, как выше

        semi_discrete_system_.assemble_matrix(NonlinearMethod::newton, time_, solution,
                                              solution_derivative, inv_dt_, matrix);
    }

    const SemiDiscreteSystem& semi_discrete_system_;
    const TimeSnapshot& previous_snapshot_;
    double time_;
    double inv_dt_;
};

class BDF1 final : public TimeScheme {
public:
    std::size_t required_snapshot_count() const override {
        return 1;
    }

    std::unique_ptr<NonlinearSystem>
    make_nonlinear_system(const SemiDiscreteSystem& semi_discrete_system,
                          const TimeHistory& time_history, double dt) const override {
        const auto& current = time_history.current();

        return std::make_unique<BDF1NonlinearSystem>(semi_discrete_system, current,
                                                     current.time + dt, dt);
    }
};
