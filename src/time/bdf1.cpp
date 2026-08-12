#include "time/bdf1.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>

namespace {

class BDF1NonlinearSystem final : public NonlinearSystem {
public:
    BDF1NonlinearSystem(const SemiDiscreteSystem& semi_discrete_system,
                        const TimeSnapshot& previous_snapshot, double step_end_time, double dt)
        : semi_discrete_system_(semi_discrete_system), previous_snapshot_(previous_snapshot),
          time_(step_end_time), inv_dt_(1.0 / dt) {}

    void assemble_residual(const Vector& solution, Vector& residual) const override {
        const Vector solution_derivative = (solution - previous_snapshot_.get().solution) * inv_dt_;

        semi_discrete_system_.get().assemble_residual(time_, solution, solution_derivative,
                                                      residual);
    }

private:
    void assemble_picard_matrix(const Vector& solution, SparseMatrix& matrix) const override {
        assemble_bdf1_matrix(NonlinearMethod::picard, solution, matrix);
    }

    void assemble_newton_matrix(const Vector& solution, SparseMatrix& matrix) const override {
        assemble_bdf1_matrix(NonlinearMethod::newton, solution, matrix);
    }

    void assemble_bdf1_matrix(NonlinearMethod method, const Vector& solution,
                              SparseMatrix& matrix) const {
        const Vector solution_derivative = (solution - previous_snapshot_.get().solution) * inv_dt_;

        semi_discrete_system_.get().assemble_matrix(method, time_, solution, solution_derivative,
                                                    inv_dt_, matrix);
    }

    std::reference_wrapper<const SemiDiscreteSystem> semi_discrete_system_;
    std::reference_wrapper<const TimeSnapshot> previous_snapshot_;
    double time_;
    double inv_dt_;
};

double validate_step_end_time(double previous_time, double step_end_time) {
    if (!std::isfinite(previous_time)) {
        throw std::logic_error("BDF1 previous time must be finite");
    }

    if (!std::isfinite(step_end_time) || !(step_end_time > previous_time)) {
        throw std::invalid_argument(
            "BDF1 step end time must be finite and greater than the previous time");
    }

    return step_end_time - previous_time;
}

} // namespace

std::size_t BDF1::required_snapshot_count() const {
    return 1;
}

std::unique_ptr<NonlinearSystem>
BDF1::make_nonlinear_system(const SemiDiscreteSystem& semi_discrete_system,
                            const TimeHistory& time_history, double step_end_time) const {
    if (time_history.size() < required_snapshot_count()) {
        throw std::logic_error("BDF1 requires one previous time snapshot");
    }

    const auto& current = time_history.current();
    const double dt = validate_step_end_time(current.time, step_end_time);

    return std::make_unique<BDF1NonlinearSystem>(semi_discrete_system, current, step_end_time, dt);
}
