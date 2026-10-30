#include "numerics/time/bdf1.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>

namespace {

class BDF1NonlinearSystem final : public NonlinearSystem {
public:
    BDF1NonlinearSystem(const SemiDiscreteSystem& system, const TimeSnapshot& previous,
                        double t_next, double dt)
        : system_(system), previous_(previous), t_(t_next), alpha_(1.0 / dt) {}

    void assemble_residual(const Vector& u, Vector& r) const override {
        const Vector du_dt = (u - previous_.get().u) * alpha_;

        system_.get().assemble_residual(t_, u, du_dt, r);
    }

private:
    void assemble_picard_matrix(const Vector& u, SparseMatrix& A) const override {
        assemble_bdf1_matrix(NonlinearMethod::picard, u, A);
    }

    void assemble_newton_matrix(const Vector& u, SparseMatrix& A) const override {
        assemble_bdf1_matrix(NonlinearMethod::newton, u, A);
    }

    void assemble_bdf1_matrix(NonlinearMethod method, const Vector& u, SparseMatrix& A) const {
        const Vector du_dt = (u - previous_.get().u) * alpha_;

        system_.get().assemble_matrix(method, t_, u, du_dt, alpha_, A);
    }

    std::reference_wrapper<const SemiDiscreteSystem> system_;
    std::reference_wrapper<const TimeSnapshot> previous_;
    double t_;
    double alpha_;
};

double validate_step_end_time(double t_prev, double t_next) {
    if (!std::isfinite(t_prev)) {
        throw std::logic_error("BDF1 previous time must be finite");
    }

    if (!std::isfinite(t_next) || !(t_next > t_prev)) {
        throw std::invalid_argument(
            "BDF1 step end time must be finite and greater than the previous time");
    }

    return t_next - t_prev;
}

} // namespace

std::size_t BDF1::required_snapshot_count() const noexcept {
    return 1;
}

std::unique_ptr<NonlinearSystem> BDF1::make_nonlinear_system(const SemiDiscreteSystem& system,
                                                             const TimeHistory& history,
                                                             double t_next) const {
    if (history.size() < required_snapshot_count()) {
        throw std::logic_error("BDF1 requires one previous time snapshot");
    }

    const auto& current = history.current();
    const double dt = validate_step_end_time(current.t, t_next);

    return std::make_unique<BDF1NonlinearSystem>(system, current, t_next, dt);
}
