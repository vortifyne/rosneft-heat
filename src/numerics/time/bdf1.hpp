#pragma once

#include "numerics/time/time_scheme.hpp"

class BDF1 final : public TimeScheme {
public:
    std::size_t required_snapshot_count() const noexcept override;

    std::unique_ptr<NonlinearSystem> make_nonlinear_system(const SemiDiscreteSystem& system,
                                                           const TimeHistory& history,
                                                           double t_next) const override;
};
