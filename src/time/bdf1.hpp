#pragma once

#include "time/time_scheme.hpp"

class BDF1 final : public TimeScheme {
public:
    std::size_t required_snapshot_count() const override;

    std::unique_ptr<NonlinearSystem>
    make_nonlinear_system(const SemiDiscreteSystem& semi_discrete_system,
                          const TimeHistory& time_history, double step_end_time) const override;
};
