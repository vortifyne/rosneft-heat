#pragma once

#include "discretization/semi_discrete_system.hpp"
#include "nonlinear/nonlinear_system.hpp"
#include "time/time_history.hpp"

#include <memory>

class TimeScheme {
public:
    virtual ~TimeScheme() = default;

    virtual std::size_t required_snapshot_count() const noexcept = 0;

    virtual std::unique_ptr<NonlinearSystem>
    make_nonlinear_system(const SemiDiscreteSystem& semi_discrete_system,
                          const TimeHistory& time_history, double step_end_time) const = 0;
};
