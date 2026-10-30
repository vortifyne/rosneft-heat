#pragma once

#include "numerics/nonlinear/nonlinear_system.hpp"
#include "numerics/time/semi_discrete_system.hpp"
#include "numerics/time/time_history.hpp"

#include <memory>

class TimeScheme {
public:
    virtual ~TimeScheme() = default;

    virtual std::size_t required_snapshot_count() const noexcept = 0;

    virtual std::unique_ptr<NonlinearSystem> make_nonlinear_system(const SemiDiscreteSystem& system,
                                                                   const TimeHistory& history,
                                                                   double t_next) const = 0;
};
