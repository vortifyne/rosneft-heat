#pragma once

#include "basin/nonmatching_column_grid.hpp"

#include <span>

struct BasinForwardSnapshot {
    enum class Kind {
        initial,
        before_event,
        after_event
    };
    Kind kind = Kind::initial;
    double age_ma = 0.0;
    const NonmatchingColumnGrid& grid;
    std::span<const double> temperature;
    std::span<const double> maturity_integrals;
    std::span<const double> reflectance_percent;
};
