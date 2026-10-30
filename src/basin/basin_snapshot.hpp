#pragma once

#include <cstddef>
#include <span>

struct ColGrid;

struct BasinSnapshot {
    enum class Kind {
        initial,
        before_event,
        after_event,
        timestep
    };
    Kind kind = Kind::initial;
    std::size_t epoch = 0; // Zero-based.
    int step = 0;
    double age_ma = 0.0; // Geological age [Ma].
    const ColGrid& grid;
    std::span<const double> T;  // Temperature [K].
    std::span<const double> I;  // Reaction integrals [-], cell-major.
    std::span<const double> Ro; // Vitrinite reflectance [%].
};
