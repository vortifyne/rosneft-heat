#pragma once

#include <array>
#include <cstddef>

enum class BasinObservable {
    temperature,
    reflectance
};

struct BasinWellSample {
    BasinObservable kind;
    // Temperature indices address centers, then horizontal nodes; Ro uses maturity nodes.
    std::array<std::size_t, 4> indices;
    std::array<double, 4> weights;
    double constant = 0.0;
};
