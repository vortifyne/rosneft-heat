#pragma once

#include "numerics/algebra/vector.hpp"

struct TimeSnapshot {
    double time;
    Vector solution;
};
