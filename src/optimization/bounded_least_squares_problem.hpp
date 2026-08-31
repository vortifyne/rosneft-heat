#pragma once

#include "linear/vector.hpp"

#include <functional>

using LeastSquaresResidualFunction = std::function<Vector(const Vector&)>;

struct BoundedLeastSquaresProblem {
    Vector lower_bounds;
    Vector upper_bounds;
    Vector::Index residual_count;
    LeastSquaresResidualFunction residual_function;
};
