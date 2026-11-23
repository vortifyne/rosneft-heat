#pragma once

#include "numerics/algebra/vector.hpp"

class BasinHeatParameterSpace {
public:
    explicit BasinHeatParameterSpace(Vector::Index size) : size_(size) {}

    [[nodiscard]] Vector to_normalized(const Vector& parameters) const;
    [[nodiscard]] Vector to_physical(const Vector& normalized) const;
    [[nodiscard]] Vector normalized_lower_bounds() const;
    [[nodiscard]] Vector normalized_upper_bounds() const;

private:
    Vector::Index size_;
};
