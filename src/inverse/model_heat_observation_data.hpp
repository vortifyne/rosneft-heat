#pragma once

#include "linear/vector.hpp"

class ModelHeatObservationData {
public:
    ModelHeatObservationData(Vector observed_temperatures, Vector temperature_error_bounds);

    Vector::Index size() const noexcept;

    const Vector& observed_temperatures() const noexcept;
    const Vector& temperature_error_bounds() const noexcept;

    Vector normalized_residuals(const Vector& calculated_temperatures) const;

private:
    Vector observed_temperatures_;
    Vector temperature_error_bounds_;
};
