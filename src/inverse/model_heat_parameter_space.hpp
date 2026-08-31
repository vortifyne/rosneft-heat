#pragma once

#include "linear/vector.hpp"
#include "model/model_heat_parameters.hpp"

struct ModelHeatParameterBounds {
    ModelHeatParameters lower;
    ModelHeatParameters upper;
};

class ModelHeatParameterSpace {
public:
    static constexpr Vector::Index kParameterCount = 2;

    explicit ModelHeatParameterSpace(ModelHeatParameterBounds bounds);

    Vector to_normalized(const ModelHeatParameters& parameters) const;
    ModelHeatParameters to_physical(const Vector& normalized) const;

    Vector normalized_lower_bounds() const;
    Vector normalized_upper_bounds() const;

    const ModelHeatParameterBounds& physical_bounds() const noexcept;

private:
    ModelHeatParameterBounds bounds_;
};
