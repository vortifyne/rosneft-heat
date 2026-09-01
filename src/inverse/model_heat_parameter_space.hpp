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

    [[nodiscard]] Vector to_normalized(const ModelHeatParameters& parameters) const;
    [[nodiscard]] ModelHeatParameters to_physical(const Vector& normalized) const;

    [[nodiscard]] Vector normalized_lower_bounds() const;
    [[nodiscard]] Vector normalized_upper_bounds() const;

    const ModelHeatParameterBounds& physical_bounds() const noexcept;

private:
    ModelHeatParameterBounds bounds_;
};
