#pragma once

#include "numerics/algebra/vector.hpp"

#include <vector>

class BasinHeatParameterSpace {
public:
    BasinHeatParameterSpace(std::vector<double> ages_ma, Vector flux);
    [[nodiscard]] const std::vector<double>& ages_ma() const noexcept {
        return ages_ma_;
    }
    [[nodiscard]] const Vector& values() const noexcept {
        return flux_;
    }
    void set_values(Vector flux);
    [[nodiscard]] double at(double age_ma) const;

    [[nodiscard]] Vector to_normalized(const Vector& parameters) const;
    [[nodiscard]] Vector to_physical(const Vector& normalized) const;
    [[nodiscard]] Vector normalized_lower_bounds() const;
    [[nodiscard]] Vector normalized_upper_bounds() const;

private:
    std::vector<double> ages_ma_; // Oldest to youngest.
    Vector flux_;                 // W/m².
};
