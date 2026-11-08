#pragma once

#include "numerics/algebra/vector.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

enum class BasinObservable {
    temperature,
    reflectance
};

struct BasinObservation {
    BasinObservable kind;
    std::string well;
    double x;
    double z;
    double value; // Temperature [K] or Ro [%].
    double error; // Absolute error bound, in the same units.
};

class BasinHeatObservationData {
public:
    explicit BasinHeatObservationData(std::vector<BasinObservation> points);
    static BasinHeatObservationData read(const std::filesystem::path& path);

    [[nodiscard]] Vector::Index size() const noexcept;
    [[nodiscard]] std::span<const BasinObservation> points() const noexcept;
    [[nodiscard]] Vector normalized_residuals(const Vector& calculated) const;

private:
    std::vector<BasinObservation> points_;
};
