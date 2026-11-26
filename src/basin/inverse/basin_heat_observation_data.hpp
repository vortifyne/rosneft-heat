#pragma once

#include "basin/forward/basin_well_sample.hpp"
#include "numerics/algebra/vector.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

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

private:
    std::vector<BasinObservation> points_;
};
