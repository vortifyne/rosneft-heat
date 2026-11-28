#pragma once

#include "basin/forward/basin_well_sample.hpp"
#include "numerics/optimization/bounded_least_squares_problem.hpp"

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

    [[nodiscard]] LeastSquaresEvaluation evaluate_predictions(Vector predictions, double tolerance,
                                                              Eigen::MatrixXd* jacobian) const;
    [[nodiscard]] Vector residuals_in_error_units(Vector weighted) const;

private:
    std::vector<BasinObservation> points_;
};
