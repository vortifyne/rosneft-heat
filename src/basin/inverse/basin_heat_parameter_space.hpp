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
    // Evaluate at the supplied eta and return their flux derivatives in dq_deta.
    [[nodiscard]] double at(double age_ma, const Vector& eta, Vector& dq_deta) const;

    [[nodiscard]] Vector to_normalized(const Vector& parameters) const;
    [[nodiscard]] Vector to_physical(const Vector& eta, Eigen::MatrixXd* jacobian = nullptr) const;
    [[nodiscard]] Vector::Index gradient_penalty_size(double regularization) const;
    void append_gradient_penalty(double regularization, const Vector& eta, Vector& residuals,
                                 Eigen::MatrixXd* jacobian) const;
    [[nodiscard]] Vector normalized_lower_bounds() const;
    [[nodiscard]] Vector normalized_upper_bounds() const;

private:
    std::vector<double> ages_ma_; // Oldest to youngest.
    Vector flux_;                 // W/m².
};
