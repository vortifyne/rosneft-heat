#include "discretization/least_squares_gradient.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>

namespace {

bool finite(const Eigen::Vector2d& vector) {
    return std::isfinite(vector.x()) && std::isfinite(vector.y());
}

} // namespace

LeastSquaresGradient::LeastSquaresGradient(
    const std::size_t cell_count, std::span<const LeastSquaresConnection> connections,
    std::span<const LeastSquaresBoundarySample> boundary_samples)
    : stencils_(cell_count), boundary_sample_count_(boundary_samples.size()) {
    if (cell_count == 0) {
        throw std::invalid_argument("Least-squares gradient requires at least one cell");
    }

    std::vector<std::vector<Sample>> samples(cell_count);
    for (const auto& connection : connections) {
        if (connection.first_cell >= cell_count || connection.second_cell >= cell_count ||
            connection.first_cell == connection.second_cell || !finite(connection.displacement) ||
            !(connection.displacement.squaredNorm() > 0.0)) {
            throw std::invalid_argument("Invalid least-squares cell connection");
        }
        samples[connection.first_cell].push_back(
            {Source::cell, connection.second_cell, connection.displacement});
        samples[connection.second_cell].push_back(
            {Source::cell, connection.first_cell, -connection.displacement});
    }
    for (std::size_t index = 0; index < boundary_samples.size(); ++index) {
        const auto& sample = boundary_samples[index];
        if (sample.cell >= cell_count || !finite(sample.displacement) ||
            !(sample.displacement.squaredNorm() > 0.0)) {
            throw std::invalid_argument("Invalid least-squares boundary sample");
        }
        samples[sample.cell].push_back({Source::boundary, index, sample.displacement});
    }

    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        Eigen::Matrix2d normal_matrix = Eigen::Matrix2d::Zero();
        for (const Sample& sample : samples[cell]) {
            const double weight = 1.0 / sample.displacement.squaredNorm();
            normal_matrix.noalias() +=
                weight * sample.displacement * sample.displacement.transpose();
        }

        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(normal_matrix);
        if (eigensolver.info() != Eigen::Success) {
            throw std::runtime_error("Cannot factor least-squares gradient matrix");
        }
        Eigen::Matrix2d inverse = Eigen::Matrix2d::Zero();
        const double maximum_eigenvalue = eigensolver.eigenvalues().maxCoeff();
        if (!(maximum_eigenvalue > 0.0) || !std::isfinite(maximum_eigenvalue)) {
            throw std::runtime_error("Cell has no usable samples for least-squares gradient");
        }
        for (Eigen::Index index = 0; index < 2; ++index) {
            const double eigenvalue = eigensolver.eigenvalues()[index];
            if (eigenvalue > 1.0e-12 * maximum_eigenvalue) {
                const Eigen::Vector2d direction = eigensolver.eigenvectors().col(index);
                inverse.noalias() += (direction * direction.transpose()) / eigenvalue;
            }
        }
        auto& stencil = stencils_[cell];
        stencil.reserve(samples[cell].size());
        for (const Sample& sample : samples[cell]) {
            const double weight = 1.0 / sample.displacement.squaredNorm();
            stencil.push_back(
                {sample.source, sample.index, inverse * (weight * sample.displacement)});
        }
    }
}

std::size_t LeastSquaresGradient::size() const noexcept {
    return stencils_.size();
}

std::size_t LeastSquaresGradient::boundary_sample_count() const noexcept {
    return boundary_sample_count_;
}

void LeastSquaresGradient::reconstruct(const std::span<const double> cell_values,
                                       const std::span<const double> boundary_values,
                                       const std::span<Gradient> gradients) const {
    if (cell_values.size() != size() || gradients.size() != size() ||
        boundary_values.size() != boundary_sample_count_) {
        throw std::invalid_argument("Least-squares gradient field sizes do not match");
    }

    for (std::size_t cell = 0; cell < size(); ++cell) {
        Gradient gradient = Gradient::Zero();
        for (const Term& term : stencils_[cell]) {
            const double sample_value =
                term.source == Source::cell ? cell_values[term.index] : boundary_values[term.index];
            gradient.noalias() += term.coefficient * (sample_value - cell_values[cell]);
        }
        gradients[cell] = gradient;
    }
}
