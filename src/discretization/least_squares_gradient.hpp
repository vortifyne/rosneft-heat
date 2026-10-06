#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <span>
#include <vector>

struct LeastSquaresConnection {
    std::size_t first_cell;
    std::size_t second_cell;
    Eigen::Vector2d displacement;
};

struct LeastSquaresBoundarySample {
    std::size_t cell;
    Eigen::Vector2d displacement;
};

class LeastSquaresGradient {
public:
    using Gradient = Eigen::Vector2d;

    struct DerivativeTerm {
        std::size_t cell;
        Gradient coefficient;
    };

    LeastSquaresGradient(std::size_t cell_count,
                         std::span<const LeastSquaresConnection> connections,
                         std::span<const LeastSquaresBoundarySample> boundary_samples);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t boundary_sample_count() const noexcept;
    [[nodiscard]] std::span<const DerivativeTerm> derivatives(std::size_t cell) const;

    void reconstruct(std::span<const double> cell_values, std::span<const double> boundary_values,
                     std::span<Gradient> gradients) const;

private:
    enum class Source {
        cell,
        boundary,
    };

    struct Sample {
        Source source;
        std::size_t index;
        Gradient displacement;
    };

    struct Term {
        Source source;
        std::size_t index;
        Gradient coefficient;
    };

    std::vector<std::vector<Term>> stencils_;
    std::vector<std::vector<DerivativeTerm>> derivatives_;
    std::size_t boundary_sample_count_ = 0;
};
