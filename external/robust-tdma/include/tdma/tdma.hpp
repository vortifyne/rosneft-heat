#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tdma {

template <class T>
concept Real = std::same_as<T, float> || std::same_as<T, double>;

/// Actual matrix entries: lower/upper have n-1 elements, diagonal has n >= 1.
template <Real T> struct MatrixView {
    std::span<const T> lower;
    std::span<const T> diagonal;
    std::span<const T> upper;
};

/// Classical sweep without pivoting. Prepare once, then solve any number of RHS.
/// Owns its factors; solve uses the output buffer for intermediate values.
template <Real T = double> class StandardSolver {
public:
    /// O(n). Reuses buffers for the same size; failure leaves the solver unprepared.
    void compute(MatrixView<T> matrix) {
        ready_ = false;
        const auto n = matrix.diagonal.size();
        if (n == 0 || matrix.lower.size() != n - 1 || matrix.upper.size() != n - 1) {
            throw std::invalid_argument("tdma: invalid diagonal sizes");
        }
        if (inverse_pivots_.size() != n) {
            upper_ = std::vector<T>(n - 1);
            multipliers_ = std::vector<T>(n - 1);
            inverse_pivots_ = std::vector<T>(n);
        }
        if (!std::isfinite(matrix.diagonal[0])) {
            throw std::invalid_argument("tdma: matrix contains NaN or infinity");
        }
        inverse_pivots_[0] = inverse_pivot(matrix.diagonal[0], 0);

        for (std::size_t i = 0; i + 1 < n; ++i) {
            const auto lower = matrix.lower[i];
            const auto upper = matrix.upper[i];
            const auto diagonal = matrix.diagonal[i + 1];
            if (!std::isfinite(lower) || !std::isfinite(upper) || !std::isfinite(diagonal)) {
                throw std::invalid_argument("tdma: matrix contains NaN or infinity");
            }
            upper_[i] = upper;
            multipliers_[i] = lower * inverse_pivots_[i];
            if (!std::isfinite(multipliers_[i])) {
                throw std::runtime_error("tdma: nonfinite factor at step " + std::to_string(i + 1));
            }
            const auto pivot = diagonal - multipliers_[i] * upper;
            inverse_pivots_[i + 1] = inverse_pivot(pivot, i + 1);
        }
        ready_ = true;
    }

    /// O(n), no allocation. Exact RHS/output aliasing is allowed; partial overlap is not.
    void solve(std::span<const T> rhs, std::span<T> solution) const {
        const auto n = size();
        check_vectors(n, rhs.size(), solution.size());
        solution[0] = rhs[0];
        for (std::size_t i = 1; i < n; ++i) {
            solution[i] = rhs[i] - multipliers_[i - 1] * solution[i - 1];
        }
        solution[n - 1] *= inverse_pivots_[n - 1];
        for (std::size_t i = n - 1; i-- > 0;) {
            solution[i] = (solution[i] - upper_[i] * solution[i + 1]) * inverse_pivots_[i];
        }
    }

    /// Convenience overload allocating the returned vector.
    [[nodiscard]] std::vector<T> solve(std::span<const T> rhs) const {
        std::vector<T> solution(size());
        solve(rhs, solution);
        return solution;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return ready_ ? inverse_pivots_.size() : 0;
    }

private:
    static T inverse_pivot(T pivot, std::size_t step) {
        if (pivot == T{} || !std::isfinite(pivot)) {
            throw std::runtime_error("tdma: zero or nonfinite pivot at step " +
                                     std::to_string(step));
        }
        const auto inverse = T{1} / pivot;
        if (!std::isfinite(inverse)) {
            throw std::runtime_error("tdma: nonfinite inverse pivot at step " +
                                     std::to_string(step));
        }
        return inverse;
    }

    static void check_vectors(std::size_t n, std::size_t rhs_size, std::size_t solution_size) {
        if (n == 0) {
            throw std::logic_error("tdma: compute a matrix before solve");
        }
        if (rhs_size != n || solution_size != n) {
            throw std::invalid_argument("tdma: vector size does not match the matrix");
        }
    }

    bool ready_{};
    std::vector<T> upper_;
    std::vector<T> multipliers_;
    std::vector<T> inverse_pivots_;
};

/// Samarskii--Nikolaev nonmonotone sweep for general nonsingular tridiagonal matrices.
/// Owns its factors and scratch; simultaneous solves require separate objects.
template <Real T = double> class PivotingSolver {
public:
    /// O(n). Selects pivots once; same-size buffers are reused. Failure leaves it unprepared.
    void compute(MatrixView<T> matrix) {
        ready_ = false;
        const auto n = matrix.diagonal.size();
        if (n == 0 || matrix.lower.size() != n - 1 || matrix.upper.size() != n - 1) {
            throw std::invalid_argument("tdma: invalid diagonal sizes");
        }
        if (beta_.size() != n) {
            alpha_ = std::vector<T>(n - 1);
            rhs_scale_ = std::vector<T>(n - 1);
            reduced_rhs_beta_ = std::vector<T>(n - 1);
            following_rhs_beta_ = std::vector<T>(n - 1);
            chi_ = std::vector<std::size_t>(n - 1);
            beta_ = std::vector<T>(n);
        }

        // Source signs: c[i] = diagonal[i], a[i+1] = -lower[i], b[i] = -upper[i].
        auto C = matrix.diagonal[0];
        auto A = n > 1 ? -matrix.lower[0] : T{};
        if (!std::isfinite(C) || !std::isfinite(A)) {
            throw std::invalid_argument("tdma: matrix contains NaN or infinity");
        }
        auto previous_chi = std::size_t{0};
        for (std::size_t i = 0; i + 1 < n; ++i) {
            const auto b = -matrix.upper[i];
            const auto c = matrix.diagonal[i + 1];
            const auto a = i + 2 < n ? -matrix.lower[i + 1] : T{};
            if (!std::isfinite(b) || !std::isfinite(c) || !std::isfinite(a)) {
                throw std::invalid_argument("tdma: matrix contains NaN or infinity");
            }
            if (std::abs(C) >= std::abs(b)) {
                rhs_scale_[i] = inverse_pivot(C, i);
                alpha_[i] = b * rhs_scale_[i];
                reduced_rhs_beta_[i] = A;
                following_rhs_beta_[i] = T{};
                C = c - A * alpha_[i];
                A = a;
                chi_[i] = i + 1;
            } else {
                rhs_scale_[i] = -inverse_pivot(b, i);
                alpha_[i] = -C * rhs_scale_[i];
                reduced_rhs_beta_[i] = -c;
                following_rhs_beta_[i] = a;
                C = c * alpha_[i] - A;
                A = a * alpha_[i];
                chi_[i] = previous_chi;
            }
            if (!std::isfinite(alpha_[i]) || !std::isfinite(C) || !std::isfinite(A)) {
                throw std::runtime_error("tdma: nonfinite factor at step " + std::to_string(i));
            }
            previous_chi = chi_[i];
        }
        inverse_final_pivot_ = inverse_pivot(C, n - 1);
        ready_ = true;
    }

    /// O(n), no allocation. Exact RHS/output aliasing is allowed; partial overlap is not.
    void solve(std::span<const T> rhs, std::span<T> solution) {
        const auto n = size();
        check_vectors(n, rhs.size(), solution.size());
        if (n == 1) {
            solution[0] = rhs[0] * inverse_final_pivot_;
            return;
        }

        auto F = rhs[0];
        auto Phi = rhs[1];
        auto* beta = beta_.data();
        for (std::size_t i = 0; i + 1 < n; ++i) {
            beta[i] = rhs_scale_[i] * F;
            F = Phi + reduced_rhs_beta_[i] * beta[i];
            if (i + 2 < n) {
                Phi = rhs[i + 2] + following_rhs_beta_[i] * beta[i];
            }
        }

        solution[chi_.back()] = F * inverse_final_pivot_;
        for (std::size_t i = n - 1; i-- > 0;) {
            const auto chi = chi_[i];
            // A regular step eliminates the previous chi; a pivot step eliminates i+1.
            const auto theta = chi == i + 1 ? (i == 0 ? 0 : chi_[i - 1]) : i + 1;
            solution[theta] = alpha_[i] * solution[chi] + beta[i];
        }
    }

    /// Convenience overload allocating the returned vector.
    [[nodiscard]] std::vector<T> solve(std::span<const T> rhs) {
        std::vector<T> solution(size());
        solve(rhs, solution);
        return solution;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return ready_ ? beta_.size() : 0;
    }

private:
    static T inverse_pivot(T pivot, std::size_t step) {
        if (pivot == T{} || !std::isfinite(pivot)) {
            throw std::runtime_error("tdma: zero or nonfinite pivot at step " +
                                     std::to_string(step));
        }
        const auto inverse = T{1} / pivot;
        if (!std::isfinite(inverse)) {
            throw std::runtime_error("tdma: nonfinite inverse pivot at step " +
                                     std::to_string(step));
        }
        return inverse;
    }

    static void check_vectors(std::size_t n, std::size_t rhs_size, std::size_t solution_size) {
        if (n == 0) {
            throw std::logic_error("tdma: compute a matrix before solve");
        }
        if (rhs_size != n || solution_size != n) {
            throw std::invalid_argument("tdma: vector size does not match the matrix");
        }
    }

    T inverse_final_pivot_{};
    bool ready_{};
    std::vector<T> alpha_;
    std::vector<T> rhs_scale_;
    std::vector<T> reduced_rhs_beta_;
    std::vector<T> following_rhs_beta_;
    std::vector<std::size_t> chi_;
    std::vector<T> beta_;
};

} // namespace tdma
