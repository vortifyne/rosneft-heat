#include "basin/inverse/basin_heat_residual_evaluator.hpp"

#include "basin/basin_snapshot.hpp"
#include "basin/mesh/col_field_transfer.hpp"

#include <algorithm>
#include <numeric>
#include <utility>

namespace {

// Reconstruct linearly from centers, including points at cell boundaries.
std::array<std::size_t, 2> neighboring(std::span<const std::size_t> indices, double value,
                                       const auto& coordinate) {
    auto upper =
        std::lower_bound(indices.begin(), indices.end(), value,
                         [&](std::size_t index, double x) { return coordinate(index) < x; });
    const auto right =
        std::min(std::max(std::size_t{1}, static_cast<std::size_t>(upper - indices.begin())),
                 indices.size() - 1);
    return {right == 0 ? indices.front() : indices[right - 1], indices[right]};
}

double fraction(double value, double first, double second) {
    return first == second ? 0.0 : (value - first) / (second - first);
}

} // namespace

BasinHeatResidualEvaluator::BasinHeatResidualEvaluator(BasinForwardSolver forward,
                                                       BasinHeatObservationData observations)
    : forward_(std::move(forward)), observations_(std::move(observations)),
      parameter_space_(static_cast<Vector::Index>(forward_.configs().size())) {
    forward_.set_intermediate_snapshots(false);
    const auto& grid = forward_.final_grid();
    std::vector<std::size_t> columns(grid.columns.size());
    std::iota(columns.begin(), columns.end(), 0);
    const auto x = [&](std::size_t column) { return 0.5 * (grid.x[column] + grid.x[column + 1]); };
    const auto depths = horizontal_node_depths(grid);
    std::vector<std::vector<std::size_t>> nodes(grid.columns.size());
    for (std::size_t c = 0; c < nodes.size(); ++c) {
        nodes[c].resize(grid.columns[c].size() + 1);
        std::iota(nodes[c].begin(), nodes[c].end(), grid.columns[c].front() + c);
    }
    interpolation_.reserve(static_cast<std::size_t>(observations_.size()));
    for (const auto& point : observations_.points()) {
        const auto [left, right] = neighboring(columns, point.x, x);
        const bool nodal = point.kind == BasinObservable::reflectance;
        const auto z = [&](std::size_t index) {
            return nodal ? depths[index] : grid.cells[index].center().z;
        };
        const auto [l0, l1] = neighboring(nodal ? nodes[left] : grid.columns[left], point.z, z);
        const auto [r0, r1] = neighboring(nodal ? nodes[right] : grid.columns[right], point.z, z);
        const double wx = fraction(point.x, x(left), x(right));
        const double wl = fraction(point.z, z(l0), z(l1));
        const double wr = fraction(point.z, z(r0), z(r1));
        interpolation_.push_back(
            {.cells = {l0, l1, r0, r1},
             .weights = {(1.0 - wx) * (1.0 - wl), (1.0 - wx) * wl, wx * (1.0 - wr), wx * wr}});
    }
}

const BasinHeatParameterSpace& BasinHeatResidualEvaluator::parameter_space() const noexcept {
    return parameter_space_;
}

const BasinHeatObservationData& BasinHeatResidualEvaluator::observations() const noexcept {
    return observations_;
}

BasinForwardSolver& BasinHeatResidualEvaluator::forward_solver() noexcept {
    return forward_;
}

Vector BasinHeatResidualEvaluator::evaluate(const Vector& normalized) {
    forward_.set_parameters(parameter_space_.to_physical(normalized));
    Vector calculated(observations_.size());
    (void)forward_.solve([&](const BasinSnapshot& snapshot) {
        if (snapshot.epoch + 1 != forward_.configs().size())
            return;
        for (Vector::Index i = 0; i < calculated.size(); ++i) {
            const auto index = static_cast<std::size_t>(i);
            const auto values = observations_.points()[index].kind == BasinObservable::temperature
                                    ? snapshot.T
                                    : snapshot.Ro;
            const auto& interpolation = interpolation_[index];
            double value = 0.0;
            for (std::size_t j = 0; j < interpolation.cells.size(); ++j)
                value += interpolation.weights[j] * values[interpolation.cells[j]];
            calculated[i] = value;
        }
    });
    return observations_.normalized_residuals(calculated);
}
