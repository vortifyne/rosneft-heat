#include "basin/inverse/basin_objective.hpp"

#include "basin/geometry/basin_geometry.hpp"
#include "basin/mesh/col_field_transfer.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace {

// Queries can interpolate or extrapolate between the outer two points.
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

BasinObjective::BasinObjective(BasinForwardSolver forward, BasinHeatObservationData observations)
    : forward_(std::move(forward)), observations_(std::move(observations)) {
    forward_.set_intermediate_snapshots(false);
    const auto& grid = forward_.final_grid();
    std::vector<std::size_t> columns(grid.columns.size());
    std::iota(columns.begin(), columns.end(), 0);
    const auto x = [&](std::size_t column) { return 0.5 * (grid.x[column] + grid.x[column + 1]); };
    const auto depths = horizontal_node_depths(grid);
    std::vector<std::vector<std::size_t>> temperatures(grid.columns.size());
    std::map<int, std::vector<std::size_t>> layer_columns;
    const auto count = grid.cells.size();
    for (std::size_t c = 0; c < grid.columns.size(); ++c) {
        for (auto cell : grid.columns[c]) {
            temperatures[c].push_back(count + cell + c);
            temperatures[c].push_back(cell);
            auto& supporting = layer_columns[grid.cells[cell].layer_id];
            if (supporting.empty() || supporting.back() != c)
                supporting.push_back(c);
        }
        temperatures[c].push_back(count + grid.columns[c].back() + c + 1);
    }
    const auto& cfg = forward_.configs().back();
    const auto geometry = make_basin_geometry(cfg);
    interpolation_.reserve(static_cast<std::size_t>(observations_.size()));
    for (const auto& point : observations_.points()) {
        if (point.kind == BasinObservable::reflectance) {
            const double geometry_x = std::clamp(point.x, geometry.x.front(), geometry.x.back());
            const BasinLayerRegion* layer = nullptr;
            double relative_depth = 0.0;
            for (std::size_t i = 0; i < geometry.layers.size(); ++i) {
                const auto& candidate = geometry.layers[i];
                const double top = geometry.boundary_at(i, geometry_x);
                const double bottom = geometry.boundary_at(i + 1, geometry_x);
                if (geometry_x < candidate.x_min || geometry_x > candidate.x_max || top >= bottom ||
                    !layer_columns.contains(candidate.id))
                    continue;
                layer = &candidate;
                relative_depth = std::clamp(fraction(point.z, top, bottom), 0.0, 1.0);
                if (point.z < bottom)
                    break;
            }
            if (!layer)
                throw std::invalid_argument("Reflectance observation has no material at well " +
                                            point.well);
            const auto [left, right] = neighboring(layer_columns.at(layer->id), point.x, x);
            const auto sample = [&](std::size_t column) {
                const auto& cells = grid.columns[column];
                const auto first = std::find_if(cells.begin(), cells.end(), [&](std::size_t cell) {
                    return grid.cells[cell].layer_id == layer->id;
                });
                const auto end = std::find_if(first, cells.end(), [&](std::size_t cell) {
                    return grid.cells[cell].layer_id != layer->id;
                });
                const double z = std::lerp(grid.cells[*first].z_top,
                                           grid.cells[*(end - 1)].z_bottom, relative_depth);
                const auto upper = std::lower_bound(first, end, z, [&](std::size_t cell, double d) {
                    return grid.cells[cell].z_bottom < d;
                });
                const auto& cell = grid.cells[*std::min(upper, end - 1)];
                return std::pair{cell.maturity_top, fraction(z, cell.z_top, cell.z_bottom)};
            };
            const auto [l0, wl] = sample(left);
            const auto [r0, wr] = sample(right);
            const double wx = std::clamp(fraction(point.x, x(left), x(right)), 0.0, 1.0);
            interpolation_.push_back(
                {.kind = point.kind,
                 .indices = {l0, l0 + 1, r0, r0 + 1},
                 .weights = {(1.0 - wx) * (1.0 - wl), (1.0 - wx) * wl, wx * (1.0 - wr), wx * wr}});
            continue;
        }
        if (point.z <= geometry.boundary_at(0, point.x)) {
            interpolation_.push_back({.kind = point.kind,
                                      .indices = {},
                                      .weights = {},
                                      .constant = cfg.T_surf.at(point.x)});
            continue;
        }
        const auto [left, right] = neighboring(columns, point.x, x);
        const auto z = [&](std::size_t index) {
            return index < count ? grid.cells[index].center().z : depths[index - count];
        };
        const auto [l0, l1] = neighboring(temperatures[left], point.z, z);
        const auto [r0, r1] = neighboring(temperatures[right], point.z, z);
        const double wx = fraction(point.x, x(left), x(right));
        const double wl = fraction(point.z, z(l0), z(l1));
        const double wr = fraction(point.z, z(r0), z(r1));
        interpolation_.push_back(
            {.kind = point.kind,
             .indices = {l0, l1, r0, r1},
             .weights = {(1.0 - wx) * (1.0 - wl), (1.0 - wx) * wl, wx * (1.0 - wr), wx * wr}});
    }
}

const BasinHeatParameterSpace& BasinObjective::parameter_space() const noexcept {
    return forward_.parameter_space();
}

const BasinHeatObservationData& BasinObjective::observations() const noexcept {
    return observations_;
}

BasinForwardSolver& BasinObjective::forward_solver() noexcept {
    return forward_;
}

Vector BasinObjective::residuals_in_error_units(Vector weighted) const {
    return observations_.residuals_in_error_units(std::move(weighted));
}

LeastSquaresEvaluation BasinObjective::evaluate(const Vector& eta, double tolerance,
                                                Eigen::MatrixXd* jacobian) {
    forward_.set_parameters(parameter_space().to_physical(eta));
    Vector residuals = forward_.solve_at_wells(interpolation_, jacobian, &eta);
    if (jacobian && jacobian->cols() != eta.size())
        throw std::runtime_error("Unexpected basin Jacobian column count");
    return observations_.evaluate_predictions(std::move(residuals), tolerance, jacobian);
}
