#include "basin_1d/inverse/fit_initial_flux_1d.hpp"

#include "basin_1d/inverse/section_observations.hpp"
#include "numerics/optimization/gauss_newton_optimizer.hpp"

#include <limits>
#include <ostream>

Vector fit_initial_flux_1d(const BasinForwardSolver1D& section,
                           const BasinHeatParameterSpace& space,
                           const BasinHeatObservationData& observations,
                           std::span<const std::string> wells,
                           const LeastSquaresSolveRequest& options, std::ostream& log) {
    auto points = select_section_observations(section.section(), observations.points(), wells, log);
    if (points.empty()) {
        log << "1D: no observations in section; using constant initial flux for 2D\n";
        return space.values();
    }
    const BasinHeatObservationData selected(std::move(points));
    const BoundedLeastSquaresProblem problem{
        .lower_bounds = space.normalized_lower_bounds(),
        .upper_bounds = space.normalized_upper_bounds(),
        .residual_count = selected.size(),
        .objective =
            [&](const Vector& eta, double tolerance, Eigen::MatrixXd* jacobian) {
                auto predictions = section.solve_at_wells(selected.points(), space, eta, jacobian);
                return selected.evaluate_predictions(std::move(predictions), tolerance, jacobian);
            },
    };
    Vector best = space.to_normalized(space.values());
    double best_objective = std::numeric_limits<double>::infinity();
    // Projected Gauss–Newton steps can increase the residual; keep the best evaluated start.
    const auto result = GaussNewtonOptimizer{}.solve(
        problem, best, options, [&](int, int, const Vector& eta, const Vector& residuals) {
            const double value = least_squares_objective(residuals);
            if (value < best_objective) {
                best = eta;
                best_objective = value;
            }
        });
    log << "1D: " << result.function_evaluations
        << " evaluations, best objective=" << best_objective << "; using fitted flux for 2D\n";
    return space.to_physical(best);
}
