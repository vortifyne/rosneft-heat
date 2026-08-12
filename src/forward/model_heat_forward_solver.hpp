#pragma once

#include "linear/linear_solver.hpp"
#include "mesh/regular_grid.hpp"
#include "model/model_heat_problem.hpp"
#include "model/model_heat_state.hpp"
#include "nonlinear/nonlinear_solver.hpp"
#include "time/time_integrator.hpp"

#include <array>
#include <vector>

struct ModelHeatForwardSettings {
    double initial_time;
    double final_time;
    double time_step;
    NonlinearSolveRequest nonlinear;
    LinearSolveRequest linear;
    LinearSolverKind linear_solver = LinearSolverKind::umfpack_lu;
};

struct ModelHeatForwardResult {
    TimeIntegrationResult integration;
    ModelHeatState final_state;
    Vector calculated_temperature;
    double elapsed_time_seconds = 0.0;

    bool completed() const noexcept {
        return integration.completed();
    }
};

class ModelHeatForwardSolver {
public:
    ModelHeatForwardSolver(RegularGrid grid, ModelHeatProblem problem,
                           ModelHeatForwardSettings settings,
                           ModelHeatParameters initial_parameters);

    void set_surface_temperature(double value);
    void set_basal_heat_flux(double value);
    void set_parameters(const ModelHeatParameters& parameters);

    const ModelHeatParameters& parameters() const noexcept;

    ModelHeatForwardResult solve() const;

private:
    struct ObservationStencil {
        std::array<Vector::Index, 4> indices;
        std::array<double, 4> weights;
    };

    ObservationStencil make_observation_stencil(const ObservationPoint& point) const;
    Vector interpolate_observations(const Vector& temperature) const;

    RegularGrid grid_;
    ModelHeatProblem problem_;
    ModelHeatForwardSettings settings_;
    ModelHeatParameters parameters_;
    Vector initial_temperature_;
    std::vector<ObservationStencil> observation_stencils_;
};
