#pragma once

#include "basin/basin_input.hpp"
#include "discretization/heat_system.hpp"
#include "linear/vector.hpp"
#include "mesh/mesh2d.hpp"
#include "physics/easy_ro.hpp"

#include <memory>
#include <span>
#include <vector>

struct BasinCellFields {
    std::vector<double> layer_id;
    std::vector<double> lithotype_code;
    std::vector<double> porosity;
    std::vector<double> material_eta;
    std::vector<double> cell_area;
    std::vector<const LithotypeThermophysicalProperties*> lithotypes;
    std::vector<double> conductivity;
    std::vector<double> heat_capacity;
    std::vector<double> heat_production;
    std::vector<double> velocity_z;
};

struct BasinState {
    Mesh2D mesh;
    BasinCellFields fields;
    Vector temperature;
    std::vector<double> maturity;
};

struct BasinStateTransfer {
    BasinState state;
    double energy_imbalance = 0.0;
};

[[nodiscard]] std::span<const double> vector_values(const Vector& vector);

[[nodiscard]] BasinState make_initial_basin_state(const BasinConfiguration& configuration,
                                                  const BasinInput& input, Mesh2D mesh,
                                                  const EasyRoModel& easy_ro,
                                                  const ScalarProfile& surface_temperature,
                                                  const ScalarProfile& basal_heat_flux);

void update_thermophysical_properties(std::span<const double> temperature, BasinCellFields& fields);

[[nodiscard]] std::unique_ptr<HeatSystem>
make_heat_system(const Mesh2D& mesh, BasinCellFields& fields,
                 HeatBoundaryConditions boundary_conditions);

[[nodiscard]] double total_energy(const BasinState& state);

[[nodiscard]] BasinStateTransfer transfer_basin_state(
    const BasinState& old_state, Mesh2D new_mesh, const BasinConfiguration& old_configuration,
    const BasinConfiguration& new_configuration, const BasinInput& input,
    const EasyRoModel& easy_ro, const ScalarProfile& surface_temperature, double cell_size);
