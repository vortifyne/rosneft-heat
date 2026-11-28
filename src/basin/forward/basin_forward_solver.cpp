#include "basin/forward/basin_forward_solver.hpp"

#include "basin/basin_snapshot.hpp"
#include "basin/discretization/col_heat_system.hpp"
#include "basin/forward/basin_time_step.hpp"
#include "basin/geometry/basin_geometry_rdp.hpp"
#include "basin/mesh/col_epoch_grid.hpp"
#include "basin/mesh/col_field_transfer.hpp"
#include "basin/physics/easy_ro.hpp"
#include "basin/physics/thermal_properties.hpp"
#include "numerics/linear/umfpack_linear_solver.hpp"
#include "numerics/time/time_integrator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

std::vector<double> lithotype_breakpoints(const BasinConfig& cfg0,
                                          const BasinConfig* cfg1 = nullptr) {
    std::vector<double> result;
    const auto append = [&result](const BasinConfig& cfg) {
        for (const auto& layer : cfg.layers) {
            for (std::size_t i = 1; i < layer.x.size(); ++i) {
                if (layer.lithotype[i - 1] != layer.lithotype[i]) {
                    result.push_back(std::midpoint(layer.x[i - 1], layer.x[i]));
                }
            }
        }
    };
    append(cfg0);
    if (cfg1)
        append(*cfg1);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::span<const double> values(const Vector& v) {
    return {v.native().data(), static_cast<std::size_t>(v.size())};
}

std::vector<double> deposition_times(const ColGrid& grid, std::span<const BasinConfig> configs) {
    std::unordered_map<int, std::pair<double, double>> birth;
    for (std::size_t i = 0; i < configs.size(); ++i) {
        for (const auto& layer : configs[i].layers) {
            const double end = (configs.front().age_ma - configs[i].age_ma) * kSecondsPerMa;
            const double begin =
                i == 0 ? end : (configs.front().age_ma - configs[i - 1].age_ma) * kSecondsPerMa;
            birth.try_emplace(layer.id, begin, end);
        }
    }
    std::vector<double> result(grid.maturity_node_count);
    for (const auto& column : grid.columns) {
        auto first = column.begin();
        while (first != column.end()) {
            const auto& top = grid.cells[*first];
            const auto last = std::find_if_not(first, column.end(), [&](std::size_t index) {
                return grid.cells[index].layer_id == top.layer_id;
            });
            const auto& bottom = grid.cells[*(last - 1)];
            const auto [begin, end] = birth.at(top.layer_id);
            for (auto cell = first; cell != last; ++cell) {
                const double depth = grid.cells[*cell].z_top;
                result[grid.cells[*cell].maturity_top] =
                    std::lerp(end, begin, (depth - top.z_top) / (bottom.z_bottom - top.z_top));
            }
            result[bottom.maturity_top + 1] = begin;
            first = last;
        }
    }
    return result;
}

BasinHeatParameterSpace epoch_flux_space(std::span<const BasinConfig> configs) {
    std::vector<double> ages;
    for (const auto& cfg : configs)
        ages.push_back(cfg.age_ma);
    Vector flux(static_cast<Vector::Index>(configs.size()));
    flux.set_constant(0.075);
    return BasinHeatParameterSpace(std::move(ages), std::move(flux));
}

std::function<double(Point2D)> basal_flux(const BasinConfig& cfg0, const BasinConfig& cfg1,
                                          double theta, const BasinHeatParameterSpace* space,
                                          double age_ma) {
    if (space)
        return [q = space->at(age_ma)](Point2D) { return q; };
    return [&cfg0, &cfg1, theta](Point2D point) {
        return std::lerp(cfg0.q_basal.at(point.x), cfg1.q_basal.at(point.x), theta);
    };
}

HeatBoundaryData bc(const BasinConfig& cfg0, const BasinConfig& cfg1, double theta,
                    const BasinHeatParameterSpace* space, double age_ma) {
    return {
        .T_surf =
            [&cfg0, &cfg1, theta](Point2D point) {
                return std::lerp(cfg0.T_surf.at(point.x), cfg1.T_surf.at(point.x), theta);
            },
        .q_basal = basal_flux(cfg0, cfg1, theta, space, age_ma),
    };
}

void for_each_layer_cell(const ColGrid& grid, const auto& visit) {
    for (const auto& column : grid.columns) {
        auto first = column.begin();
        while (first != column.end()) {
            const auto& top = grid.cells[*first];
            const auto last = std::find_if_not(first, column.end(), [&](std::size_t index) {
                return grid.cells[index].layer_id == top.layer_id;
            });
            const double height = grid.cells[*(last - 1)].z_bottom - top.z_top;
            for (auto cell = first; cell != last; ++cell) {
                const auto& value = grid.cells[*cell];
                visit(value, (value.center().z - top.z_top) / height);
            }
            first = last;
        }
    }
}

void observe_state(const ColGrid& grid, const Vector& T, std::span<const double> I,
                   const EasyRoModel& easy_ro, BasinSnapshot::Kind kind, std::size_t epoch,
                   int step, double age_ma, const BasinObserver& observer, BasinTimings& timings) {
    if (!observer)
        return;
    auto start = Clock::now();
    std::vector<double> Ro(grid.maturity_node_count);
    easy_ro.reflectance(I, Ro);
    timings.easy_ro_s += elapsed_seconds(start);
    start = Clock::now();
    observer({.kind = kind,
              .epoch = epoch,
              .step = step,
              .age_ma = age_ma,
              .grid = grid,
              .T = values(T),
              .I = I,
              .Ro = Ro});
    timings.output_s += elapsed_seconds(start);
}

TimeIntegrationResult advance_bdf1(TimeIntegrator& integrator, const ColHeatSystem& system,
                                   double t_next, double dt, double end_age,
                                   const BasinForwardSettings& settings) {
    integrator.set_timestep(dt);
    const auto step = integrator.advance_to(system, t_next, settings.nonlinear, settings.linear);
    if (!step.completed()) {
        throw std::runtime_error(
            "basin nonlinear solve failed at age " + std::to_string(end_age) +
            ": nonlinear_status=" +
            std::to_string(
                step.last_nonlinear_status ? static_cast<int>(*step.last_nonlinear_status) : -1) +
            " linear_status=" +
            std::to_string(step.last_linear_status ? static_cast<int>(*step.last_linear_status)
                                                   : -1) +
            " nonlinear_iterations=" + std::to_string(step.nonlinear_iterations) +
            " residual=" + std::to_string(step.last_r_norm));
    }
    return step;
}

Eigen::MatrixXd advance_cell_sensitivities(const ColHeatSystem& system, const Vector& T, double dt,
                                           const Eigen::MatrixXd& previous, const Vector& dq) {
    SparseMatrix matrix(T.size(), T.size(), SparseStorageOrder::csc);
    auto rhs = system.assemble_ddm_step(T, dt, previous, dq, matrix);
    return UmfpackLinearSolver::solve_columns(matrix, std::move(rhs));
}

Eigen::MatrixXd material_sensitivities(const ColGrid& grid, const Eigen::MatrixXd& face_S) {
    Eigen::MatrixXd node_S(grid.maturity_node_count, face_S.cols());
    for (const auto& cell : grid.cells)
        node_S.middleRows(cell.maturity_top, 2) = face_S.middleRows(cell.index + cell.column, 2);
    return node_S;
}

} // namespace

BasinForwardSolver::BasinForwardSolver(BasinInput input, BasinForwardSettings options)
    : input_(std::move(input)), settings_(std::move(options)), easy_ro_(input_.easy_ro()),
      parameter_space_(epoch_flux_space(input_.configs())) {
    const auto configs = input_.configs();
    const auto& settings = settings_;
    if (settings.nonlinear.method != NonlinearMethod::picard || configs.size() < 2 ||
        !(settings.dx > 0.0) || !(settings.dz > 0.0) || !(settings.dz_min > 0.0) ||
        !(settings.dt_ma > 0.0) || !(settings.nonlinear.rtol > 0.0) ||
        !(settings.nonlinear.rtol < 1.0)) {
        throw std::invalid_argument("Invalid basin configurations or settings");
    }
    std::vector<BasinGeometry> geometries;
    geometries.reserve(configs.size());
    for (const auto& cfg : configs)
        geometries.push_back(
            simplify_basin_geometry_rdp(make_basin_geometry(cfg), settings.rdp_tol));
    epochs_.reserve(configs.size() - 1);
    for (std::size_t i = 0; i + 1 < configs.size(); ++i) {
        if (!(configs[i].age_ma > configs[i + 1].age_ma))
            throw std::invalid_argument("basin ages must decrease");
        EpochData data;
        data.mesh = make_col_epoch_grid(geometries[i], geometries[i + 1], settings.dx, settings.dz,
                                        settings.dz_min, settings.axis_rdp_tol,
                                        lithotype_breakpoints(configs[i], &configs[i + 1]));
        data.initial_grid = data.mesh.at(0.0);
        // Use the final layer coordinates: the new cells are compressed at the start.
        data.deposition_times = deposition_times(data.mesh.at(1.0), configs);
        const auto count = data.initial_grid.cells.size();
        data.lithotypes.resize(count);
        data.phi0.resize(count);
        data.phi1.resize(count);
        for (const auto& cell : data.initial_grid.cells) {
            const double x = cell.center().x;
            const auto& layer0 = data.mesh.born_layer_id == cell.layer_id
                                     ? configs[i + 1].layer(cell.layer_id)
                                     : configs[i].layer(cell.layer_id);
            data.lithotypes[cell.index] = layer0.lithotype_at(x);
            const auto& layer1 = configs[i + 1].layer(cell.layer_id);
            data.phi0[cell.index] = {layer0.phi_at(x, 0.0), layer0.phi_at(x, 1.0)};
            data.phi1[cell.index] = {layer1.phi_at(x, 0.0), layer1.phi_at(x, 1.0)};
        }
        epochs_.push_back(std::move(data));
    }
    const auto& geometry = geometries.back();
    const auto knots = settings.axis_rdp_tol
                           ? simplify_basin_geometry_rdp(geometry, *settings.axis_rdp_tol).x
                           : geometry.x;
    final_grid_ = make_col_grid(
        geometry, make_column_axis(knots, settings.dx, lithotype_breakpoints(configs.back())),
        settings.dz);
    initial_grid_ = make_col_grid(
        geometries.front(),
        make_column_axis(geometries.front().x, settings.dx, lithotype_breakpoints(configs.front())),
        settings.dz);
}

void BasinForwardSolver::EpochData::update_porosity(const ColGrid& grid, double theta,
                                                    std::span<double> phi) const {
    for_each_layer_cell(grid, [&](const ColCell& cell, double depth) {
        const auto i = cell.index;
        const double top = std::lerp(phi0[i][0], phi1[i][0], theta);
        const double bottom = std::lerp(phi0[i][1], phi1[i][1], theta);
        phi[i] = std::lerp(top, bottom, depth);
    });
}

std::vector<const LithotypeProperties*>
BasinForwardSolver::EpochData::materials(const BasinInput& input) const {
    std::vector<const LithotypeProperties*> result(lithotypes.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = &input.lithotype(lithotypes[i]);
    return result;
}

void BasinForwardSolver::advance_temperature_step(std::size_t index, double age_begin_ma,
                                                  double age_end_ma, const Vector& eta, Vector& T,
                                                  Eigen::MatrixXd* jacobian,
                                                  Eigen::MatrixXd* face_jacobian) const {
    if (face_jacobian && (!jacobian || jacobian == face_jacobian))
        throw std::invalid_argument("Face sensitivities require a separate cell Jacobian");
    const auto& epoch = epochs_.at(index);
    const auto& first = configs()[index];
    const auto& last = configs()[index + 1];
    if (!(first.age_ma >= age_begin_ma && age_begin_ma > age_end_ma && age_end_ma >= last.age_ma))
        throw std::invalid_argument("Temperature step must lie within one epoch");
    const auto& knots = parameter_space_.ages_ma();
    const auto knot = std::upper_bound(knots.begin(), knots.end(), age_begin_ma, std::greater<>{});
    if (knot != knots.end() && *knot > age_end_ma)
        throw std::invalid_argument("End the temperature step at the next basal flux knot");
    if (T.size() != static_cast<Vector::Index>(epoch.initial_grid.cells.size()) || !T.all_finite())
        throw std::invalid_argument("Expected one finite initial temperature per epoch cell");
    if (jacobian &&
        (jacobian->rows() != T.size() || jacobian->cols() != eta.size() || !jacobian->allFinite()))
        throw std::invalid_argument(
            "Expected finite initial sensitivities for every cell and parameter");

    const double theta = (first.age_ma - age_end_ma) / (first.age_ma - last.age_ma);
    const auto grid = epoch.mesh.at(theta);
    std::vector<double> phi(grid.cells.size());
    epoch.update_porosity(grid, theta, phi);
    const auto lithotypes = epoch.materials(input_);
    auto boundaries = bc(first, last, theta, nullptr, age_end_ma);
    const double mid_age = std::midpoint(age_begin_ma, age_end_ma);
    Vector dq_deta;
    double q;
    if (jacobian) {
        q = parameter_space_.at(mid_age, eta, dq_deta);
    } else {
        const BasinHeatParameterSpace flux(parameter_space_.ages_ma(),
                                           parameter_space_.to_physical(eta));
        q = flux.at(mid_age);
    }
    boundaries.q_basal = [q](Point2D) { return q; };
    ColHeatSystem system(grid, phi, lithotypes, kBasinFluid, std::move(boundaries));
    system.set_flux_scheme(settings_.scheme);
    TimeIntegrator integrator;
    integrator.set_linear_solver(settings_.linear_solver);
    integrator.set_preconditioner_reuse(settings_.pc_reuse);
    integrator.set_initial_solution(0.0, T);
    const double dt = (age_begin_ma - age_end_ma) * kSecondsPerMa;
    (void)advance_bdf1(integrator, system, dt, dt, age_end_ma, settings_);
    const Vector& accepted_T = integrator.current_snapshot().u;
    if (jacobian) {
        auto cell_S = advance_cell_sensitivities(system, accepted_T, dt, *jacobian, dq_deta);
        if (face_jacobian) {
            (void)parameter_space_.at(age_end_ma, eta, dq_deta);
            *face_jacobian = system.horizontal_face_sensitivities(accepted_T, cell_S, dq_deta);
        }
        *jacobian = std::move(cell_S);
    }
    T = accepted_T;
}

Vector BasinForwardSolver::solve_initial_temperature(Eigen::MatrixXd* jacobian,
                                                     Eigen::MatrixXd* face_jacobian) const {
    if (face_jacobian && (!jacobian || jacobian == face_jacobian))
        throw std::invalid_argument("Face sensitivities require a separate cell Jacobian");
    if (jacobian && !use_parameterized_flux_)
        throw std::logic_error(
            "Set basal flux parameters before requesting the stationary Jacobian");
    const auto& cfg = input_.configs().front();
    Vector T(static_cast<Vector::Index>(initial_grid_.cells.size()));
    std::vector<double> phi(initial_grid_.cells.size());
    std::vector<const LithotypeProperties*> lithotypes(phi.size());
    for_each_layer_cell(initial_grid_, [&](const ColCell& cell, double depth) {
        const auto& layer = cfg.layer(cell.layer_id);
        const double x = cell.center().x;
        phi[cell.index] = layer.phi_at(x, depth);
        lithotypes[cell.index] = &input_.lithotype(layer.lithotype_at(x));
        T[static_cast<Vector::Index>(cell.index)] = cfg.T_surf.at(x);
    });
    const auto boundaries =
        bc(cfg, cfg, 0.0, use_parameterized_flux_ ? &parameter_space_ : nullptr, cfg.age_ma);
    ColHeatSystem system(initial_grid_, phi, lithotypes, kBasinFluid, boundaries);
    system.set_flux_scheme(settings_.scheme);
    NonlinearSolver solver;
    solver.set_linear_solver(settings_.linear_solver);
    const auto result = solver.solve(system, T, settings_.nonlinear, settings_.linear);
    if (!result.converged())
        throw std::runtime_error("Initial stationary heat solve failed");
    if (jacobian) {
        Vector dq_deta;
        const Vector eta = parameter_space_.to_normalized(parameter_space_.values());
        (void)parameter_space_.at(cfg.age_ma, eta, dq_deta);
        SparseMatrix matrix(T.size(), T.size(), SparseStorageOrder::csc);
        const Vector basal = system.assemble_ddm_matrix(T, 0.0, matrix);
        const auto dT_dq = UmfpackLinearSolver::solve_columns(matrix, basal.native());
        Eigen::MatrixXd cell_S = dT_dq * dq_deta.native().transpose();
        if (face_jacobian)
            *face_jacobian = system.horizontal_face_sensitivities(T, cell_S, dq_deta);
        *jacobian = std::move(cell_S);
    }
    return T;
}

void BasinForwardSolver::set_parameters(BasinHeatParameterSpace space) {
    if (space.ages_ma().front() != configs().front().age_ma ||
        space.ages_ma().back() != configs().back().age_ma)
        throw std::invalid_argument("Basal flux knots must start and end at the basin ages");
    parameter_space_ = std::move(space);
    use_parameterized_flux_ = true;
}

void BasinForwardSolver::set_parameters(Vector parameters) {
    parameter_space_.set_values(std::move(parameters));
    use_parameterized_flux_ = true;
}

void BasinForwardSolver::set_intermediate_snapshots(bool enabled) noexcept {
    settings_.intermediate_snapshots = enabled;
}

std::span<const BasinConfig> BasinForwardSolver::configs() const noexcept {
    return input_.configs();
}

const ColGrid& BasinForwardSolver::final_grid() const noexcept {
    return final_grid_;
}

void BasinForwardSolver::final_temperature_nodes(std::span<const double> T,
                                                 std::span<double> node_T) const {
    final_temperature_nodes(T, node_T, nullptr);
}

void BasinForwardSolver::final_temperature_nodes(std::span<const double> T,
                                                 std::span<double> node_T, DdmFields* ddm) const {
    const auto& cfg = input_.configs().back();
    std::vector<double> lambda(T.size());
    for_each_layer_cell(final_grid_, [&](const ColCell& cell, double depth) {
        const auto& layer = cfg.layer(cell.layer_id);
        const double x = cell.center().x;
        lambda[cell.index] =
            thermal_properties(T[cell.index], layer.phi_at(x, depth),
                               input_.lithotype(layer.lithotype_at(x)), kBasinFluid)
                .lambda;
    });
    const auto boundaries =
        bc(cfg, cfg, 0.0, use_parameterized_flux_ ? &parameter_space_ : nullptr, cfg.age_ma);
    interpolate_horizontal_temperature(final_grid_, T, lambda, boundaries, node_T);
    if (ddm) {
        Vector dq;
        (void)parameter_space_.at(cfg.age_ma, ddm->eta, dq);
        ddm->face_S = horizontal_face_sensitivities(final_grid_, lambda, ddm->S, dq);
    }
}

Vector BasinForwardSolver::solve_at_wells(std::span<const BasinWellSample> samples,
                                          Eigen::MatrixXd* jacobian, const Vector* eta) const {
    DdmFields fields;
    if (jacobian) {
        fields.eta = eta ? *eta : parameter_space_.to_normalized(parameter_space_.values());
        jacobian->setZero(samples.size(), fields.eta.size());
    }
    auto* ddm = jacobian ? &fields : nullptr;
    Vector calculated(static_cast<Vector::Index>(samples.size()));
    (void)solve(
        [&](const BasinSnapshot& snapshot) {
            if (snapshot.epoch + 1 != configs().size())
                return;
            std::vector<double> node_T(final_grid_.cells.size() + final_grid_.columns.size());
            final_temperature_nodes(snapshot.T, node_T, ddm);
            Eigen::MatrixXd dRo;
            if (ddm)
                dRo = easy_ro_.reflectance_sensitivities(snapshot.I, snapshot.Ro, ddm->Z);
            for (std::size_t i = 0; i < samples.size(); ++i) {
                const auto& sample = samples[i];
                double value = sample.constant;
                for (std::size_t j = 0; j < sample.indices.size(); ++j) {
                    const auto source = sample.indices[j];
                    double field;
                    if (sample.kind == BasinObservable::reflectance) {
                        if (source >= snapshot.Ro.size())
                            throw std::out_of_range("Invalid well reflectance index");
                        field = snapshot.Ro[source];
                    } else {
                        field = source < snapshot.T.size() ? snapshot.T[source]
                                                           : node_T.at(source - snapshot.T.size());
                    }
                    if (ddm) {
                        const bool face = sample.kind == BasinObservable::temperature &&
                                          source >= snapshot.T.size();
                        const auto& derivatives = sample.kind == BasinObservable::reflectance
                                                      ? dRo
                                                      : (face ? ddm->face_S : ddm->S);
                        jacobian->row(i) +=
                            sample.weights[j] *
                            derivatives.row(face ? source - snapshot.T.size() : source);
                    }
                    value += sample.weights[j] * field;
                }
                calculated[static_cast<Vector::Index>(i)] = value;
            }
        },
        false, ddm);
    return calculated;
}

BasinForwardResult BasinForwardSolver::solve(const BasinObserver& observe) const {
    return solve(observe, false);
}

BasinForwardResult BasinForwardSolver::solve(const BasinObserver& observe,
                                             bool configuration_output, DdmFields* ddm) const {
    const auto solve_start = Clock::now();
    BasinForwardResult result;
    const auto& input = input_;
    const auto& settings = settings_;
    const auto configs = input.configs();
    const auto& easy_ro = easy_ro_;
    const auto initial_start = Clock::now();
    Vector T = solve_initial_temperature(ddm ? &ddm->S : nullptr);
    result.timings.initial_s = elapsed_seconds(initial_start);
    if (settings.intermediate_snapshots || configuration_output) {
        const auto initial_I = easy_ro.initial_state(initial_grid_.maturity_node_count);
        observe_state(initial_grid_, T, initial_I, easy_ro, BasinSnapshot::Kind::initial, 0, 0,
                      configs.front().age_ma, observe, result.timings);
    }
    auto grid_start = Clock::now();
    ColGrid grid = epochs_.front().initial_grid;
    const int born = epochs_.front().mesh.born_layer_id.value_or(-1);
    if (born != -1 || !same_transfer_grid(initial_grid_, grid)) {
        const auto P_T = temperature_transfer(initial_grid_, grid, born);
        T = Vector(P_T * T.native());
        if (ddm)
            ddm->S = (P_T * ddm->S).eval();
    }
    for (const auto& cell : grid.cells) {
        if (cell.layer_id == born)
            T[static_cast<Vector::Index>(cell.index)] = configs.front().T_surf.at(cell.center().x);
    }
    auto I = easy_ro.initial_state(epochs_.front().deposition_times.size());
    if (ddm)
        ddm->Z.setZero(I.size(), ddm->eta.size());
    result.epochs.reserve(configs.size() - 1);
    TimeIntegrator integrator;
    integrator.set_linear_solver(settings.linear_solver);
    integrator.set_preconditioner_reuse(settings.pc_reuse);
    result.timings.grid_s += elapsed_seconds(grid_start);

    for (std::size_t epoch_index = 0; epoch_index + 1 < configs.size(); ++epoch_index) {
        const auto epoch_start = Clock::now();
        grid_start = epoch_start;
        const auto& cfg0 = configs[epoch_index];
        const auto& cfg1 = configs[epoch_index + 1];
        BasinIntervalStats epoch_stats{
            .epoch = epoch_index, .age_begin_ma = cfg0.age_ma, .age_end_ma = cfg1.age_ma};
        const double duration = (cfg0.age_ma - cfg1.age_ma) * kSecondsPerMa;
        const double epoch_begin = (configs.front().age_ma - cfg0.age_ma) * kSecondsPerMa;
        const double dt_max = settings.dt_ma * kSecondsPerMa;
        const auto& epoch = epochs_[epoch_index];
        std::vector<double> face_T(grid.cells.size() + grid.columns.size());
        std::vector<double> node_T0(epoch.deposition_times.size()), node_T1(node_T0.size());
        Eigen::MatrixXd node_S0, node_S1;
        Vector dq;
        const auto copy_face_temperatures = [&](std::span<double> material_T) {
            for (const auto& cell : grid.cells) {
                const auto face = cell.index + cell.column;
                material_T[cell.maturity_top] = face_T[face];
                material_T[cell.maturity_top + 1] = face_T[face + 1];
            }
        };
        const auto lithotypes = epoch.materials(input);
        std::vector<double> phi(grid.cells.size());
        epoch.update_porosity(grid, 0.0, phi);
        const auto* flux = use_parameterized_flux_ ? &parameter_space_ : nullptr;
        {
            std::vector<double> lambda(grid.cells.size());
            for (const auto& cell : grid.cells)
                lambda[cell.index] = thermal_properties(T[cell.index], phi[cell.index],
                                                        *lithotypes[cell.index], kBasinFluid)
                                         .lambda;
            interpolate_horizontal_temperature(grid, values(T), lambda,
                                               bc(cfg0, cfg1, 0.0, flux, cfg0.age_ma), face_T);
            copy_face_temperatures(node_T0);
            if (ddm) {
                (void)flux->at(cfg0.age_ma, ddm->eta, dq);
                node_S0 = material_sensitivities(
                    grid, horizontal_face_sensitivities(grid, lambda, ddm->S, dq));
            }
        }
        double t = 0.0;
        integrator.set_initial_solution(t, T);
        if (epoch_index > 0 && settings.pc_reuse == PcReuse::manual) {
            integrator.reset_linear_solver();
        }
        epoch_stats.timings.grid_s += elapsed_seconds(grid_start);
        const auto& ages = parameter_space_.ages_ma();
        auto knot = std::upper_bound(ages.begin(), ages.end(), cfg0.age_ma, std::greater<>{});
        double age = cfg0.age_ma;
        while (t < duration) {
            const auto step_start = Clock::now();
            grid_start = step_start;
            const bool interior_knot = flux && knot != ages.end() && *knot > cfg1.age_ma;
            const double stop_age = interior_knot ? *knot : cfg1.age_ma;
            const auto [t_next, theta, end_age, at_stop] =
                next_basin_step(t, dt_max, cfg0.age_ma, cfg1.age_ma, duration, stop_age);
            if (interior_knot && at_stop)
                ++knot;
            const double dt = t_next - t;
            BasinIntervalStats step_stats{
                .epoch = epoch_index, .age_begin_ma = age, .age_end_ma = end_age};
            grid = epoch.mesh.at(theta);
            epoch.update_porosity(grid, theta, phi);
            const auto end_bc = bc(cfg0, cfg1, theta, flux, end_age);
            auto step_bc = end_bc;
            // Steps end at flux knots, so the midpoint gives the exact mean basal flux.
            step_bc.q_basal = basal_flux(cfg0, cfg1, std::midpoint(t / duration, theta), flux,
                                         std::midpoint(age, end_age));
            ColHeatSystem system(grid, phi, lithotypes, kBasinFluid, std::move(step_bc));
            system.set_flux_scheme(settings.scheme);
            step_stats.timings.grid_s += elapsed_seconds(grid_start);
            const TimeIntegrationResult step =
                advance_bdf1(integrator, system, t_next, dt, end_age, settings);
            step_stats.timings.nonlinear = step.timings;
            step_stats.timings.properties = system.property_timings();
            const Vector& T1 = integrator.current_snapshot().u;
            const auto easy_ro_start = Clock::now();
            system.interpolate_horizontal_temperature(T1, end_bc, face_T);
            copy_face_temperatures(node_T1);
            std::optional<EasyRoStepSensitivities> ro_ddm;
            if (ddm) {
                (void)flux->at(std::midpoint(age, end_age), ddm->eta, dq);
                ddm->S = advance_cell_sensitivities(system, T1, dt, ddm->S, dq);
                (void)flux->at(end_age, ddm->eta, dq);
                node_S1 = material_sensitivities(
                    grid, system.horizontal_face_sensitivities(T1, ddm->S, dq));
                ro_ddm.emplace(node_S0, node_S1, ddm->Z);
            }
            easy_ro.advance(node_T0, node_T1, epoch_begin + t, epoch_begin + t_next,
                            epoch.deposition_times, I, ro_ddm ? &*ro_ddm : nullptr);
            if (ddm)
                node_S0.swap(node_S1);
            node_T0.swap(node_T1);
            step_stats.timings.easy_ro_s += elapsed_seconds(easy_ro_start);
            step_stats.timesteps = step.accepted_steps;
            step_stats.nonlinear_iterations = step.nonlinear_iterations;
            step_stats.linear_iterations = step.linear_iterations;
            t = t_next;
            age = end_age;
            const bool at_event = t == duration;
            if (settings.intermediate_snapshots || (configuration_output && at_event))
                observe_state(
                    grid, T1, I, easy_ro,
                    at_event ? BasinSnapshot::Kind::before_event : BasinSnapshot::Kind::timestep,
                    epoch_index, epoch_stats.timesteps + step_stats.timesteps,
                    at_event ? cfg1.age_ma : step_stats.age_end_ma, observe, step_stats.timings);
            step_stats.timings.total_s = elapsed_seconds(step_start);
            epoch_stats.timesteps += step_stats.timesteps;
            epoch_stats.nonlinear_iterations += step_stats.nonlinear_iterations;
            epoch_stats.linear_iterations += step_stats.linear_iterations;
            epoch_stats.timings += step_stats.timings;
            result.steps.push_back(step_stats);
        }
        grid_start = Clock::now();
        const bool last = epoch_index + 2 == configs.size();
        int born = -1;
        if (!last)
            born = epochs_[epoch_index + 1].mesh.born_layer_id.value_or(-1);
        else if (settings.reset_final_maturity)
            born = epoch.mesh.born_layer_id.value_or(-1);
        ColGrid next_grid = last ? final_grid_ : epochs_[epoch_index + 1].initial_grid;
        epoch_stats.timings.grid_s += elapsed_seconds(grid_start);
        const auto transfer_start = Clock::now();
        const auto P_T = temperature_transfer(grid, next_grid, last ? -1 : born);
        const auto P_I = maturity_transfer(grid, next_grid, born);
        T = Vector(P_T * integrator.current_snapshot().u.native());
        I = apply_maturity_transfer(P_I, I, easy_ro.reaction_count());
        if (ddm) {
            ddm->S = (P_T * ddm->S).eval();
            ddm->Z = apply_maturity_transfer(P_I, ddm->Z, easy_ro.reaction_count());
        }
        if (!last && born != -1) {
            for (const auto& cell : next_grid.cells) {
                if (cell.layer_id == born)
                    T[static_cast<Vector::Index>(cell.index)] = cfg1.T_surf.at(cell.center().x);
            }
        }
        grid = std::move(next_grid);
        epoch_stats.timings.transfer_s += elapsed_seconds(transfer_start);
        if (settings.intermediate_snapshots || last)
            observe_state(grid, T, I, easy_ro, BasinSnapshot::Kind::after_event, epoch_index + 1, 0,
                          cfg1.age_ma, observe, epoch_stats.timings);
        epoch_stats.timings.total_s = elapsed_seconds(epoch_start);
        result.timesteps += epoch_stats.timesteps;
        result.nonlinear_iterations += epoch_stats.nonlinear_iterations;
        result.linear_iterations += epoch_stats.linear_iterations;
        result.timings += epoch_stats.timings;
        result.epochs.push_back(epoch_stats);
    }
    result.timings.total_s = elapsed_seconds(solve_start);
    return result;
}
