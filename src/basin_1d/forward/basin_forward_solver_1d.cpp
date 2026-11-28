#include "basin_1d/forward/basin_forward_solver_1d.hpp"

#include "basin/forward/basin_time_step.hpp"
#include "basin/mesh/vertical_grid.hpp"
#include "basin_1d/discretization/heat_system_1d.hpp"
#include "numerics/linear/tdma_linear_solver.hpp"
#include "numerics/time/time_integrator.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace {

SectionEpoch1D static_mesh(const SectionConfig1D& cfg, double dz) {
    return {
        cfg.contacts, cfg.contacts, vertical_epoch_rows(cfg.contacts, cfg.contacts, dz), {}, 0.};
}

struct Fields {
    Vector T;
    Eigen::MatrixXd S, Z;
    std::vector<double> I;
};

struct FaceFields {
    Vector T;
    Eigen::MatrixXd S;
};

FaceFields material_nodes(const SectionLayout& layout, const FaceFields& faces) {
    FaceFields nodes{Vector(layout.node_faces.size()),
                     Eigen::MatrixXd(layout.node_faces.size(), faces.S.cols())};
    for (std::size_t i = 0; i < layout.node_faces.size(); ++i) {
        nodes.T[i] = faces.T[layout.node_faces[i]];
        if (faces.S.cols())
            nodes.S.row(i) = faces.S.row(layout.node_faces[i]);
    }
    return nodes;
}
std::span<const double> values(const Vector& v) {
    return {v.native().data(), static_cast<std::size_t>(v.size())};
}

Vector sample(std::span<const BasinObservation> points, const SectionLayout& layout,
              std::span<const double> z, const SectionConfig1D& config, const Fields& state,
              const FaceFields& faces, const EasyRoModel& ro, Eigen::MatrixXd* J) {
    std::vector<double> reflectance(layout.node_faces.size());
    ro.reflectance(state.I, reflectance);
    Eigen::MatrixXd dRo;
    if (J) {
        dRo = ro.reflectance_sensitivities(state.I, reflectance, state.Z);
        J->resize(points.size(), state.S.cols());
    }
    Vector result(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double observed_depth = points[i].z;
        if (!(observed_depth >= config.contacts.front() &&
              observed_depth <= config.contacts.back()))
            throw std::out_of_range("Observation outside 1D section");
        const double depth = std::clamp(observed_depth, z.front(), z.back());
        const std::size_t cell =
            std::upper_bound(z.begin() + 1, z.end() - 1, depth) - z.begin() - 1;
        const double fraction = (depth - z[cell]) / (z[cell + 1] - z[cell]);
        if (points[i].kind == BasinObservable::reflectance) {
            const auto layer = std::prev(std::upper_bound(
                layout.layers.begin(), layout.layers.end(), cell,
                [](std::size_t c, const SectionLayerRange& l) { return c < l.cell; }));
            const auto node = layer->node + cell - layer->cell;
            result[i] = std::lerp(reflectance[node], reflectance[node + 1], fraction);
            if (J)
                J->row(i) = (1. - fraction) * dRo.row(node) + fraction * dRo.row(node + 1);
        } else {
            const auto face = fraction <= 0.5 ? cell : cell + 1;
            const double w = std::abs(2. * fraction - 1.);
            result[i] = std::lerp(state.T[cell], faces.T[face], w);
            if (J)
                J->row(i) = (1. - w) * state.S.row(cell) + w * faces.S.row(face);
        }
    }
    return result;
}
} // namespace

BasinForwardSolver1D::BasinForwardSolver1D(const BasinInput& input, double x,
                                           BasinForwardSettings settings)
    : section_(make_basin_section_1d(input.configs(), x, settings.dz, settings.dz_min)),
      settings_(settings), easy_ro_(input.easy_ro()) {
    if (!std::isfinite(settings.dt_ma) || settings.dt_ma <= 0 ||
        settings.nonlinear.method != NonlinearMethod::picard)
        throw std::invalid_argument("1D basin requires a positive timestep and Picard iterations");
    const auto& cfg = section_.configs;
    std::unordered_map<int, std::pair<double, double>> birth;
    const auto elapsed = [&](double age) { return (cfg.front().age_ma - age) * kSecondsPerMa; };
    for (std::size_t i = 0; i < cfg.size(); ++i)
        for (const auto& layer : cfg[i].layers)
            birth.try_emplace(layer.id, elapsed(cfg[i == 0 ? 0 : i - 1].age_ma),
                              elapsed(cfg[i].age_ma));

    auto append = [&](SectionEpoch1D mesh, const SectionConfig1D& first,
                      const SectionConfig1D& last, bool final) {
        SectionLayout layout(last.layers, mesh.rows);
        Stage stage{std::move(mesh),
                    std::move(layout),
                    first.age_ma,
                    last.age_ma,
                    first.T_surf,
                    last.T_surf,
                    {},
                    {},
                    {}};
        auto end_z = stage.mesh.faces_at(1.);
        for (std::size_t i = 0; i < last.layers.size(); ++i) {
            if (!stage.mesh.rows[i])
                continue;
            const auto& end = last.layers[i];
            const auto& start = end.id == stage.mesh.born_layer_id
                                    ? end
                                    : first.layers[i - stage.mesh.born_layer_id.has_value()];
            const auto& range = stage.layout.layers[stage.materials.size()];
            stage.materials.push_back({input.lithotype(start.lithotype), start, end});
            const auto [begin, finish] = birth.at(end.id);
            const double top = end_z[range.cell], height = end_z[range.cell + range.count] - top;
            for (std::size_t j = 0; j <= range.count; ++j)
                stage.deposition_times.push_back(
                    std::lerp(finish, begin, (end_z[range.cell + j] - top) / height));
        }
        if (!stages_.empty()) {
            const auto& previous = stages_.back();
            const auto reset = final && settings.reset_final_maturity ? previous.mesh.born_layer_id
                                                                      : stage.mesh.born_layer_id;
            stage.transfer =
                make_section_transfer(previous.layout, previous.mesh.faces_at(1.), stage.layout,
                                      stage.mesh.faces_at(0.), stage.mesh.born_layer_id, reset);
        }
        stages_.push_back(std::move(stage));
    };
    append(static_mesh(cfg.front(), settings.dz), cfg.front(), cfg.front(), false);
    for (std::size_t i = 0; i < section_.epochs.size(); ++i)
        append(section_.epochs[i], cfg[i], cfg[i + 1], false);
    append(static_mesh(cfg.back(), settings.dz), cfg.back(), cfg.back(), true);
}

void BasinForwardSolver1D::Stage::update_porosity(std::span<const double> z, double theta,
                                                  std::span<double> phi) const {
    for (std::size_t i = 0; i < layout.layers.size(); ++i) {
        const auto& m = materials[i];
        const double top = std::lerp(m.start.phi_top, m.end.phi_top, theta);
        const double bottom = std::lerp(m.start.phi_bottom, m.end.phi_bottom, theta);
        const auto& layer = layout.layers[i];
        const double height = z[layer.cell + layer.count] - z[layer.cell];
        for (std::size_t c = layer.cell; c < layer.cell + layer.count; ++c)
            phi[c] =
                std::lerp(top, bottom, (std::midpoint(z[c], z[c + 1]) - z[layer.cell]) / height);
    }
}

Vector BasinForwardSolver1D::solve_at_wells(std::span<const BasinObservation> observations,
                                            const BasinHeatParameterSpace& space, const Vector& eta,
                                            Eigen::MatrixXd* jacobian) const {
    const auto& knots = space.ages_ma();
    if (knots.front() != stages_.front().age0 || knots.back() != stages_.back().age1)
        throw std::invalid_argument("Flux knots must span the 1D basin history");
    BasinHeatParameterSpace history(knots, space.to_physical(eta));
    Vector dq;
    const auto flux = [&](double age) {
        return jacobian ? space.at(age, eta, dq) : history.at(age);
    };
    TimeIntegrator integrator;
    integrator.set_linear_solver({.method = LinearMethod::tdma});
    Fields state;
    for (std::size_t index = 0; index < stages_.size(); ++index) {
        const auto& stage = stages_[index];
        auto z = stage.mesh.faces_at(0.);
        std::vector<const LithotypeProperties*> rocks;
        for (std::size_t i = 0; i < stage.layout.layers.size(); ++i)
            rocks.insert(rocks.end(), stage.layout.layers[i].count, &stage.materials[i].rock);
        std::vector<double> phi(rocks.size());
        const auto system_at = [&](double theta, double q) {
            stage.update_porosity(z, theta, phi);
            return HeatSystem1D(z, phi, rocks, kBasinFluid,
                                std::lerp(stage.surface0, stage.surface1, theta), q);
        };
        const auto solve_sensitivities = [&](const HeatSystem1D& system, double alpha) {
            SparseMatrix A(state.T.size(), state.T.size(), MatrixStorageFormat::tridiagonal);
            auto rhs = system.assemble_ddm(state.T, alpha, state.S, dq, A);
            state.S = TdmaLinearSolver::solve_columns(A, std::move(rhs));
        };
        const auto face_values = [&](const HeatSystem1D& system, double q) {
            FaceFields faces;
            faces.T =
                system.face_temperatures(state.T, q, state.S, dq, jacobian ? &faces.S : nullptr);
            return faces;
        };

        double q = flux(stage.age0);
        auto initial = system_at(0., q);
        if (index == 0) {
            state.T.resize(rocks.size());
            state.T.set_constant(stage.surface0);
            if (!integrator.nonlinear_solver()
                     .solve(initial, state.T, settings_.nonlinear, settings_.linear)
                     .converged())
                throw std::runtime_error("Stationary 1D basin solve failed");
            if (jacobian)
                solve_sensitivities(initial, 0.);
            state.I = easy_ro_.initial_state(stage.layout.node_faces.size());
            if (jacobian)
                state.Z.setZero(state.I.size(), eta.size());
            continue;
        }
        const auto& transfer = stage.transfer;
        state.T = Vector(transfer.temperature * state.T.native());
        state.I = apply_maturity_transfer(transfer.maturity, state.I, easy_ro_.reaction_count());
        if (jacobian) {
            state.S = (transfer.temperature * state.S).eval();
            state.Z =
                apply_maturity_transfer(transfer.maturity, state.Z, easy_ro_.reaction_count());
        }
        if (stage.mesh.born_layer_id)
            state.T.native().head(stage.mesh.rows.front()).setConstant(stage.surface0);
        const auto faces = face_values(initial, q);
        if (index + 1 == stages_.size())
            return sample(observations, stage.layout, z, section_.configs.back(), state, faces,
                          easy_ro_, jacobian);
        auto previous = material_nodes(stage.layout, faces);
        const double duration = (stage.age0 - stage.age1) * kSecondsPerMa;
        const double epoch_start = (stages_.front().age0 - stage.age0) * kSecondsPerMa;
        double t = 0., age = stage.age0;
        integrator.set_initial_solution(t, state.T);
        auto knot = std::upper_bound(knots.begin(), knots.end(), age, std::greater<>{});
        while (t < duration) {
            const bool interior = knot != knots.end() && *knot > stage.age1;
            const double stop_age = interior ? *knot : stage.age1;
            const auto [next, theta, end_age, at_stop] = next_basin_step(
                t, settings_.dt_ma * kSecondsPerMa, stage.age0, stage.age1, duration, stop_age);
            if (interior && at_stop)
                ++knot;
            const double dt = next - t;
            z = stage.mesh.faces_at(theta);
            auto system = system_at(theta, flux(std::midpoint(age, end_age)));
            integrator.set_timestep(dt);
            if (!integrator.advance_to(system, next, settings_.nonlinear, settings_.linear)
                     .completed())
                throw std::runtime_error("1D basin solve failed at age " + std::to_string(end_age));
            state.T = integrator.current_snapshot().u;
            if (jacobian)
                solve_sensitivities(system, 1. / dt);
            q = flux(end_age);
            auto current = material_nodes(stage.layout, face_values(system, q));
            std::optional<EasyRoStepSensitivities> derivatives;
            if (jacobian)
                derivatives.emplace(previous.S, current.S, state.Z);
            easy_ro_.advance(values(previous.T), values(current.T), epoch_start + t,
                             epoch_start + next, stage.deposition_times, state.I,
                             derivatives ? &*derivatives : nullptr);
            previous = std::move(current);
            t = next;
            age = end_age;
        }
    }
    throw std::logic_error("Missing final 1D configuration");
}
