#include "basin/basin_snapshot.hpp"
#include "basin/inverse/basin_heat_inverse_solver.hpp"
#include "basin/io/basin_parameters.hpp"
#include "basin_1d/forward/basin_forward_solver_1d.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

void write_file(const std::filesystem::path& path, const char* text) {
    std::ofstream(path) << text;
}

std::filesystem::path small_problem() {
    const auto path = std::filesystem::temp_directory_path() /
                      ("basin-inverse-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
    write_file(path / "configurations.csv", "age_ma\n20\n10\n0\n");
    write_file(
        path / "layers.csv",
        "age_ma,layer_id,layer_name,x,z_top,z_bottom,lithotype,porosity_top,porosity_bottom\n"
        "20,1,rock,0,0,2000,1,0.1,0.1\n20,1,rock,2000,0,2000,1,0.1,0.1\n"
        "10,1,rock,0,0,2000,1,0.1,0.1\n10,1,rock,2000,0,2000,1,0.1,0.1\n"
        "0,1,rock,0,0,2000,1,0.1,0.1\n0,1,rock,2000,0,2000,1,0.1,0.1\n");
    write_file(path / "boundaries.csv",
               "age_ma,kind,x,value_si\n"
               "20,surface_temperature,0,283.15\n20,surface_temperature,2000,283.15\n"
               "10,surface_temperature,0,283.15\n10,surface_temperature,2000,283.15\n"
               "0,surface_temperature,0,283.15\n0,surface_temperature,2000,283.15\n");
    write_file(path / "lithotypes.csv",
               "code,name,solid_density,conductivity_20,specific_heat_20,heat_production\n"
               "1,rock,2600,2.5,900,0\n");
    write_file(path / "kinetics.csv", "preexponential,activation_energy,weight,gas_constant\n"
                                      "1e13,190000,0.4,8.314\n1e13,210000,0.6,8.314\n");
    return path;
}

BasinForwardSettings small_settings() {
    return {.dx = 1000.0, .dz = 200.0, .dt_ma = 2.5};
}

} // namespace

TEST(BasinHeatParameterSpace, InterpolatesAndPreservesMonotonicityAtArbitraryAges) {
    const Vector flux{0.1, 0.08, 0.08, 0.05};
    const BasinHeatParameterSpace space({20, 16, 6, 0}, flux);
    EXPECT_DOUBLE_EQ(space.at(20), 0.1);
    EXPECT_DOUBLE_EQ(space.at(18), 0.09);
    EXPECT_DOUBLE_EQ(space.at(10), 0.08);
    EXPECT_DOUBLE_EQ(space.at(3), 0.065);
    EXPECT_DOUBLE_EQ(space.at(0), 0.05);
    const Vector recovered = space.to_physical(space.to_normalized(flux));
    EXPECT_LT((recovered - flux).norm(), 1e-15);
    Vector eta(flux.size());
    eta.set_constant(0.9);
    const auto physical = space.to_physical(eta);
    for (Vector::Index i = 1; i < physical.size(); ++i) {
        EXPECT_LE(physical[i], physical[i - 1]);
        EXPECT_GE(physical[i], 0.05);
    }
    EXPECT_THROW(BasinHeatParameterSpace({20}, {0.08}), std::invalid_argument);
    EXPECT_THROW(BasinHeatParameterSpace({20, 0, 0}, {0.08, 0.07, 0.06}), std::invalid_argument);
    EXPECT_THROW(BasinHeatParameterSpace({20, 0}, {0.07, 0.08}), std::invalid_argument);
    EXPECT_THROW(BasinHeatParameterSpace({20, 0}, {0.101, 0.07}), std::invalid_argument);
    EXPECT_THROW(BasinHeatParameterSpace({20, 0}, {0.08, 0.049}), std::invalid_argument);
}

TEST(BasinHeatParameterSpace, FluxDerivativesMatchFiniteDifferencesInNormalizedParameters) {
    const BasinHeatParameterSpace space({20, 16, 6, 0}, {0.09, 0.08, 0.07, 0.06});
    const auto flux_at = [&](double age, const Vector& eta) {
        return BasinHeatParameterSpace(space.ages_ma(), space.to_physical(eta)).at(age);
    };
    for (const Vector& eta :
         {Vector{0.8, 0.7, 0.4, 0.3}, Vector{0.8, 0.0, 0.4, 1.0}, Vector{0.0, 0.7, 0.0, 0.3}}) {
        for (double age : {20.0, 16.0, 10.0, 0.0}) {
            Vector gradient;
            EXPECT_DOUBLE_EQ(space.at(age, eta, gradient), flux_at(age, eta));
            ASSERT_EQ(gradient.size(), eta.size());
            for (Vector::Index j = 0; j < eta.size(); ++j) {
                Vector lower = eta, upper = eta;
                lower[j] = std::max(0.0, eta[j] - 1e-5);
                upper[j] = std::min(1.0, eta[j] + 1e-5);
                const double difference =
                    (flux_at(age, upper) - flux_at(age, lower)) / (upper[j] - lower[j]);
                EXPECT_NEAR(gradient[j], difference, 1e-10) << "age=" << age << ", j=" << j;
            }
        }
    }
}

TEST(BasinHeatObservationData, UsesCelsiusErrorBoundsAndSeparateObservables) {
    const auto path = small_problem();
    write_file(path / "observations.csv", "kind,well,x,z,value,error\n"
                                          "temperature,A,500,1000,20,0.4,0.4\n"
                                          "reflectance,A,500,1000,0.5,0.01\n");
    const auto data = BasinHeatObservationData::read(path / "observations.csv");
    ASSERT_EQ(data.size(), 2);
    EXPECT_EQ(data.points()[0].kind, BasinObservable::temperature);
    EXPECT_DOUBLE_EQ(data.points()[0].value, 293.15);
    EXPECT_DOUBLE_EQ(data.points()[0].error, 0.4);
    EXPECT_EQ(data.points()[1].kind, BasinObservable::reflectance);
    EXPECT_DOUBLE_EQ(data.points()[1].value, 0.5);
    EXPECT_DOUBLE_EQ(data.points()[1].error, 0.01);
    std::filesystem::remove_all(path);
}

TEST(BasinForwardSolver, StationarySensitivityUsesOnlyFirstFluxKnot) {
    const auto path = small_problem();
    BasinForwardSolver solver(BasinInput::read(path), small_settings());
    std::filesystem::remove_all(path);
    solver.set_parameters(Vector{0.09, 0.08, 0.06});
    Eigen::MatrixXd S, face_S;
    const Vector T = solver.solve_initial_temperature(&S, &face_S);
    ASSERT_EQ(S.rows(), T.size());
    ASSERT_EQ(S.cols(), 3);
    EXPECT_TRUE(S.rightCols(2).isZero(0.0));
    ASSERT_EQ(face_S.cols(), 3);
    EXPECT_TRUE(face_S.rightCols(2).isZero(0.0));
    // With uniform surface temperature and no sources, frozen K gives K*(T-Tsurf) = q*basal.
    for (Vector::Index i = 0; i < T.size(); ++i)
        EXPECT_NEAR(S(i, 0), (T[i] - 283.15) * 0.05 / 0.09, 1e-4);
}

TEST(BasinForwardSolver, AdvancesTemperatureAndOptionalCellJacobianWithinEpoch) {
    const auto path = small_problem();
    BasinForwardSolver solver(BasinInput::read(path), small_settings());
    std::filesystem::remove_all(path);
    const Vector eta{0.9, 0.7, 0.8};
    Vector plain(solver.epoch_initial_grid(0).cells.size());
    plain.set_constant(300.0);
    Vector T = plain;
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(T.size(), eta.size());
    S.col(2).setOnes(); // This column can only propagate from the previous step.
    for (const double age : {20.0, 17.5}) {
        solver.advance_temperature_step(0, age, age - 2.5, eta, plain);
        Eigen::MatrixXd face_S;
        solver.advance_temperature_step(0, age, age - 2.5, eta, T, &S, &face_S);
        const auto& grid = solver.epoch_initial_grid(0); // This fixture has fixed geometry.
        ASSERT_EQ(face_S.rows(), grid.cells.size() + grid.columns.size());
        ASSERT_EQ(face_S.cols(), eta.size());
        Vector dq;
        (void)solver.parameter_space().at(age - 2.5, eta, dq);
        for (const auto& column : grid.columns) {
            const auto top = column.front() + grid.cells[column.front()].column;
            EXPECT_TRUE(face_S.row(top).isZero(0.0));
            const auto bottom = column.back();
            const auto delta = face_S.row(top + column.size()) - S.row(bottom);
            EXPECT_GT(delta[0], 0.0);
            EXPECT_NEAR(delta[1] * dq[0], delta[0] * dq[1], 1e-12);
            EXPECT_DOUBLE_EQ(delta[2], 0.0);
            for (std::size_t i = 1; i < column.size(); ++i) {
                const double a = S(column[i - 1], 0), b = S(column[i], 0);
                EXPECT_GE(face_S(top + i, 0), std::min(a, b));
                EXPECT_LE(face_S(top + i, 0), std::max(a, b));
            }
        }
        EXPECT_LT((T - plain).infinity_norm(), 1e-10);
        EXPECT_GT(S.col(0).norm(), 0.0);
        EXPECT_GT(S.col(1).norm(), 0.0);
        EXPECT_GT(S.col(2).minCoeff(), 0.0);
        EXPECT_LT(S.col(2).maxCoeff(), 1.0);
    }
}

TEST(BasinForwardSolver, ReusesPreparationAndResetsStateAfterUpdatingParameters) {
    const auto path = small_problem();
    BasinForwardSolver solver(BasinInput::read(path), small_settings());
    const Vector original{0.09, 0.08, 0.06};
    solver.set_parameters(original);
    Vector initial;
    Vector first_temperature;
    std::vector<double> first_reflectance;
    const auto result = solver.solve([&](const BasinSnapshot& snapshot) {
        if (snapshot.kind == BasinSnapshot::Kind::initial) {
            initial = Vector(static_cast<Vector::Index>(snapshot.T.size()));
            for (const auto& cell : snapshot.grid.cells) {
                initial[static_cast<Vector::Index>(cell.index)] = snapshot.T[cell.index];
                EXPECT_GT(snapshot.T[cell.index], 283.15);
            }
        }
        if (snapshot.age_ma == 0 && snapshot.kind == BasinSnapshot::Kind::after_event) {
            first_temperature = Vector(static_cast<Vector::Index>(snapshot.T.size()));
            for (Vector::Index i = 0; i < first_temperature.size(); ++i)
                first_temperature[i] = snapshot.T[static_cast<std::size_t>(i)];
            first_reflectance.assign(snapshot.Ro.begin(), snapshot.Ro.end());
        }
    });
    EXPECT_EQ(result.timesteps, 8);
    solver.set_parameters(Vector{0.1, 0.09, 0.08});
    (void)solver.solve([&](const BasinSnapshot& snapshot) {
        if (snapshot.kind == BasinSnapshot::Kind::initial) {
            for (Vector::Index i = 0; i < initial.size(); ++i)
                EXPECT_GT(snapshot.T[static_cast<std::size_t>(i)], initial[i]);
            for (double integral : snapshot.I)
                EXPECT_DOUBLE_EQ(integral, 0.0);
        }
    });
    solver.set_parameters(original);
    solver.set_intermediate_snapshots(false);
    int callbacks = 0;
    (void)solver.solve([&](const BasinSnapshot& snapshot) {
        ++callbacks;
        EXPECT_DOUBLE_EQ(snapshot.age_ma, 0.0);
        for (Vector::Index i = 0; i < first_temperature.size(); ++i) {
            EXPECT_DOUBLE_EQ(snapshot.T[static_cast<std::size_t>(i)], first_temperature[i]);
        }
        ASSERT_EQ(snapshot.Ro.size(), first_reflectance.size());
        for (std::size_t i = 0; i < first_reflectance.size(); ++i)
            EXPECT_DOUBLE_EQ(snapshot.Ro[i], first_reflectance[i]);
    });
    EXPECT_EQ(callbacks, 1);
    std::filesystem::remove_all(path);
}

TEST(BasinForwardSolver, StepsToFluxKnotsLoadedFromCsv) {
    const auto path = small_problem();
    const auto csv = path / "parameters.csv";
    write_file(csv, "age_ma,basal_heat_flux_mw_m2\n20,90\n16,82\n10,80\n0,60\n");
    BasinForwardSolver solver(BasinInput::read(path), small_settings());
    EXPECT_THROW(solver.set_parameters(BasinHeatParameterSpace({19, 0}, {0.09, 0.06})),
                 std::invalid_argument);
    solver.set_parameters(read_basin_parameters(csv));
    write_basin_parameters(csv, solver.parameter_space());
    const auto saved = read_basin_parameters(csv);
    EXPECT_EQ(saved.ages_ma().size(), 4u);
    EXPECT_DOUBLE_EQ(saved.at(16), 0.082);
    const auto result = solver.solve();
    std::vector<double> ends;
    for (const auto& step : result.steps) {
        EXPECT_GT(step.age_begin_ma, step.age_end_ma);
        ends.push_back(step.age_end_ma);
    }
    EXPECT_EQ(ends, (std::vector<double>{17.5, 16, 13.5, 11, 10, 7.5, 5, 2.5, 0}));
    std::filesystem::remove_all(path);
}

TEST(BasinHeatInverseSolver, RecoversKnownFluxHistoryFromSyntheticTemperatureAndReflectance) {
    const auto path = small_problem();
    BasinForwardSolver forward(BasinInput::read(path), small_settings());
    const Vector known{0.09, 0.06};
    forward.set_parameters(BasinHeatParameterSpace({20, 0}, known));
    forward.set_intermediate_snapshots(false);
    std::vector<BasinObservation> points;
    (void)forward.solve([&](const BasinSnapshot& snapshot) {
        for (const auto index : snapshot.grid.top_cells) {
            const auto& cell = snapshot.grid.cells[index];
            points.push_back({.kind = BasinObservable::temperature,
                              .well = "surface",
                              .x = cell.center().x,
                              .z = cell.z_top,
                              .value = 283.15,
                              .error = 0.2});
        }
        for (const auto& cell : snapshot.grid.cells) {
            const auto center = cell.center();
            points.push_back({.kind = BasinObservable::temperature,
                              .well = "synthetic",
                              .x = center.x,
                              .z = center.z,
                              .value = snapshot.T[cell.index],
                              .error = 0.02 * (snapshot.T[cell.index] - 273.15)});
            const auto node = cell.maturity_top;
            const double Ro = 0.5 * (snapshot.Ro[node] + snapshot.Ro[node + 1]);
            points.push_back({.kind = BasinObservable::reflectance,
                              .well = "synthetic",
                              .x = center.x,
                              .z = center.z,
                              .value = Ro,
                              .error = 0.02 * Ro});
        }
    });
    BasinHeatInverseSolver inverse(
        BasinObjective(std::move(forward), BasinHeatObservationData(std::move(points))));
    BasinHeatInverseSolveRequest request{.initial_parameters = Vector{0.075, 0.075}};
    request.optimization = {
        .gatol = 1e-10, .grtol = 1e-10, .gttol = 1e-10, .maxiter = 100, .maxfev = 400};
    int evaluations = 0;
    int last_iteration = 0;
    const Vector found =
        inverse.solve(request, [&](int iteration, int evaluation, const Vector&, const Vector& r) {
            EXPECT_GE(iteration, last_iteration);
            last_iteration = iteration;
            EXPECT_EQ(evaluation, ++evaluations);
            EXPECT_EQ(r.size(), inverse.objective().observations().size());
        });
    const auto& result = inverse.optimization_result();
    EXPECT_LT((found - known).infinity_norm(), 5e-4);
    EXPECT_LT(result.residuals.infinity_norm(), 1e-3);
    EXPECT_EQ(evaluations, result.function_evaluations);
    EXPECT_GT(last_iteration, 0);
    EXPECT_LE(last_iteration, result.iterations);
    EXPECT_LE(evaluations, 401);
    for (Vector::Index i = 1; i < found.size(); ++i)
        EXPECT_LE(found[i], found[i - 1]);
    std::filesystem::remove_all(path);
}

TEST(BasinObjective, SamplesTheDeeperSideOfAContactAndStaysWithinPinchingLayers) {
    const auto path = small_problem();
    write_file(
        path / "layers.csv",
        "age_ma,layer_id,layer_name,x,z_top,z_bottom,lithotype,porosity_top,porosity_bottom\n"
        "20,1,rock,0,0,3600,1,0.1,0.1\n20,1,rock,2000,0,3600,1,0.1,0.1\n"
        "10,1,rock,0,0,3600,1,0.1,0.1\n10,1,rock,2000,0,3600,1,0.1,0.1\n"
        "0,2,cap,0,0,1600,1,0.1,0.1\n0,2,cap,1000,0,0,1,0.1,0.1\n"
        "0,1,rock,0,1600,3600,1,0.1,0.1\n0,1,rock,1000,0,3600,1,0.1,0.1\n"
        "0,1,rock,2000,0,3600,1,0.1,0.1\n");
    write_file(path / "boundaries.csv",
               "age_ma,kind,x,value_si\n"
               "20,surface_temperature,0,330\n20,surface_temperature,2000,330\n"
               "10,surface_temperature,0,330\n10,surface_temperature,2000,330\n"
               "0,surface_temperature,0,330\n0,surface_temperature,2000,330\n");
    write_file(path / "kinetics.csv", "preexponential,activation_energy,weight,gas_constant\n"
                                      "1e13,190000,1.0,8.314\n");
    auto settings = small_settings();
    settings.dz = 400.0;
    BasinForwardSolver forward(BasinInput::read(path), settings);
    const Vector flux{0.09, 0.08, 0.06};
    forward.set_parameters(flux);
    forward.set_intermediate_snapshots(false);
    std::vector<BasinObservation> points;
    (void)forward.solve([&](const BasinSnapshot& snapshot) {
        const auto& grid = snapshot.grid;
        const auto cap_top = grid.cells[grid.columns[0][0]].maturity_top;
        const auto cap_second = grid.cells[grid.columns[0][1]].maturity_top;
        const auto rock_left = grid.cells[grid.columns[0][2]].maturity_top;
        const auto rock_right = grid.cells[grid.columns[1][0]].maturity_top;
        EXPECT_EQ(grid.cells[grid.columns[0][2]].layer_id, 1);
        EXPECT_NE(snapshot.Ro[cap_second + 1], snapshot.Ro[rock_left]);
        points = {{BasinObservable::reflectance, "contact", 750.0, 400.0,
                   0.75 * snapshot.Ro[rock_left] + 0.25 * snapshot.Ro[rock_right], 1.0},
                  {BasinObservable::reflectance, "cap", 750.0, 100.0,
                   0.5 * (snapshot.Ro[cap_top] + snapshot.Ro[cap_top + 1]), 1.0}};
    });
    BasinObjective objective(std::move(forward), BasinHeatObservationData(std::move(points)));
    Eigen::MatrixXd J;
    const auto [residual, within_errors] =
        objective.evaluate(objective.parameter_space().to_normalized(flux), 1.0, &J);
    ASSERT_EQ(J.rows(), 2);
    ASSERT_EQ(J.cols(), flux.size());
    EXPECT_TRUE(J.allFinite());
    EXPECT_GT(J.norm(), 0.0);
    EXPECT_TRUE(within_errors);
    EXPECT_NEAR(residual[0], 0.0, 1e-12);
    EXPECT_NEAR(residual[1], 0.0, 1e-12);
    std::filesystem::remove_all(path);
}

TEST(BasinForwardSolver1D, MatchesLayeredBasinAtWellsAcrossDepositionAndFluxKnot) {
    const auto path = small_problem();
    write_file(
        path / "layers.csv",
        "age_ma,layer_id,layer_name,x,z_top,z_bottom,lithotype,porosity_top,porosity_bottom\n"
        "20,1,old,0,0,2000,1,0.2,0.1\n20,1,old,2000,0,2000,1,0.2,0.1\n"
        "10,2,new,0,0,500,1,0.3,0.2\n10,2,new,2000,0,500,1,0.3,0.2\n"
        "10,1,old,0,500,2500,1,0.2,0.1\n10,1,old,2000,500,2500,1,0.2,0.1\n"
        "0,2,new,0,0,350,1,0.2,0.1\n0,2,new,2000,0,350,1,0.2,0.1\n"
        "0,1,old,0,350,2150,1,0.1,0.05\n0,1,old,2000,350,2150,1,0.1,0.05\n");
    const auto input = BasinInput::read(path);
    std::filesystem::remove_all(path);
    const BasinHeatParameterSpace space({20, 14, 0}, {0.09, 0.08, 0.06});
    const auto eta = space.to_normalized(space.values());
    const BasinForwardSolver1D section(input, 500, small_settings());
    BasinForwardSolver forward(input, small_settings());
    forward.set_parameters(space);
    using enum BasinObservable;
    const std::vector<BasinObservation> points{{temperature, "A", 500, 0, 0, 1},
                                               {temperature, "A", 500, 2150, 0, 1},
                                               {reflectance, "A", 500, 310, 0, 1},
                                               {reflectance, "A", 500, 350, 0, 1},
                                               {reflectance, "A", 500, 1800, 0, 1}};
    BasinObjective reference(std::move(forward), BasinHeatObservationData(points));
    Eigen::MatrixXd J, expected_J;
    const auto expected = reference.evaluate(eta, 1., &expected_J).residuals;
    const auto actual = section.solve_at_wells(points, space, eta, &J);
    EXPECT_LT((section.solve_at_wells(points, space, eta) - actual).infinity_norm(), 1e-9);
    for (Eigen::Index i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1.e-7 * std::max(1., std::abs(expected[i])));
        EXPECT_LT((J.row(i) - expected_J.row(i)).norm(),
                  1.e-6 * std::max(1., expected_J.row(i).norm()));
    }
}
