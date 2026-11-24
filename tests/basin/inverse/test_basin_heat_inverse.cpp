#include "basin/basin_snapshot.hpp"
#include "basin/inverse/basin_heat_inverse_solver.hpp"
#include "basin/io/basin_parameters.hpp"

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
    Vector u(flux.size());
    u.set_constant(0.9);
    const auto physical = space.to_physical(u);
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

TEST(BasinHeatObservationData, UsesCelsiusErrorBoundsAndSeparateObservables) {
    const auto path = small_problem();
    write_file(path / "observations.csv", "kind,well,x,z,value,error\n"
                                          "temperature,A,500,1000,20,0.4,0.4\n"
                                          "reflectance,A,500,1000,0.5,0.01\n");
    const auto data = BasinHeatObservationData::read(path / "observations.csv");
    const auto residuals = data.normalized_residuals(Vector{293.55, 0.49});
    ASSERT_EQ(data.size(), 2);
    EXPECT_NEAR(residuals[0], 1.0, 1e-12);
    EXPECT_NEAR(residuals[1], -1.0, 1e-12);
    std::filesystem::remove_all(path);
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
            const auto node = cell.index + cell.column;
            const double Ro = 0.5 * (snapshot.Ro[node] + snapshot.Ro[node + 1]);
            points.push_back({.kind = BasinObservable::reflectance,
                              .well = "synthetic",
                              .x = center.x,
                              .z = center.z,
                              .value = Ro,
                              .error = 0.02 * Ro});
        }
    });
    BasinHeatInverseSolver inverse(BasinHeatResidualEvaluator(
        std::move(forward), BasinHeatObservationData(std::move(points))));
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
            EXPECT_EQ(r.size(), inverse.residual_evaluator().observations().size());
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
