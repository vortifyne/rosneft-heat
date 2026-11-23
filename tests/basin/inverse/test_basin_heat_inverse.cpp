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

TEST(BasinHeatParameterSpace, RepresentsAllEpochBoundaryFluxesAndPreservesMonotonicity) {
    BasinHeatParameterSpace space(17);
    Vector flux(17);
    for (Vector::Index i = 0; i < flux.size(); ++i)
        flux[i] = 0.1 - 0.05 * static_cast<double>(i) / 16.0;
    const Vector recovered = space.to_physical(space.to_normalized(flux));
    EXPECT_LT((recovered - flux).norm(), 1e-15);
    Vector u(17);
    u.set_constant(0.9);
    const auto physical = space.to_physical(u);
    for (Vector::Index i = 1; i < physical.size(); ++i) {
        EXPECT_LE(physical[i], physical[i - 1]);
        EXPECT_GE(physical[i], 0.05);
    }
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
    (void)solver.solve([&](const BasinSnapshot& snapshot) {
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
    write_basin_parameters(path / "parameters.csv", solver.configs(), original);
    EXPECT_LT((read_basin_parameters(path / "parameters.csv", solver.configs()) - original).norm(),
              1e-16);
    std::filesystem::remove_all(path);
}

TEST(BasinHeatInverseSolver, RecoversKnownFluxHistoryFromSyntheticTemperatureAndReflectance) {
    const auto path = small_problem();
    BasinForwardSolver forward(BasinInput::read(path), small_settings());
    const Vector known{0.09, 0.08, 0.06};
    forward.set_parameters(known);
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
    BasinHeatInverseSolveRequest request{.initial_parameters = Vector{0.075, 0.075, 0.075}};
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
