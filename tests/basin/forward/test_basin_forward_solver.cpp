#include "basin/basin_snapshot.hpp"
#include "basin/forward/basin_forward_solver.hpp"
#include "basin/mesh/col_grid.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <vector>

namespace {

void write_file(const std::filesystem::path& path, const char* contents) {
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("Cannot write basin test input");
    output << contents;
}

void expect_timing_tree(const BasinTimings& t) {
    const auto& n = t.nonlinear;
    for (double seconds :
         {t.total_s, t.grid_s, t.transfer_s, n.total_s, n.assembly.residual_s, n.assembly.matrix_s,
          t.properties.residual_s, t.properties.matrix_s, n.linear_s, t.easy_ro_s, t.output_s}) {
        EXPECT_GE(seconds, 0.0);
    }
    EXPECT_LE(t.grid_s + t.transfer_s + n.total_s + t.easy_ro_s + t.output_s, t.total_s + 1e-10);
    EXPECT_LE(n.assembly.total_s() + n.linear_s, n.total_s + 1e-10);
    EXPECT_LE(t.properties.residual_s, n.assembly.residual_s + 1e-10);
    EXPECT_LE(t.properties.matrix_s, n.assembly.matrix_s + 1e-10);
}

} // namespace

TEST(BasinForwardSolver, PreservesUniformTemperatureAndResetsBornLayerHistory) {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory =
        std::filesystem::temp_directory_path() / ("column-basin-test-" + std::to_string(suffix));
    std::filesystem::create_directories(directory);
    write_file(directory / "configurations.csv", "age_ma\n1\n0\n");
    write_file(directory / "layers.csv",
               "age_ma,layer_id,layer_name,x,z_top,z_bottom,lithotype,porosity\n"
               "1,2,old,0,0,2,1,0.1\n1,2,old,1,0,2,2,0.1\n"
               "1,2,old,2,0,2,2,0.1\n"
               "0,1,new,0,0,0.5,1,0.1\n0,1,new,2,0,0.5,1,0.1\n"
               "0,2,old,0,0.5,2,1,0.1\n0,2,old,2,0.5,2,2,0.1\n");
    write_file(directory / "boundaries.csv",
               "age_ma,kind,x,value_si\n"
               "1,surface_temperature,0,300\n1,surface_temperature,2,300\n"
               "1,basal_heat_flux,0,0\n1,basal_heat_flux,2,0\n"
               "0,surface_temperature,0,300\n0,surface_temperature,2,300\n"
               "0,basal_heat_flux,0,0\n0,basal_heat_flux,2,0\n");
    write_file(directory / "lithotypes.csv",
               "code,name,solid_density,conductivity_20,specific_heat_20,heat_production\n"
               "1,rock,2600,2.5,900,0\n"
               "2,rock2,2600,2.5,900,0\n");
    write_file(directory / "kinetics.csv", "preexponential,activation_energy,weight,gas_constant\n"
                                           "1e-15,1,1,8.314\n");
    const BasinInput input = BasinInput::read(directory);
    bool saw_final = false;
    bool saw_initial_axis = false;
    std::vector<std::tuple<BasinSnapshot::Kind, std::size_t, int, double>> observations;
    const BasinForwardSolver solver(
        input, {.dx = 10.0, .dz = 1.0, .dz_min = 1.0e-4, .dt_ma = 0.5, .axis_rdp_tol = 5.0});
    const BasinForwardResult result = solver.solve([&](const BasinSnapshot& snapshot) {
        observations.emplace_back(snapshot.kind, snapshot.epoch, snapshot.step, snapshot.age_ma);
        if (snapshot.kind == BasinSnapshot::Kind::initial) {
            saw_initial_axis = true;
            EXPECT_EQ(snapshot.grid.x, (std::vector<double>{0.0, 0.5, 1.0, 2.0}));
        }
        if (snapshot.kind != BasinSnapshot::Kind::after_event)
            return;
        saw_final = true;
        EXPECT_EQ(snapshot.grid.x, (std::vector<double>{0.0, 1.0, 2.0}));
        ASSERT_EQ(snapshot.I.size(), snapshot.grid.cells.size());
        for (const auto& cell : snapshot.grid.cells) {
            EXPECT_NEAR(snapshot.T[cell.index], 300.0, 1.0e-8);
            if (cell.layer_id == 1) {
                EXPECT_DOUBLE_EQ(snapshot.I[cell.index], 0.0);
                EXPECT_NEAR(snapshot.Ro[cell.index], std::exp(-1.6), 1.0e-12);
            } else {
                EXPECT_GT(snapshot.I[cell.index], 0.0);
                EXPECT_GT(snapshot.Ro[cell.index], std::exp(-1.6));
            }
        }
    });
    EXPECT_TRUE(saw_final);
    EXPECT_TRUE(saw_initial_axis);
    EXPECT_EQ(observations, (std::vector<std::tuple<BasinSnapshot::Kind, std::size_t, int, double>>{
                                {BasinSnapshot::Kind::initial, 0, 0, 1.0},
                                {BasinSnapshot::Kind::timestep, 0, 1, 0.5},
                                {BasinSnapshot::Kind::before_event, 0, 2, 0.0},
                                {BasinSnapshot::Kind::after_event, 1, 0, 0.0}}));
    EXPECT_EQ(result.timesteps, 2);
    ASSERT_EQ(result.epochs.size(), 1U);
    ASSERT_EQ(result.steps.size(), 2U);
    EXPECT_EQ(result.epochs.front().timesteps, result.timesteps);
    EXPECT_EQ(result.epochs.front().nonlinear_iterations, result.nonlinear_iterations);
    EXPECT_EQ(result.epochs.front().linear_iterations, result.linear_iterations);
    expect_timing_tree(result.timings);
    expect_timing_tree(result.epochs.front().timings);
    int nonlinear_iterations = 0;
    int linear_iterations = 0;
    double step_seconds = 0.0;
    double age = 1.0;
    for (const auto& step : result.steps) {
        EXPECT_EQ(step.epoch, 0U);
        EXPECT_EQ(step.timesteps, 1);
        EXPECT_DOUBLE_EQ(step.age_begin_ma, age);
        age = step.age_end_ma;
        nonlinear_iterations += step.nonlinear_iterations;
        linear_iterations += step.linear_iterations;
        step_seconds += step.timings.total_s;
        expect_timing_tree(step.timings);
    }
    EXPECT_DOUBLE_EQ(age, 0.0);
    EXPECT_EQ(nonlinear_iterations, result.nonlinear_iterations);
    EXPECT_EQ(linear_iterations, result.linear_iterations);
    EXPECT_LE(step_seconds, result.epochs.front().timings.total_s);
    EXPECT_LE(result.epochs.front().timings.total_s, result.timings.total_s);
    const auto output_directory = directory / "output";
    const auto repeated = solver.solve(output_directory, [&](const BasinSnapshot& snapshot) {
        if (snapshot.kind == BasinSnapshot::Kind::initial) {
            for (const double I : snapshot.I) {
                EXPECT_DOUBLE_EQ(I, 0.0);
            }
        }
    });
    EXPECT_EQ(repeated.timesteps, result.timesteps);
    EXPECT_EQ(repeated.nonlinear_iterations, result.nonlinear_iterations);
    EXPECT_EQ(repeated.linear_iterations, result.linear_iterations);
    expect_timing_tree(repeated.timings);
    EXPECT_TRUE(std::filesystem::is_regular_file(output_directory / "summary.txt"));
    for (const auto* name : {"epochs.csv", "timesteps.csv"}) {
        std::ifstream statistics(output_directory / name);
        std::string line;
        ASSERT_TRUE(std::getline(statistics, line));
        EXPECT_TRUE(line.starts_with("index,epoch,age_begin_ma,age_end_ma,timesteps,"));
        std::size_t rows = 0;
        while (std::getline(statistics, line))
            ++rows;
        EXPECT_EQ(rows, std::string(name) == "epochs.csv" ? repeated.epochs.size()
                                                          : repeated.steps.size());
    }
    for (const auto* name : {"snapshot_e01_s0000.cgrd", "snapshot_e01_s0001.cgrd",
                             "snapshot_e01_s0002.cgrd", "snapshot_e02_s0000.cgrd"}) {
        EXPECT_GT(std::filesystem::file_size(output_directory / "snapshots" / name), 40U);
        EXPECT_FALSE(std::filesystem::exists(output_directory / name));
    }
    std::filesystem::remove_all(directory);
}
