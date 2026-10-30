#include "basin/basin_snapshot.hpp"
#include "basin/forward/basin_forward_solver.hpp"
#include "basin/mesh/col_grid.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

namespace {

void write_file(const std::filesystem::path& path, const char* contents) {
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("Cannot write basin test input");
    output << contents;
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
    const BasinForwardSolver solver(
        input, {.dx = 10.0, .dz = 1.0, .dz_min = 1.0e-4, .dt_ma = 0.5, .axis_rdp_tol = 5.0});
    const BasinForwardResult result = solver.solve([&](const BasinSnapshot& snapshot) {
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
    EXPECT_EQ(result.timesteps, 2);
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
    EXPECT_TRUE(std::filesystem::is_regular_file(output_directory / "summary.txt"));
    for (const auto* name : {"snapshot_0000.cgrd", "snapshot_0001.cgrd", "snapshot_0002.cgrd"}) {
        EXPECT_GT(std::filesystem::file_size(output_directory / name), 40U);
    }
    std::filesystem::remove_all(directory);
}
