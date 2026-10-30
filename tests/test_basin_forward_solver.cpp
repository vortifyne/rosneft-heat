#include "basin/basin_snapshot.hpp"
#include "basin/forward/basin_forward_solver.hpp"
#include "basin/io/basin_forward_options.hpp"
#include "basin/mesh/nonmatching_column_grid.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
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
    const BasinForwardSolver solver(input, {.horizontal_step = 10.0,
                                            .vertical_step = 1.0,
                                            .minimum_cell_height = 1.0e-4,
                                            .timestep_ma = 0.5,
                                            .axis_rdp_tolerance = 5.0});
    const BasinForwardResult result = solver.solve([&](const BasinForwardSnapshot& snapshot) {
        if (snapshot.kind == BasinForwardSnapshot::Kind::initial) {
            saw_initial_axis = true;
            EXPECT_EQ(snapshot.grid.x, (std::vector<double>{0.0, 0.5, 1.0, 2.0}));
        }
        if (snapshot.kind != BasinForwardSnapshot::Kind::after_event)
            return;
        saw_final = true;
        EXPECT_EQ(snapshot.grid.x, (std::vector<double>{0.0, 1.0, 2.0}));
        ASSERT_EQ(snapshot.maturity_integrals.size(), snapshot.grid.cells.size());
        for (const auto& cell : snapshot.grid.cells) {
            EXPECT_NEAR(snapshot.temperature[cell.index], 300.0, 1.0e-8);
            if (cell.layer_id == 1) {
                EXPECT_DOUBLE_EQ(snapshot.maturity_integrals[cell.index], 0.0);
                EXPECT_NEAR(snapshot.reflectance_percent[cell.index], std::exp(-1.6), 1.0e-12);
            } else {
                EXPECT_GT(snapshot.maturity_integrals[cell.index], 0.0);
                EXPECT_GT(snapshot.reflectance_percent[cell.index], std::exp(-1.6));
            }
        }
    });
    EXPECT_TRUE(saw_final);
    EXPECT_TRUE(saw_initial_axis);
    EXPECT_EQ(result.timesteps, 2);
    const auto output_directory = directory / "output";
    const auto repeated = solver.solve(output_directory, [&](const BasinForwardSnapshot& snapshot) {
        if (snapshot.kind == BasinForwardSnapshot::Kind::initial) {
            for (const double maturity : snapshot.maturity_integrals) {
                EXPECT_DOUBLE_EQ(maturity, 0.0);
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

TEST(BasinForwardOptions, KeepsIndependentChoicesAndSerializesToleranceGroups) {
    BasinForwardSettings settings;
    set_basin_forward_option(settings, "--linear-method", "bicgstab");
    set_basin_forward_option(settings, "--pc", "ilu2");
    set_basin_forward_option(settings, "--pc-reuse", "epoch");
    set_basin_forward_option(settings, "--linear-rtol", "1e-9");
    set_basin_forward_option(settings, "--linear-atol", "1e-14");
    set_basin_forward_option(settings, "--linear-maxiter", "42");
    set_basin_forward_option(settings, "--nonlinear-rtol", "1e-7");
    set_basin_forward_option(settings, "--nonlinear-atol", "1e-10");
    set_basin_forward_option(settings, "--nonlinear-maxiter", "30");
    EXPECT_EQ(settings.linear_solver.method, LinearMethod::bicgstab);
    EXPECT_EQ(settings.linear_solver.preconditioner, Preconditioner::ilu2);
    EXPECT_EQ(settings.pc_reuse, PreconditionerReuse::manual);
    EXPECT_EQ(settings.linear.max_iterations, 42);
    EXPECT_DOUBLE_EQ(settings.linear.relative_tolerance, 1e-9);
    EXPECT_DOUBLE_EQ(settings.linear.absolute_tolerance, 1e-14);
    EXPECT_EQ(settings.nonlinear.max_iterations, 30);
    EXPECT_DOUBLE_EQ(settings.nonlinear.relative_tolerance, 1e-7);
    EXPECT_DOUBLE_EQ(settings.nonlinear.absolute_tolerance, 1e-10);
    std::ostringstream description;
    write_basin_forward_settings(description, settings);
    EXPECT_NE(description.str().find("linear_method=bicgstab\npc=ilu2\npc_reuse=epoch\n"),
              std::string::npos);
    EXPECT_THROW(set_basin_forward_option(settings, "--linear-solver", "gamg"),
                 std::invalid_argument);
    EXPECT_THROW(set_basin_forward_option(settings, "--linear-maxiter", "42junk"),
                 std::invalid_argument);
    EXPECT_THROW(set_basin_forward_option(settings, "--nonlinear-atol", "nan"),
                 std::invalid_argument);
}
