#include "basin/basin_forward_solver.hpp"

#include <algorithm>
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
    EXPECT_EQ(result.accepted_steps, 2);
    EXPECT_LT(result.maximum_cell_residual, 1.0e-8);
    EXPECT_LT(result.maximum_global_balance, 1.0e-8);
    const auto output_directory = directory / "output";
    const auto repeated = solver.solve(output_directory, [&](const BasinForwardSnapshot& snapshot) {
        if (snapshot.kind == BasinForwardSnapshot::Kind::initial) {
            for (const double maturity : snapshot.maturity_integrals) {
                EXPECT_DOUBLE_EQ(maturity, 0.0);
            }
        }
    });
    EXPECT_EQ(repeated.accepted_steps, result.accepted_steps);
    EXPECT_DOUBLE_EQ(repeated.maximum_cell_residual, result.maximum_cell_residual);
    EXPECT_DOUBLE_EQ(repeated.maximum_global_balance, result.maximum_global_balance);
    EXPECT_TRUE(std::filesystem::is_regular_file(output_directory / "summary.txt"));
    for (const auto* name : {"snapshot_0000.cgrid", "snapshot_0001.cgrid", "snapshot_0002.cgrid"}) {
        EXPECT_GT(std::filesystem::file_size(output_directory / name), 40U);
    }
    std::filesystem::remove_all(directory);
}

TEST(BasinForwardSolver, ExactZeroCollapseTransfersToNextEpoch) {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() /
                           ("column-basin-zero-test-" + std::to_string(suffix));
    std::filesystem::create_directories(directory);
    write_file(directory / "configurations.csv", "age_ma\n2\n1\n0\n");
    write_file(directory / "layers.csv",
               "age_ma,layer_id,layer_name,x,z_top,z_bottom,lithotype,porosity\n"
               "2,2,pinch,0,0,0.5,1,0.1\n2,2,pinch,2,0,0.5,1,0.1\n"
               "2,3,base,0,0.5,2,1,0.1\n2,3,base,2,0.5,2,1,0.1\n"
               "1,1,new,0,0,0.2,1,0.1\n1,1,new,2,0,0.2,1,0.1\n"
               "1,2,pinch,0,0.2,0.2,1,0.1\n1,2,pinch,2,0.2,0.2,1,0.1\n"
               "1,3,base,0,0.2,2,1,0.1\n1,3,base,2,0.2,2,1,0.1\n"
               "0,1,new,0,0,0.3,1,0.1\n0,1,new,2,0,0.3,1,0.1\n"
               "0,2,pinch,0,0.3,0.3,1,0.1\n0,2,pinch,2,0.3,0.3,1,0.1\n"
               "0,3,base,0,0.3,2,1,0.1\n0,3,base,2,0.3,2,1,0.1\n");
    write_file(directory / "boundaries.csv",
               "age_ma,kind,x,value_si\n"
               "2,surface_temperature,0,300\n2,surface_temperature,2,300\n"
               "2,basal_heat_flux,0,0\n2,basal_heat_flux,2,0\n"
               "1,surface_temperature,0,300\n1,surface_temperature,2,300\n"
               "1,basal_heat_flux,0,0\n1,basal_heat_flux,2,0\n"
               "0,surface_temperature,0,300\n0,surface_temperature,2,300\n"
               "0,basal_heat_flux,0,0\n0,basal_heat_flux,2,0\n");
    write_file(directory / "lithotypes.csv",
               "code,name,solid_density,conductivity_20,specific_heat_20,heat_production\n"
               "1,rock,2600,2.5,900,0\n");
    write_file(directory / "kinetics.csv", "preexponential,activation_energy,weight,gas_constant\n"
                                           "1e-15,1,1,8.314\n");
    const BasinInput input = BasinInput::read(directory);
    bool saw_collapse = false;
    bool saw_final = false;
    const BasinForwardSolver solver(input,
                                    {.horizontal_step = 1.0,
                                     .vertical_step = 0.5,
                                     .minimum_cell_height = 1.0e-4,
                                     .timestep_ma = 0.5,
                                     .rdp_tolerance = 0.0,
                                     .pinchout = BasinForwardSettings::Pinchout::exact_zero});
    const BasinForwardResult result = solver.solve([&](const BasinForwardSnapshot& snapshot) {
        if (snapshot.kind == BasinForwardSnapshot::Kind::before_event && snapshot.age_ma == 1.0) {
            saw_collapse = true;
            for (const auto& cell : snapshot.grid.cells) {
                EXPECT_EQ(cell.layer_id, 3);
                EXPECT_GT(cell.height(), 0.0);
            }
        }
        if (snapshot.kind == BasinForwardSnapshot::Kind::after_event && snapshot.age_ma == 0.0) {
            saw_final = true;
            for (const auto& cell : snapshot.grid.cells) {
                EXPECT_NEAR(snapshot.temperature[cell.index], 300.0, 1.0e-8);
                EXPECT_GT(snapshot.maturity_integrals[cell.index], 0.0);
                EXPECT_NE(cell.layer_id, 2);
            }
        }
    });
    EXPECT_TRUE(saw_collapse);
    EXPECT_TRUE(saw_final);
    EXPECT_EQ(result.accepted_steps, 4);
    EXPECT_LT(result.maximum_global_balance, 1.0e-8);
    std::filesystem::remove_all(directory);
}
