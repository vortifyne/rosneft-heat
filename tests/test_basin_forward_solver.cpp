#include "basin/basin_forward_solver.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string_view>

namespace {

void write_file(const std::filesystem::path& path, const std::string_view contents) {
    std::ofstream output(path);
    ASSERT_TRUE(output);
    output << contents;
}

class BasinForwardSolverTest : public testing::Test {
protected:
    void SetUp() override {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = std::filesystem::temp_directory_path() /
                     ("basin-forward-solver-test-" + std::to_string(suffix));
        input_ = directory_ / "input";
        std::filesystem::create_directories(input_);
        write_file(input_ / "configurations.csv", "age_ma\n1\n0\n");
        write_file(input_ / "layers.csv",
                   "age_ma,layer_id,layer_name,x,z_top,z_bottom,lithotype,porosity\n"
                   "1,1,layer,0,0,100,1,0.1\n"
                   "1,1,layer,100,0,100,1,0.1\n"
                   "0,1,layer,0,0,100,1,0.1\n"
                   "0,1,layer,100,0,100,1,0.1\n");
        write_file(input_ / "boundaries.csv", "age_ma,kind,x,value_si\n"
                                              "1,surface_temperature,0,300\n"
                                              "1,surface_temperature,100,300\n"
                                              "1,basal_heat_flux,0,0.05\n"
                                              "1,basal_heat_flux,100,0.05\n"
                                              "0,surface_temperature,0,300\n"
                                              "0,surface_temperature,100,300\n"
                                              "0,basal_heat_flux,0,0.05\n"
                                              "0,basal_heat_flux,100,0.05\n");
        write_file(input_ / "lithotypes.csv",
                   "code,name,solid_density,conductivity_20,specific_heat_20,heat_production\n"
                   "1,rock,2600,2.5,900,0.000001\n");
        write_file(input_ / "kinetics.csv", "preexponential,activation_energy,weight,gas_constant\n"
                                            "10000000000000,200000,1,8.314\n");
    }

    void TearDown() override {
        std::filesystem::remove_all(directory_);
    }

    std::filesystem::path directory_;
    std::filesystem::path input_;
};

TEST_F(BasinForwardSolverTest, ReusesPreparedProblemForDifferentBoundaryParameters) {
    const BasinForwardSolver solver(input_, {.cell_size = 100.0, .timestep_ma = 0.5});
    const BasinForwardParameters defaults = solver.default_parameters();
    BasinForwardParameters changed = defaults;
    for (double& temperature : changed.surface_temperature.front().values) {
        temperature += 5.0;
    }

    const BasinForwardResult first = solver.solve(defaults, directory_ / "first");
    const BasinForwardResult second = solver.solve(changed, directory_ / "second");

    EXPECT_EQ(first.configurations, 2);
    EXPECT_EQ(first.accepted_steps, 2);
    EXPECT_EQ(second.configurations, 2);
    EXPECT_EQ(second.accepted_steps, 2);
    EXPECT_TRUE(std::filesystem::is_regular_file(directory_ / "first" / "summary.csv"));
    EXPECT_TRUE(std::filesystem::is_regular_file(directory_ / "second" / "summary.csv"));
    EXPECT_TRUE(std::filesystem::is_regular_file(directory_ / "first" / "timings.csv"));
    EXPECT_GT(first.timings.linear_solve_seconds, 0.0);
    EXPECT_GT(first.timings.assembly_seconds, 0.0);
}

} // namespace
