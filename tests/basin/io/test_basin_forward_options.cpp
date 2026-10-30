#include "basin/io/basin_forward_options.hpp"

#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>

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
