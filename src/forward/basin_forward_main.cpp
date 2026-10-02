#include "forward/basin_forward_solver.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace {

std::string_view require_value(const int argc, const char* const argv[], int& index,
                               const std::string_view option) {
    if (++index >= argc) {
        throw std::invalid_argument("Missing value for " + std::string(option));
    }
    return argv[index];
}

} // namespace

int main(const int argc, const char* const argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: heat_forward <input-dir> <output-dir> [options]\n"
                     "Options:\n"
                     "  --cell-size <m>\n"
                     "  --dt-ma <million-years>\n"
                     "  --states-per-epoch <count>\n"
                     "  --thin-layer-cell-fraction <value>\n"
                     "  --max-configurations <count>\n"
                     "  --comparison-points <csv>\n"
                     "  --quad-dominant\n"
                     "  --layered-mesh\n"
                     "  --triangular-mesh\n"
                     "  --estimate-condition\n"
                     "  --fixed-mesh\n";
        return EXIT_FAILURE;
    }
    try {
        BasinForwardOptions options;
        for (int index = 3; index < argc; ++index) {
            const std::string_view argument = argv[index];
            if (argument == "--cell-size") {
                options.cell_size =
                    std::stod(std::string(require_value(argc, argv, index, argument)));
            } else if (argument == "--dt-ma") {
                options.timestep_ma =
                    std::stod(std::string(require_value(argc, argv, index, argument)));
            } else if (argument == "--states-per-epoch") {
                options.saved_states_per_epoch =
                    std::stoi(std::string(require_value(argc, argv, index, argument)));
            } else if (argument == "--thin-layer-cell-fraction") {
                options.thin_layer_cell_fraction =
                    std::stod(std::string(require_value(argc, argv, index, argument)));
            } else if (argument == "--max-configurations") {
                options.max_configurations = static_cast<std::size_t>(
                    std::stoull(std::string(require_value(argc, argv, index, argument))));
            } else if (argument == "--comparison-points") {
                options.comparison_points = std::string(require_value(argc, argv, index, argument));
            } else if (argument == "--fixed-mesh") {
                options.fixed_mesh = true;
            } else if (argument == "--quad-dominant") {
                options.quad_dominant = true;
                options.layered_mesh = false;
            } else if (argument == "--layered-mesh") {
                options.layered_mesh = true;
                options.quad_dominant = false;
            } else if (argument == "--triangular-mesh") {
                options.layered_mesh = false;
                options.quad_dominant = false;
            } else if (argument == "--estimate-condition") {
                options.estimate_condition = true;
            } else {
                throw std::invalid_argument("Unknown option: " + std::string(argument));
            }
        }
        const BasinForwardResult result = run_basin_forward(argv[1], argv[2], options);
        std::cout << "accepted_steps=" << result.accepted_steps
                  << " nonlinear_iterations=" << result.nonlinear_iterations
                  << " nonlinear_iterations_per_step_max="
                  << result.maximum_nonlinear_iterations_per_step
                  << " linear_iterations=" << result.linear_iterations
                  << " configurations=" << result.configurations
                  << " final_age_ma=" << result.final_age_ma
                  << " global_energy_balance=" << result.global_energy_balance
                  << " cells_min=" << result.minimum_cells << " cells_max=" << result.maximum_cells
                  << " cell_diameter_max=" << result.maximum_cell_diameter
                  << " condition_estimate_max=" << result.maximum_condition_estimate
                  << " wall_seconds=" << result.wall_seconds << '\n';
    } catch (const std::exception& error) {
        std::cerr << "heat_forward: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
