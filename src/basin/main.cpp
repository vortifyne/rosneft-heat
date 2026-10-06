#include "basin/basin_forward_solver.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
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
                     "  --dt-ma <million-years>\n";
        return EXIT_FAILURE;
    }
    try {
        BasinForwardSettings settings;
        for (int index = 3; index < argc; ++index) {
            const std::string_view argument = argv[index];
            if (argument == "--cell-size") {
                settings.cell_size =
                    std::stod(std::string(require_value(argc, argv, index, argument)));
            } else if (argument == "--dt-ma") {
                settings.timestep_ma =
                    std::stod(std::string(require_value(argc, argv, index, argument)));
            } else {
                throw std::invalid_argument("Unknown option: " + std::string(argument));
            }
        }
        const BasinForwardSolver solver(argv[1], settings);
        const BasinForwardResult result = solver.solve(solver.default_parameters(), argv[2]);
        std::cout << "accepted_steps=" << result.accepted_steps
                  << " nonlinear_iterations=" << result.nonlinear_iterations
                  << " nonlinear_iterations_per_step_max="
                  << result.maximum_nonlinear_iterations_per_step
                  << " linear_iterations=" << result.linear_iterations
                  << " topology_regularized_epochs=" << result.topology_regularized_epochs
                  << " configurations=" << result.configurations
                  << " final_age_ma=" << result.final_age_ma
                  << " global_energy_balance=" << result.global_energy_balance
                  << " cells_min=" << result.minimum_cells << " cells_max=" << result.maximum_cells
                  << " cell_diameter_max=" << result.maximum_cell_diameter
                  << " cell_thickness_min=" << result.minimum_cell_thickness
                  << " cell_elongation_max=" << result.maximum_cell_elongation
                  << " wall_seconds=" << result.wall_seconds << '\n';
    } catch (const std::exception& error) {
        std::cerr << "heat_forward: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
