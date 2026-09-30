#include "forward/basin_forward_solver.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

int main(const int argc, const char* const argv[]) {
    if (argc < 3 || argc > 7) {
        std::cerr << "Usage: heat_forward <input-dir> <output-dir> [cell-size] [dt-ma] "
                     "[save-every] [fixed]\n";
        return EXIT_FAILURE;
    }
    try {
        BasinForwardOptions options;
        if (argc >= 4) {
            options.cell_size = std::stod(argv[3]);
        }
        if (argc >= 5) {
            options.timestep_ma = std::stod(argv[4]);
        }
        if (argc >= 6) {
            options.save_every = std::stoi(argv[5]);
        }
        if (argc == 7) {
            if (std::string(argv[6]) != "fixed") {
                throw std::invalid_argument("The only optional mode is 'fixed'");
            }
            options.fixed_mesh = true;
        }
        const BasinForwardResult result = run_basin_forward(argv[1], argv[2], options);
        std::cout << "accepted_steps=" << result.accepted_steps
                  << " nonlinear_iterations=" << result.nonlinear_iterations
                  << " linear_iterations=" << result.linear_iterations
                  << " final_age_ma=" << result.final_age_ma
                  << " transfer_energy_error=" << result.transfer_energy_error
                  << " wall_seconds=" << result.wall_seconds << '\n';
    } catch (const std::exception& error) {
        std::cerr << "heat_forward: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
