#include "basin/forward/basin_forward_solver.hpp"
#include "basin/io/basin_forward_options.hpp"
#include "basin/io/basin_statistics.hpp"
#include "numerics/timing.hpp"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <petscsys.h>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {

class PetscSession {
public:
    explicit PetscSession(double& seconds) : seconds_(seconds) {
        const auto start = Clock::now();
        if (PetscInitializeNoArguments() != PETSC_SUCCESS) {
            throw std::runtime_error("Cannot initialize PETSc");
        }
        seconds_ += elapsed_seconds(start);
    }
    ~PetscSession() {
        const auto start = Clock::now();
        PetscFinalize();
        seconds_ += elapsed_seconds(start);
    }

private:
    double& seconds_;
};

std::string_view value(int argc, const char* const argv[], int& index) {
    if (++index >= argc)
        throw std::invalid_argument("Missing option value");
    return argv[index];
}

} // namespace

int main(int argc, const char* const argv[]) {
    const auto program_start = Clock::now();
    if (argc < 3) {
        std::cerr << "Usage: basin_forward_solver <input-dir> <output-dir> "
                     "[--dx m] [--dz m] [--dt-ma Ma] [--nonlinear-rtol tolerance] "
                     "[--nonlinear-atol tolerance] [--nonlinear-maxiter n] [--min-height m] "
                     "[--rdp-tol m] [--axis-rdp-tol m] "
                     "[--scheme linear|nmpfa] "
                     "[--linear-method lu|gmres|bicgstab] [--pc ilu|ilu2|gamg|boomeramg] "
                     "[--pc-reuse none|step|epoch] [--linear-rtol tolerance] [--linear-atol "
                     "tolerance] [--linear-maxiter n]\n";
        return EXIT_FAILURE;
    }
    try {
        double initialization_s = 0.0;
        double input_s = 0.0;
        BasinForwardResult result;
        const std::filesystem::path output_directory(argv[2]);
        {
            PetscSession petsc(initialization_s);
            BasinForwardSettings settings;
            for (int index = 3; index < argc; ++index) {
                const std::string_view option = argv[index];
                const std::string_view setting = value(argc, argv, index);
                set_basin_forward_option(settings, option, setting);
            }
            const auto input_start = Clock::now();
            auto input = BasinInput::read(argv[1]);
            input_s = elapsed_seconds(input_start);
            BasinForwardSolver solver(std::move(input), settings);
            result = solver.solve(output_directory);
            std::cout << "timesteps=" << result.timesteps
                      << " nonlinear_iterations=" << result.nonlinear_iterations
                      << " linear_iterations=" << result.linear_iterations << '\n';
        }
        const double program_s = elapsed_seconds(program_start);
        write_basin_statistics(std::cout, result, program_s, initialization_s, input_s);
        std::ofstream statistics(output_directory / "timings.txt");
        write_basin_statistics(statistics, result, program_s, initialization_s, input_s);
        statistics.close();
        if (!statistics)
            throw std::runtime_error("Cannot write program timing report");
    } catch (const std::exception& error) {
        std::cerr << "basin_forward_solver: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
