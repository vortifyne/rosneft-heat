#include "basin/basin_snapshot.hpp"
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
#include <vector>

namespace {

class PetscSession {
public:
    PetscSession(int& argc, char**& argv, double& seconds) : seconds_(seconds) {
        const auto start = Clock::now();
        if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS) {
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

int main(int argc, char* argv[]) {
    const auto program_start = Clock::now();
    if (argc < 3) {
        std::cerr << "Usage: basin_forward_solver <input-dir> <output-dir> "
                     "[--dx m] [--dz m] [--dt-ma Ma] [--nonlinear-rtol tolerance] "
                     "[--nonlinear-atol tolerance] [--nonlinear-maxiter n] [--min-height m] "
                     "[--rdp-tol m] [--axis-rdp-tol m] "
                     "[--scheme linear|nmpfa] "
                     "[--linear-method lu|gmres|bicgstab] [--pc ilu|ilu2|gamg|boomeramg] "
                     "[--pc-reuse none|step|epoch] [--linear-rtol tolerance] [--linear-atol "
                     "tolerance] [--linear-maxiter n] [PETSc -options ...]\n";
        return EXIT_FAILURE;
    }
    try {
        double initialization_s = 0.0;
        double input_s = 0.0;
        BasinForwardResult result;
        const std::filesystem::path output_directory(argv[2]);
        {
            BasinForwardSettings settings;
            int index = 3;
            for (; index < argc && std::string_view(argv[index]).starts_with("--"); ++index) {
                const std::string_view option = argv[index];
                const std::string_view setting = value(argc, argv, index);
                set_basin_forward_option(settings, option, setting);
            }
            std::vector<char*> petsc_arguments{argv[0]};
            petsc_arguments.insert(petsc_arguments.end(), argv + index, argv + argc);
            int petsc_argc = static_cast<int>(petsc_arguments.size());
            petsc_arguments.push_back(nullptr);
            char** petsc_argv = petsc_arguments.data();
            PetscSession petsc(petsc_argc, petsc_argv, initialization_s);
            const auto input_start = Clock::now();
            auto input = BasinInput::read(argv[1]);
            input_s = elapsed_seconds(input_start);
            const auto configs = input.configs();
            BasinForwardSolver solver(std::move(input), settings);
            result = solver.solve(output_directory, [configs](const BasinSnapshot& snapshot) {
                if (snapshot.kind == BasinSnapshot::Kind::after_event)
                    return;
                if (snapshot.kind == BasinSnapshot::Kind::initial) {
                    std::cout << "Starting calculation: age " << snapshot.age_ma << " Ma\n"
                              << std::flush;
                    return;
                }
                const double progress = 100.0 * (configs.front().age_ma - snapshot.age_ma) /
                                        (configs.front().age_ma - configs.back().age_ma);
                std::cout << "Epoch " << snapshot.epoch + 1 << '/' << configs.size() - 1
                          << ", step " << snapshot.step << ": age " << snapshot.age_ma << " Ma; "
                          << progress << "% of model time completed\n"
                          << std::flush;
            });
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
