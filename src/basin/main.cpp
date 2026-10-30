#include "basin/basin_forward_options.hpp"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <petscsys.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

class PetscSession {
public:
    PetscSession() {
        if (PetscInitializeNoArguments() != PETSC_SUCCESS) {
            throw std::runtime_error("Cannot initialize PETSc");
        }
    }
    ~PetscSession() {
        PetscFinalize();
    }
};

std::string_view value(int argc, const char* const argv[], int& index) {
    if (++index >= argc)
        throw std::invalid_argument("Missing option value");
    return argv[index];
}

} // namespace

int main(int argc, const char* const argv[]) {
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
        PetscSession petsc;
        BasinForwardSettings settings;
        for (int index = 3; index < argc; ++index) {
            const std::string_view option = argv[index];
            const std::string_view setting = value(argc, argv, index);
            set_basin_forward_option(settings, option, setting);
        }
        BasinForwardSolver solver(BasinInput::read(argv[1]), settings);
        const BasinForwardResult result = solver.solve(std::filesystem::path(argv[2]));
        std::cout << "accepted_steps=" << result.accepted_steps
                  << " maximum_cells=" << result.maximum_cells
                  << " maximum_global_balance=" << result.maximum_global_balance
                  << " final_age_ma=" << result.final_age_ma << '\n';
    } catch (const std::exception& error) {
        std::cerr << "basin_forward_solver: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
