#include "basin/basin_forward_solver.hpp"

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
                     "[--dx m] [--dz m] [--dt-ma Ma] [--picard-rtol tolerance] [--min-height m] "
                     "[--rdp-tol m] [--axis-rdp-tol m] [--pinchout positive|zero] "
                     "[--scheme linear|nmpfa] "
                     "[--linear-solver umfpack|gamg|boomeramg]\n";
        return EXIT_FAILURE;
    }
    try {
        PetscSession petsc;
        BasinForwardSettings settings;
        for (int index = 3; index < argc; ++index) {
            const std::string_view option = argv[index];
            const std::string_view setting = value(argc, argv, index);
            if (option == "--dx")
                settings.horizontal_step = std::stod(std::string(setting));
            else if (option == "--dz")
                settings.vertical_step = std::stod(std::string(setting));
            else if (option == "--dt-ma")
                settings.timestep_ma = std::stod(std::string(setting));
            else if (option == "--picard-rtol") {
                settings.picard_relative_tolerance = std::stod(std::string(setting));
            } else if (option == "--rdp-tol") {
                settings.rdp_tolerance = std::stod(std::string(setting));
            } else if (option == "--axis-rdp-tol") {
                settings.axis_rdp_tolerance = std::stod(std::string(setting));
            } else if (option == "--min-height") {
                settings.minimum_cell_height = std::stod(std::string(setting));
            } else if (option == "--scheme") {
                if (setting == "linear")
                    settings.scheme = MultipointFluxScheme::linear;
                else if (setting == "nmpfa") {
                    settings.scheme = MultipointFluxScheme::nonlinear_conservative;
                } else
                    throw std::invalid_argument("Unknown flux scheme");
            } else if (option == "--pinchout") {
                if (setting == "positive") {
                    settings.pinchout = BasinForwardSettings::Pinchout::positive_height;
                } else if (setting == "zero") {
                    settings.pinchout = BasinForwardSettings::Pinchout::exact_zero;
                } else
                    throw std::invalid_argument("Unknown pinchout handling");
            } else if (option == "--linear-solver") {
                if (setting == "umfpack")
                    settings.linear_solver = LinearSolverKind::umfpack_lu;
                else if (setting == "gamg") {
                    settings.linear_solver = LinearSolverKind::petsc_gmres_gamg;
                } else if (setting == "boomeramg") {
                    settings.linear_solver = LinearSolverKind::petsc_gmres_boomeramg;
                } else
                    throw std::invalid_argument("Unknown linear solver");
            } else
                throw std::invalid_argument("Unknown option: " + std::string(option));
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
