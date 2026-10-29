#include "basin/basin_binary.hpp"
#include "basin/basin_forward_solver.hpp"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <petscsys.h>
#include <sstream>
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
        std::cerr << "Usage: heat_forward <input-dir> <output-dir> "
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
        const std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);
        std::size_t snapshot_number = 0;
        const BasinForwardResult result = solver.solve([&](const BasinForwardSnapshot& snapshot) {
            std::ostringstream name;
            name << "snapshot_" << std::setw(4) << std::setfill('0') << snapshot_number++
                 << ".cgrid";
            write_basin_binary(output / name.str(), snapshot);
        });
        std::ofstream summary(output / "summary.txt");
        summary << std::setprecision(17) << "dx=" << settings.horizontal_step << '\n'
                << "dz=" << settings.vertical_step << '\n'
                << "dt_ma=" << settings.timestep_ma << '\n'
                << "rdp_tol=" << settings.rdp_tolerance << '\n'
                << "axis_rdp_tol=";
        if (settings.axis_rdp_tolerance)
            summary << *settings.axis_rdp_tolerance;
        else
            summary << "none";
        summary << '\n'
                << "min_height=" << settings.minimum_cell_height << '\n'
                << "pinchout="
                << (settings.pinchout == BasinForwardSettings::Pinchout::exact_zero ? "zero"
                                                                                    : "positive")
                << '\n'
                << "scheme="
                << (settings.scheme == MultipointFluxScheme::linear ? "linear" : "nmpfa") << '\n'
                << "linear_solver=";
        switch (settings.linear_solver) {
        case LinearSolverKind::umfpack_lu:
            summary << "umfpack";
            break;
        case LinearSolverKind::petsc_gmres_gamg:
            summary << "gamg";
            break;
        case LinearSolverKind::petsc_gmres_boomeramg:
            summary << "boomeramg";
            break;
        default:
            throw std::logic_error("Linear solver is not supported by the command line");
        }
        summary << '\n'
                << "picard_rtol=" << settings.picard_relative_tolerance << '\n'
                << "accepted_steps=" << result.accepted_steps << '\n'
                << "nonlinear_iterations=" << result.nonlinear_iterations << '\n'
                << "linear_iterations=" << result.linear_iterations << '\n'
                << "maximum_cells=" << result.maximum_cells << '\n'
                << "maximum_cell_residual=" << result.maximum_cell_residual << '\n'
                << "maximum_global_balance=" << result.maximum_global_balance << '\n'
                << "maximum_internal_flux_defect=" << result.maximum_internal_flux_defect << '\n'
                << "final_age_ma=" << result.final_age_ma << '\n';
        if (!summary)
            throw std::runtime_error("Failed writing basin summary");
        std::cout << "accepted_steps=" << result.accepted_steps
                  << " maximum_cells=" << result.maximum_cells
                  << " maximum_global_balance=" << result.maximum_global_balance
                  << " final_age_ma=" << result.final_age_ma << '\n';
    } catch (const std::exception& error) {
        std::cerr << "heat_forward: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
