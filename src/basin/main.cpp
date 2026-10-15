#include "basin/basin_forward_solver.hpp"

#include <array>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <petscsys.h>
#include <string>
#include <string_view>
#include <utility>

namespace {

class PetscSession {
public:
    PetscSession() {
        if (PetscInitializeNoArguments() != PETSC_SUCCESS) {
            throw std::runtime_error("Failed to initialize PETSc");
        }
    }
    ~PetscSession() {
        PetscFinalize();
    }
    PetscSession(const PetscSession&) = delete;
    PetscSession& operator=(const PetscSession&) = delete;
};

LinearSolverKind parse_linear_solver(std::string_view name) {
    static constexpr std::array solvers{
        std::pair{"umfpack", LinearSolverKind::umfpack_lu},
        std::pair{"petsc-gmres-ilu", LinearSolverKind::petsc_gmres_ilu},
        std::pair{"petsc-gmres-ilu2", LinearSolverKind::petsc_gmres_ilu2},
        std::pair{"petsc-gmres-gamg", LinearSolverKind::petsc_gmres_gamg},
        std::pair{"petsc-gmres-boomeramg", LinearSolverKind::petsc_gmres_boomeramg},
        std::pair{"petsc-bicgstab-ilu", LinearSolverKind::petsc_bicgstab_ilu},
        std::pair{"petsc-bicgstab-ilu2", LinearSolverKind::petsc_bicgstab_ilu2},
        std::pair{"petsc-bicgstab-gamg", LinearSolverKind::petsc_bicgstab_gamg},
        std::pair{"petsc-bicgstab-boomeramg", LinearSolverKind::petsc_bicgstab_boomeramg},
        std::pair{"petsc-gmres-gamg-reuse", LinearSolverKind::petsc_gmres_gamg_reuse},
        std::pair{"petsc-gmres-boomeramg-reuse", LinearSolverKind::petsc_gmres_boomeramg_reuse},
        std::pair{"petsc-bicgstab-gamg-reuse", LinearSolverKind::petsc_bicgstab_gamg_reuse},
        std::pair{"petsc-bicgstab-boomeramg-reuse",
                  LinearSolverKind::petsc_bicgstab_boomeramg_reuse},
    };
    for (const auto& [label, kind] : solvers) {
        if (name == label)
            return kind;
    }
    throw std::invalid_argument("Unknown linear solver: " + std::string(name));
}

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
        std::cerr
            << "Usage: heat_forward <input-dir> <output-dir> [options]\n"
               "Options:\n"
               "  --cell-size <m>\n"
               "  --dt-ma <million-years>\n"
               "  --linear-solver <umfpack|petsc-{gmres|bicgstab}-{ilu|ilu2|gamg|boomeramg}>\n"
               "    AMG variants also accept the -reuse suffix (freeze AMG within an epoch)\n"
               "  --linear-rtol <value> (default: 1e-8)\n"
               "  --linear-atol <value> (default: 1e-50)\n"
               "  --linear-max-iterations <count> (default: 1000)\n";
        return EXIT_FAILURE;
    }
    try {
        PetscSession petsc;
        BasinForwardSettings settings;
        for (int index = 3; index < argc; ++index) {
            const std::string_view argument = argv[index];
            if (argument == "--cell-size") {
                settings.cell_size =
                    std::stod(std::string(require_value(argc, argv, index, argument)));
            } else if (argument == "--dt-ma") {
                settings.timestep_ma =
                    std::stod(std::string(require_value(argc, argv, index, argument)));
            } else if (argument == "--linear-solver") {
                const auto value = require_value(argc, argv, index, argument);
                settings.linear_solver = parse_linear_solver(value);
            } else if (argument == "--linear-rtol") {
                settings.linear.relative_tolerance =
                    std::stod(std::string(require_value(argc, argv, index, argument)));
            } else if (argument == "--linear-atol") {
                settings.linear.absolute_tolerance =
                    std::stod(std::string(require_value(argc, argv, index, argument)));
            } else if (argument == "--linear-max-iterations") {
                settings.linear.max_iterations =
                    std::stoi(std::string(require_value(argc, argv, index, argument)));
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
