#include "basin/inverse/basin_heat_inverse_solver.hpp"
#include "basin/io/basin_forward_options.hpp"
#include "basin/io/basin_parameters.hpp"
#include "numerics/timing.hpp"

#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <petscsys.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

const char* termination(PoundersStatus status) {
    switch (status) {
    case PoundersStatus::user_converged:
        return "measurement_tolerance";
    case PoundersStatus::maxiter:
        return "iteration_limit";
    case PoundersStatus::maxfev:
        return "evaluation_limit";
    default:
        return is_pounders_converged(status) ? "optimizer_converged" : "optimizer_failed";
    }
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr
            << "Usage: basin_inverse_solver <input-dir> <output-dir> "
               "[--initial-q mW/m2] [--initial-parameters csv] [--maxiter n] [--maxfev n] "
               "[--residual-tol value] [--save-forward] [forward options...] [PETSc -options...]\n";
        return 1;
    }
    bool petsc_initialized = false;
    try {
        BasinForwardSettings settings;
        BasinHeatInverseSolveRequest request;
        double initial_flux = 75.0;
        bool save_forward = false;
        std::filesystem::path initial_path;
        int index = 3;
        for (; index < argc && std::string_view(argv[index]).starts_with("--"); ++index) {
            const std::string_view option = argv[index];
            if (option == "--save-forward") {
                save_forward = true;
                continue;
            }
            if (++index >= argc)
                throw std::invalid_argument("Missing option value");
            const std::string value(argv[index]);
            if (option == "--initial-q")
                initial_flux = std::stod(value);
            else if (option == "--initial-parameters")
                initial_path = value;
            else if (option == "--maxiter")
                request.optimization.maxiter = std::stoi(value);
            else if (option == "--maxfev")
                request.optimization.maxfev = std::stoi(value);
            else if (option == "--residual-tol")
                request.optimization.residual_tolerance = std::stod(value);
            else
                set_basin_forward_option(settings, option, value);
        }
        std::vector<char*> petsc_arguments{argv[0]};
        petsc_arguments.insert(petsc_arguments.end(), argv + index, argv + argc);
        int petsc_argc = static_cast<int>(petsc_arguments.size());
        petsc_arguments.push_back(nullptr);
        char** petsc_argv = petsc_arguments.data();
        if (PetscInitialize(&petsc_argc, &petsc_argv, nullptr, nullptr) != PETSC_SUCCESS)
            throw std::runtime_error("Cannot initialize PETSc");
        petsc_initialized = true;
        {
            const auto start = Clock::now();
            const std::filesystem::path input(argv[1]), output(argv[2]);
            std::filesystem::create_directories(output);
            settings.intermediate_snapshots = false;
            BasinForwardSolver forward(BasinInput::read(input), settings);
            request.initial_parameters =
                Vector(static_cast<Vector::Index>(forward.configs().size()));
            request.initial_parameters.set_constant(initial_flux * 1e-3);
            if (!initial_path.empty())
                request.initial_parameters = read_basin_parameters(initial_path, forward.configs());
            BasinHeatInverseSolver solver(BasinHeatResidualEvaluator(
                std::move(forward), BasinHeatObservationData::read(input / "observations.csv")));
            std::ofstream progress(output / "progress.csv");
            progress << "evaluation,objective,max_normalized_residual,elapsed_s\n"
                     << std::setprecision(17);
            const Vector parameters =
                solver.solve(request, [&](int evaluation, const Vector& q, const Vector& r) {
                    if (evaluation == 1)
                        write_basin_parameters(
                            output / "initial_parameters.csv",
                            solver.residual_evaluator().forward_solver().configs(), q);
                    const double elapsed = elapsed_seconds(start);
                    progress << evaluation << ',' << r.squared_norm() << ',' << r.infinity_norm()
                             << ',' << elapsed << '\n'
                             << std::flush;
                    std::cout << "Forward " << evaluation << ": objective=" << r.squared_norm()
                              << "; max residual=" << r.infinity_norm() << "; elapsed=" << elapsed
                              << " s\n"
                              << std::flush;
                });
            const auto& result = solver.optimization_result();
            auto& evaluator = solver.residual_evaluator();
            auto& final_forward = evaluator.forward_solver();
            write_basin_parameters(output / "parameters.csv", final_forward.configs(), parameters);
            std::ofstream comparison(output / "observations.csv");
            comparison << "kind,well,x,z,observed,calculated,error,normalized_residual\n"
                       << std::setprecision(17);
            Vector::Index i = 0;
            for (const auto& point : evaluator.observations().points()) {
                const bool temperature = point.kind == BasinObservable::temperature;
                const double observed = point.value - (temperature ? 273.15 : 0.0);
                comparison << (temperature ? "temperature" : "reflectance") << ',' << point.well
                           << ',' << point.x << ',' << point.z << ',' << observed << ','
                           << observed + point.error * result.residuals[i] << ',' << point.error
                           << ',' << result.residuals[i] << '\n';
                ++i;
            }
            std::ofstream summary(output / "summary.txt");
            write_basin_forward_settings(summary, settings);
            summary << std::setprecision(17) << "termination=" << termination(result.status)
                    << "\nparameters=" << parameters.size()
                    << "\nobservations=" << result.residuals.size()
                    << "\nobjective=" << result.objective
                    << "\nmax_normalized_residual=" << result.residuals.infinity_norm()
                    << "\niterations=" << result.iterations
                    << "\nforward_evaluations=" << result.function_evaluations
                    << "\nelapsed_s=" << elapsed_seconds(start) << '\n';
            std::cout << "Finished: " << termination(result.status)
                      << "; objective=" << result.objective
                      << "; max residual=" << result.residuals.infinity_norm() << '\n';
            if (save_forward) {
                final_forward.set_parameters(parameters);
                final_forward.set_intermediate_snapshots(true);
                (void)final_forward.solve(output / "forward");
            }
        }
        PetscFinalize();
    } catch (const std::exception& error) {
        if (petsc_initialized)
            PetscFinalize();
        std::cerr << "basin_inverse_solver: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
