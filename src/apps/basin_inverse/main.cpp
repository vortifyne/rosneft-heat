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

const char* termination(LeastSquaresStatus status) {
    switch (status) {
    case LeastSquaresStatus::within_errors:
        return "measurement_tolerance";
    case LeastSquaresStatus::limit_reached:
        return "limit_reached";
    default:
        return is_least_squares_converged(status) ? "optimizer_converged" : "optimizer_failed";
    }
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr
            << "Usage: basin_inverse_solver <input-dir> <output-dir> "
               "[--method pounders|gauss-newton] [--initial-q mW/m2] [--initial-parameters csv] "
               "[--maxiter n] [--maxfev n] "
               "[--residual-tol value] [--save-forward] [forward options...] [PETSc -options...]\n";
        return 1;
    }
    std::vector<char*> petsc_arguments{argv[0]};
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
            else if (option == "--method") {
                if (value == "gauss-newton")
                    request.method = BasinInverseMethod::gauss_newton;
                else if (value == "pounders")
                    request.method = BasinInverseMethod::pounders;
                else
                    throw std::invalid_argument("Expected --method pounders or gauss-newton");
            } else if (option == "--maxiter")
                request.optimization.maxiter = std::stoi(value);
            else if (option == "--maxfev")
                request.optimization.maxfev = std::stoi(value);
            else if (option == "--residual-tol")
                request.optimization.residual_tolerance = std::stod(value);
            else
                set_basin_forward_option(settings, option, value);
        }
        const auto& linear = settings.linear_solver;
        const bool forward_uses_petsc = linear.method != LinearMethod::lu &&
                                        !(linear.method == LinearMethod::fgmres &&
                                          linear.preconditioner == PreconditionerType::none);
        if (request.method == BasinInverseMethod::pounders || forward_uses_petsc) {
            petsc_arguments.insert(petsc_arguments.end(), argv + index, argv + argc);
            int petsc_argc = static_cast<int>(petsc_arguments.size());
            petsc_arguments.push_back(nullptr);
            char** petsc_argv = petsc_arguments.data();
            if (PetscInitialize(&petsc_argc, &petsc_argv, nullptr, nullptr) != PETSC_SUCCESS)
                throw std::runtime_error("Cannot initialize PETSc");
            petsc_initialized = true;
        }
        {
            const auto start = Clock::now();
            const std::filesystem::path input(argv[1]), output(argv[2]);
            std::filesystem::create_directories(output);
            BasinForwardSolver forward(BasinInput::read(input), settings);
            if (initial_path.empty()) {
                Vector initial(static_cast<Vector::Index>(forward.configs().size()));
                initial.set_constant(initial_flux / 1000.0);
                forward.set_parameters(initial);
            } else {
                forward.set_parameters(read_basin_parameters(initial_path));
            }
            request.initial_parameters = forward.parameter_space().values();
            write_basin_parameters(output / "initial_parameters.csv", forward.parameter_space());
            BasinHeatInverseSolver solver(BasinObjective(
                std::move(forward), BasinHeatObservationData::read(input / "observations.csv")));
            auto& objective = solver.objective();
            const auto points = objective.observations().points();
            int temperature_count = 0, reflectance_count = 0;
            for (const auto& point : points) {
                if (point.kind == BasinObservable::temperature)
                    ++temperature_count;
                else
                    ++reflectance_count;
            }
            const auto count_within_errors = [points](const Vector& r) {
                int temperature = 0, reflectance = 0;
                for (std::size_t j = 0; j < points.size(); ++j) {
                    if (std::abs(r[static_cast<Vector::Index>(j)]) > 1.0)
                        continue;
                    if (points[j].kind == BasinObservable::temperature)
                        ++temperature;
                    else
                        ++reflectance;
                }
                return std::pair{temperature, reflectance};
            };
            std::cout << "Starting inversion: " << request.initial_parameters.size()
                      << " parameters; T observations=" << temperature_count
                      << "; Ro observations=" << reflectance_count << '\n'
                      << std::flush;
            std::ofstream progress(output / "progress.csv");
            progress << "iteration,evaluation,objective,max_normalized_residual,"
                        "temperature_in_tolerance,temperature_count,"
                        "reflectance_in_tolerance,reflectance_count,elapsed_s\n"
                     << std::setprecision(17);
            const Vector parameters = solver.solve(
                request, [&](int iteration, int evaluation, const Vector&, const Vector& r) {
                    const double elapsed = elapsed_seconds(start);
                    const double value = least_squares_objective(r);
                    const auto normalized = objective.residuals_in_error_units(r);
                    const auto [temperature_inside, reflectance_inside] =
                        count_within_errors(normalized);
                    progress << iteration << ',' << evaluation << ',' << value << ','
                             << normalized.infinity_norm() << ',' << temperature_inside << ','
                             << temperature_count << ',' << reflectance_inside << ','
                             << reflectance_count << ',' << elapsed << '\n'
                             << std::flush;
                    std::cout << "Iteration " << iteration << ", forward " << evaluation
                              << ": objective=" << value
                              << "; max residual=" << normalized.infinity_norm()
                              << "; T=" << temperature_inside << '/' << temperature_count
                              << "; Ro=" << reflectance_inside << '/' << reflectance_count
                              << "; elapsed=" << elapsed << " s\n"
                              << std::flush;
                });
            const auto& result = solver.optimization_result();
            const auto normalized = objective.residuals_in_error_units(result.residuals);
            const auto [temperature_inside, reflectance_inside] = count_within_errors(normalized);
            auto& final_forward = objective.forward_solver();
            final_forward.set_parameters(parameters);
            write_basin_parameters(output / "parameters.csv", final_forward.parameter_space());
            std::ofstream comparison(output / "observations.csv");
            comparison << "kind,well,x,z,observed,calculated,error,normalized_residual\n"
                       << std::setprecision(17);
            Vector::Index i = 0;
            for (const auto& point : points) {
                const bool temperature = point.kind == BasinObservable::temperature;
                const double observed = point.value - (temperature ? 273.15 : 0.0);
                comparison << (temperature ? "temperature" : "reflectance") << ',' << point.well
                           << ',' << point.x << ',' << point.z << ',' << observed << ','
                           << observed + point.error * normalized[i] << ',' << point.error << ','
                           << normalized[i] << '\n';
                ++i;
            }
            std::ofstream summary(output / "summary.txt");
            write_basin_forward_settings(summary, settings);
            summary << std::setprecision(17) << "termination=" << termination(result.status)
                    << "\nparameters=" << parameters.size()
                    << "\nobservations=" << result.residuals.size()
                    << "\nobjective=" << result.objective
                    << "\nmax_normalized_residual=" << normalized.infinity_norm()
                    << "\ntemperature_in_tolerance=" << temperature_inside
                    << "\ntemperature_count=" << temperature_count
                    << "\nreflectance_in_tolerance=" << reflectance_inside
                    << "\nreflectance_count=" << reflectance_count
                    << "\niterations=" << result.iterations
                    << "\nforward_evaluations=" << result.function_evaluations
                    << "\nelapsed_s=" << elapsed_seconds(start) << '\n';
            std::cout << "Finished: " << termination(result.status)
                      << "; objective=" << result.objective
                      << "; max residual=" << normalized.infinity_norm()
                      << "; T=" << temperature_inside << '/' << temperature_count
                      << "; Ro=" << reflectance_inside << '/' << reflectance_count << '\n';
            if (save_forward) {
                final_forward.set_intermediate_snapshots(settings.intermediate_snapshots);
                (void)final_forward.solve(output / "forward");
            }
        }
        if (petsc_initialized)
            PetscFinalize();
    } catch (const std::exception& error) {
        if (petsc_initialized)
            PetscFinalize();
        std::cerr << "basin_inverse_solver: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
