#include "basin/basin_forward_options.hpp"
#include "physics/easy_ro.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <petscsys.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void easy_ro_advance(benchmark::State& state) {
    constexpr std::size_t reactions = 20;
    std::vector<double> energies(reactions), weights(reactions, 1.0 / reactions);
    for (std::size_t i = 0; i < reactions; ++i) {
        energies[i] = (34.0 + 2.0 * static_cast<double>(i)) * 4186.8;
    }
    const EasyRoModel model({.preexponential = 1e13,
                             .gas_constant = 8.314,
                             .activation_energies = energies,
                             .weights = weights});
    const auto cells = static_cast<std::size_t>(state.range(0));
    std::vector<double> old_temperature(cells), new_temperature(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        old_temperature[i] = 300.0 + static_cast<double>(i % 200);
        new_temperature[i] = old_temperature[i] + 10.0;
    }
    auto integrals = model.initial_state(cells);
    for (auto _ : state) {
        state.PauseTiming();
        std::fill(integrals.begin(), integrals.end(), 0.0);
        state.ResumeTiming();
        model.advance(old_temperature, new_temperature, 1e12, integrals);
        benchmark::DoNotOptimize(integrals.data());
        benchmark::ClobberMemory();
    }
    if (!std::all_of(integrals.begin(), integrals.end(),
                     [](double value) { return std::isfinite(value) && value >= 0.0; }))
        state.SkipWithError("Nonfinite maturity integrals");
    state.SetItemsProcessed(state.iterations() * static_cast<benchmark::IterationCount>(cells));
    state.counters["reactions"] = static_cast<double>(reactions);
}

} // namespace

int main(int argc, char** argv) {
    try {
        struct PetscSession {
            bool active = false;
            ~PetscSession() {
                if (active)
                    PetscFinalize();
            }
        } petsc;
        bool failed = false;
        benchmark::Initialize(&argc, argv);
        BasinForwardSettings settings;
        std::filesystem::path input, output;
        for (int i = 1; i < argc; ++i) {
            const std::string_view option(argv[i]);
            if (++i == argc)
                throw std::invalid_argument("Missing option value");
            const std::string_view value(argv[i]);
            if (option == "--input")
                input = value;
            else if (option == "--output")
                output = value;
            else
                set_basin_forward_option(settings, option, value);
        }
        if (!output.empty() && input.empty()) {
            throw std::invalid_argument("--output requires --input");
        }
        for (const auto* key : {"HEAT_BENCHMARK_REVISION", "HEAT_BENCHMARK_DIRTY",
                                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}) {
            const auto* value = std::getenv(key);
            benchmark::AddCustomContext(key, value ? value : "unspecified");
        }
        benchmark::AddCustomContext("compiler", HEAT_BENCHMARK_COMPILER);
        benchmark::RegisterBenchmark("EasyRoAdvance", easy_ro_advance)->Arg(1024)->Arg(16384);

        if (!input.empty()) {
            if (PetscInitializeNoArguments() != PETSC_SUCCESS) {
                throw std::runtime_error("Cannot initialize PETSc");
            }
            petsc.active = true;
            benchmark::AddCustomContext("input", std::filesystem::absolute(input).string());
            benchmark::AddCustomContext("output", output.empty() ? "disabled" : output.string());
            std::ostringstream description;
            write_basin_forward_settings(description, settings);
            benchmark::AddCustomContext("settings", description.str());
            const auto solver =
                std::make_shared<BasinForwardSolver>(BasinInput::read(input), settings);
            benchmark::RegisterBenchmark(
                output.empty() ? "BasinForward/compute" : "BasinForward/output",
                [solver, output, &failed](benchmark::State& state) {
                    BasinForwardResult result;
                    for (auto _ : state) {
                        try {
                            result = output.empty() ? solver->solve() : solver->solve(output);
                            benchmark::DoNotOptimize(result);
                        } catch (const std::exception& error) {
                            failed = true;
                            state.SkipWithError(error.what());
                            break;
                        }
                    }
                    state.counters["cells_max"] = static_cast<double>(result.maximum_cells);
                    state.counters["steps"] = result.accepted_steps;
                    state.counters["nonlinear_iterations"] = result.nonlinear_iterations;
                    state.counters["linear_iterations"] = result.linear_iterations;
                    state.counters["global_balance"] = result.maximum_global_balance;
                })
                ->UseRealTime();
        }
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        if (failed)
            return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "benchmarks: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
