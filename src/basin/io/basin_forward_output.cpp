#include "basin/basin_snapshot.hpp"
#include "basin/forward/basin_forward_solver.hpp"
#include "basin/io/basin_binary.hpp"
#include "basin/io/basin_forward_options.hpp"
#include "basin/io/basin_statistics.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

BasinForwardResult BasinForwardSolver::solve(const std::filesystem::path& output,
                                             const BasinObserver& observe) const {
    const auto solve_start = Clock::now();
    auto output_start = solve_start;
    const auto snapshots = output / "snapshots";
    std::filesystem::create_directories(snapshots);
    const double prepare_output_s = elapsed_seconds(output_start);
    std::size_t snapshot_number = 0;
    BasinForwardResult result = solve([&](const BasinSnapshot& snapshot) {
        std::ostringstream name;
        name << "snapshot_" << std::setw(4) << std::setfill('0') << snapshot_number++ << ".cgrd";
        write_basin_binary(snapshots / name.str(), snapshot);
        if (observe)
            observe(snapshot);
    });
    output_start = Clock::now();
    std::ofstream summary(output / "summary.txt");
    write_basin_forward_settings(summary, settings_);
    summary << "timesteps=" << result.timesteps << '\n'
            << "nonlinear_iterations=" << result.nonlinear_iterations << '\n'
            << "linear_iterations=" << result.linear_iterations << '\n';
    if (!summary)
        throw std::runtime_error("Failed writing basin summary");
    summary.close();
    if (!summary)
        throw std::runtime_error("Failed closing basin summary");
    write_basin_intervals(output, result);
    result.timings.output_s += prepare_output_s + elapsed_seconds(output_start);
    result.timings.total_s = elapsed_seconds(solve_start);
    return result;
}
