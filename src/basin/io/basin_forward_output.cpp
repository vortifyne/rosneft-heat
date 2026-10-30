#include "basin/basin_snapshot.hpp"
#include "basin/forward/basin_forward_solver.hpp"
#include "basin/io/basin_binary.hpp"
#include "basin/io/basin_forward_options.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

BasinForwardResult BasinForwardSolver::solve(const std::filesystem::path& output,
                                             const BasinForwardObserver& observe) const {
    std::filesystem::create_directories(output);
    std::size_t snapshot_number = 0;
    const BasinForwardResult result = solve([&](const BasinForwardSnapshot& snapshot) {
        std::ostringstream name;
        name << "snapshot_" << std::setw(4) << std::setfill('0') << snapshot_number++ << ".cgrd";
        write_basin_binary(output / name.str(), snapshot);
        if (observe)
            observe(snapshot);
    });
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
    return result;
}
