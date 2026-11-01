#include "basin/io/basin_statistics.hpp"

#include "basin/forward/basin_forward_solver.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

double other_seconds(const BasinTimings& timings) {
    return std::max(0.0, timings.total_s - timings.grid_s - timings.transfer_s -
                             timings.nonlinear.total_s - timings.easy_ro_s - timings.output_s);
}

void write_intervals(const std::filesystem::path& path,
                     std::span<const BasinIntervalStats> intervals) {
    std::ofstream file(path);
    file << "index,epoch,age_begin_ma,age_end_ma,timesteps,nonlinear_iterations,linear_iterations,"
            "total_s,grid_s,transfer_s,nonlinear_s,assembly_s,residual_s,matrix_s,"
            "residual_properties_s,matrix_properties_s,linear_s,easy_ro_s,output_s,other_s\n";
    file << std::setprecision(17);
    std::size_t index = 0;
    for (const auto& interval : intervals) {
        const auto& t = interval.timings;
        file << ++index << ',' << interval.epoch + 1 << ',' << interval.age_begin_ma << ','
             << interval.age_end_ma << ',' << interval.timesteps << ','
             << interval.nonlinear_iterations << ',' << interval.linear_iterations << ','
             << t.total_s << ',' << t.grid_s << ',' << t.transfer_s << ',' << t.nonlinear.total_s
             << ',' << t.nonlinear.assembly.total_s() << ',' << t.nonlinear.assembly.residual_s
             << ',' << t.nonlinear.assembly.matrix_s << ',' << t.properties.residual_s << ','
             << t.properties.matrix_s << ',' << t.nonlinear.linear_s << ',' << t.easy_ro_s << ','
             << t.output_s << ',' << other_seconds(t) << '\n';
    }
    file.close();
    if (!file)
        throw std::runtime_error("Cannot write calculation statistics: " + path.string());
}

} // namespace

void write_basin_statistics(std::ostream& output, const BasinForwardResult& result,
                            double program_s, double initialization_s, double input_s) {
    const auto flags = output.flags();
    const auto precision = output.precision();
    const auto percent = [program_s](double seconds) {
        return program_s > 0.0 ? 100.0 * seconds / program_s : 0.0;
    };
    const auto row = [&](std::string_view label, double seconds) {
        output << std::left << std::setw(52) << label << std::right << std::fixed << std::setw(14)
               << std::setprecision(6) << seconds << std::setw(11) << std::setprecision(2)
               << percent(seconds) << '\n';
    };
    output << '\n'
           << std::left << std::setw(52) << "Stage" << std::right << std::setw(14) << "Time, s"
           << std::setw(11) << "% program" << '\n';
    const auto& t = result.timings;
    const auto& nonlinear = t.nonlinear;
    row("Program", program_s);
    row("  Initialization and shutdown", initialization_s);
    row("  Input", input_s);
    row("  Forward solve", t.total_s);
    row("    Geometry, grid and step preparation", t.grid_s);
    row("    State transfer", t.transfer_s);
    row("    Nonlinear solve", nonlinear.total_s);
    row("      Assembly", nonlinear.assembly.total_s());
    row("        Residual", nonlinear.assembly.residual_s);
    row("          Properties", t.properties.residual_s);
    row("          Other", std::max(0.0, nonlinear.assembly.residual_s - t.properties.residual_s));
    row("        Matrix", nonlinear.assembly.matrix_s);
    row("          Properties", t.properties.matrix_s);
    row("          Other", std::max(0.0, nonlinear.assembly.matrix_s - t.properties.matrix_s));
    row("      Linear solve", nonlinear.linear_s);
    row("      Other",
        std::max(0.0, nonlinear.total_s - nonlinear.assembly.total_s() - nonlinear.linear_s));
    row("    EASY%Ro", t.easy_ro_s);
    row("    Output", t.output_s);
    row("    Other", other_seconds(t));
    row("  Other", std::max(0.0, program_s - initialization_s - input_s - t.total_s));
    output << '\n';
    row("Properties, total (included in assembly)", t.properties.total_s());

    output << '\n'
           << std::setw(5) << "Epoch" << std::setw(16) << "Start age, Ma" << std::setw(14)
           << "End age, Ma" << std::setw(6) << "Steps" << std::setw(12) << "Nonlin. it."
           << std::setw(10) << "Lin. it." << std::setw(14) << "Time, s" << std::setw(13)
           << "% program" << '\n';
    for (const auto& epoch : result.epochs) {
        output << std::setw(5) << epoch.epoch + 1 << std::setprecision(3) << std::setw(16)
               << epoch.age_begin_ma << std::setw(14) << epoch.age_end_ma << std::setw(6)
               << epoch.timesteps << std::setw(12) << epoch.nonlinear_iterations << std::setw(10)
               << epoch.linear_iterations << std::setw(14) << std::setprecision(6)
               << epoch.timings.total_s << std::setw(13) << std::setprecision(2)
               << percent(epoch.timings.total_s) << '\n';
    }
    output << "\nTotal timesteps: " << result.timesteps
           << "; nonlinear iterations: " << result.nonlinear_iterations
           << "; linear iterations: " << result.linear_iterations << '\n';
    output << "Average nonlinear iterations per timestep: "
           << (result.timesteps > 0
                   ? static_cast<double>(result.nonlinear_iterations) / result.timesteps
                   : 0.0)
           << "; linear iterations per nonlinear iteration: "
           << (result.nonlinear_iterations > 0
                   ? static_cast<double>(result.linear_iterations) / result.nonlinear_iterations
                   : 0.0)
           << '\n';
    output.flags(flags);
    output.precision(precision);
}

void write_basin_intervals(const std::filesystem::path& directory,
                           const BasinForwardResult& result) {
    write_intervals(directory / "epochs.csv", result.epochs);
    write_intervals(directory / "timesteps.csv", result.steps);
}
