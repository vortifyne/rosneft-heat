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
        const auto width = static_cast<std::size_t>(std::count_if(
            label.begin(), label.end(), [](unsigned char c) { return (c & 0xc0) != 0x80; }));
        output << label << std::string(width < 52 ? 52 - width : 1, ' ') << std::right << std::fixed
               << std::setw(14) << std::setprecision(6) << seconds << std::setw(11)
               << std::setprecision(2) << percent(seconds) << '\n';
    };
    output << "\nЧасть программы                                      Время, с   % программы\n";
    const auto& t = result.timings;
    const auto& nonlinear = t.nonlinear;
    row("Программа", program_s);
    row("  Инициализация и завершение", initialization_s);
    row("  Чтение входных данных", input_s);
    row("  Прямой расчёт", t.total_s);
    row("    Геометрия, сетка и подготовка шагов", t.grid_s);
    row("    Перенос состояния", t.transfer_s);
    row("    Нелинейное решение", nonlinear.total_s);
    row("      Сборка", nonlinear.assembly.total_s());
    row("        Невязка", nonlinear.assembly.residual_s);
    row("          Свойства", t.properties.residual_s);
    row("          Прочее", std::max(0.0, nonlinear.assembly.residual_s - t.properties.residual_s));
    row("        Матрица", nonlinear.assembly.matrix_s);
    row("          Свойства", t.properties.matrix_s);
    row("          Прочее", std::max(0.0, nonlinear.assembly.matrix_s - t.properties.matrix_s));
    row("      Линейное решение", nonlinear.linear_s);
    row("      Прочее",
        std::max(0.0, nonlinear.total_s - nonlinear.assembly.total_s() - nonlinear.linear_s));
    row("    EASY%Ro", t.easy_ro_s);
    row("    Запись результатов", t.output_s);
    row("    Прочее", other_seconds(t));
    row("  Прочее", std::max(0.0, program_s - initialization_s - input_s - t.total_s));
    output << '\n';
    row("Свойства, всего (включены в сборку)", t.properties.total_s());

    output << "\nЭпоха  Возраст начала  Возраст конца  Шаги  Нелин. ит.  Лин. ит."
              "      Время, с  % программы\n";
    for (const auto& epoch : result.epochs) {
        output << std::setw(5) << epoch.epoch + 1 << std::setprecision(3) << std::setw(16)
               << epoch.age_begin_ma << std::setw(14) << epoch.age_end_ma << std::setw(6)
               << epoch.timesteps << std::setw(12) << epoch.nonlinear_iterations << std::setw(10)
               << epoch.linear_iterations << std::setw(14) << std::setprecision(6)
               << epoch.timings.total_s << std::setw(13) << std::setprecision(2)
               << percent(epoch.timings.total_s) << '\n';
    }
    output << "\nВсего шагов: " << result.timesteps
           << "; нелинейных итераций: " << result.nonlinear_iterations
           << "; линейных итераций: " << result.linear_iterations << '\n';
    output << "В среднем нелинейных итераций на шаг: "
           << (result.timesteps > 0
                   ? static_cast<double>(result.nonlinear_iterations) / result.timesteps
                   : 0.0)
           << "; линейных на нелинейную: "
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
