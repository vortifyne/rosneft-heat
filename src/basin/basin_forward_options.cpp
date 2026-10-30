#include "basin/basin_forward_options.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using MethodName = std::pair<std::string_view, LinearMethod>;
constexpr std::array kMethods{MethodName{"lu", LinearMethod::lu},
                              MethodName{"gmres", LinearMethod::gmres},
                              MethodName{"bicgstab", LinearMethod::bicgstab}};
using PcName = std::pair<std::string_view, Preconditioner>;
constexpr std::array kPreconditioners{
    PcName{"ilu", Preconditioner::ilu}, PcName{"ilu2", Preconditioner::ilu2},
    PcName{"gamg", Preconditioner::gamg}, PcName{"boomeramg", Preconditioner::boomeramg}};
using ReuseName = std::pair<std::string_view, PreconditionerReuse>;
constexpr std::array kReuse{ReuseName{"none", PreconditionerReuse::none},
                            ReuseName{"step", PreconditionerReuse::step},
                            ReuseName{"epoch", PreconditionerReuse::manual}};

template <typename T, std::size_t N>
T parse(std::string_view value, const std::array<std::pair<std::string_view, T>, N>& names) {
    for (const auto& [name, setting] : names) {
        if (name == value)
            return setting;
    }
    throw std::invalid_argument("Unknown setting: " + std::string(value));
}

template <typename T, std::size_t N>
std::string_view name(T value, const std::array<std::pair<std::string_view, T>, N>& names) {
    for (const auto& [text, setting] : names) {
        if (setting == value)
            return text;
    }
    throw std::invalid_argument("Unknown solver setting");
}

double number(std::string_view value) {
    std::size_t consumed = 0;
    const double result = std::stod(std::string(value), &consumed);
    if (consumed != value.size() || !std::isfinite(result)) {
        throw std::invalid_argument("Expected a finite number: " + std::string(value));
    }
    return result;
}

int iterations(std::string_view value) {
    std::size_t consumed = 0;
    const int result = std::stoi(std::string(value), &consumed);
    if (consumed != value.size() || result <= 0) {
        throw std::invalid_argument("Expected a positive iteration limit");
    }
    return result;
}

} // namespace

void set_basin_forward_option(BasinForwardSettings& settings, std::string_view option,
                              std::string_view value) {
    if (option == "--dx")
        settings.horizontal_step = number(value);
    else if (option == "--dz")
        settings.vertical_step = number(value);
    else if (option == "--dt-ma")
        settings.timestep_ma = number(value);
    else if (option == "--rdp-tol")
        settings.rdp_tolerance = number(value);
    else if (option == "--axis-rdp-tol")
        settings.axis_rdp_tolerance = number(value);
    else if (option == "--min-height")
        settings.minimum_cell_height = number(value);
    else if (option == "--linear-method")
        settings.linear_solver.method = parse(value, kMethods);
    else if (option == "--pc")
        settings.linear_solver.preconditioner = parse(value, kPreconditioners);
    else if (option == "--pc-reuse")
        settings.pc_reuse = parse(value, kReuse);
    else if (option == "--linear-rtol")
        settings.linear.relative_tolerance = number(value);
    else if (option == "--linear-atol")
        settings.linear.absolute_tolerance = number(value);
    else if (option == "--linear-maxiter")
        settings.linear.max_iterations = iterations(value);
    else if (option == "--nonlinear-rtol")
        settings.nonlinear.relative_tolerance = number(value);
    else if (option == "--nonlinear-atol")
        settings.nonlinear.absolute_tolerance = number(value);
    else if (option == "--nonlinear-maxiter")
        settings.nonlinear.max_iterations = iterations(value);
    else if (option == "--scheme") {
        if (value == "linear")
            settings.scheme = MultipointFluxScheme::linear;
        else if (value == "nmpfa")
            settings.scheme = MultipointFluxScheme::nonlinear_conservative;
        else
            throw std::invalid_argument("Unknown flux scheme");
    } else
        throw std::invalid_argument("Unknown option: " + std::string(option));
}

void write_basin_forward_settings(std::ostream& output, const BasinForwardSettings& settings) {
    output << std::setprecision(17) << "dx=" << settings.horizontal_step << '\n'
           << "dz=" << settings.vertical_step << '\n'
           << "dt_ma=" << settings.timestep_ma << '\n'
           << "rdp_tol=" << settings.rdp_tolerance << '\n'
           << "axis_rdp_tol=";
    if (settings.axis_rdp_tolerance)
        output << *settings.axis_rdp_tolerance;
    else
        output << "none";
    output << '\n'
           << "min_height=" << settings.minimum_cell_height << '\n'
           << "scheme=" << (settings.scheme == MultipointFluxScheme::linear ? "linear" : "nmpfa")
           << '\n'
           << "linear_method=" << name(settings.linear_solver.method, kMethods) << '\n'
           << "pc=" << name(settings.linear_solver.preconditioner, kPreconditioners) << '\n'
           << "pc_reuse=" << name(settings.pc_reuse, kReuse) << '\n'
           << "linear_rtol=" << settings.linear.relative_tolerance << '\n'
           << "linear_atol=" << settings.linear.absolute_tolerance << '\n'
           << "linear_maxiter=" << settings.linear.max_iterations << '\n'
           << "nonlinear_rtol=" << settings.nonlinear.relative_tolerance << '\n'
           << "nonlinear_atol=" << settings.nonlinear.absolute_tolerance << '\n'
           << "nonlinear_maxiter=" << settings.nonlinear.max_iterations << '\n';
}
