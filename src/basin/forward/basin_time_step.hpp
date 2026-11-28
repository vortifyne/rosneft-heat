#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>

inline constexpr double kSecondsPerMa = 1.e6 * 365.25 * 86400.;

struct BasinTimeStep {
    double time, theta, age_ma;
    bool at_stop;
};

// A flux knot or epoch boundary is an exact endpoint for the mean basal flux.
inline BasinTimeStep next_basin_step(double t, double dt_max, double oldest, double youngest,
                                     double duration, double stop_age) {
    const double stop = (oldest - stop_age) * kSecondsPerMa;
    const double proposed = t + dt_max;
    const bool at_stop = stop - proposed <= 8. * std::numeric_limits<double>::epsilon() * stop;
    const double next = at_stop ? stop : proposed;
    if (!(next > t))
        throw std::runtime_error("Basal flux knot or timestep is below time resolution");
    const double theta = next / duration;
    return {next, theta, at_stop ? stop_age : std::lerp(oldest, youngest, theta), at_stop};
}
