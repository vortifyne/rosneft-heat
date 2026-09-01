#pragma once

#include <cmath>
#include <stdexcept>

struct RegularGrid {
    const int nx;
    const int nz;
    const double width;
    const double surface_depth;
    const double bottom_depth;

    const double dx;
    const double dz;
    const double inv_dx;
    const double inv_dz;
    const double cell_area;
    const double inv_cell_area;

    RegularGrid(int nx, int nz, double width, double surface_depth, double bottom_depth)
        : nx(validate_cell_count(nx)), nz(validate_cell_count(nz)), width(validate_width(width)),
          surface_depth(validate_surface_depth(surface_depth)),
          bottom_depth(validate_bottom_depth(bottom_depth, this->surface_depth)),
          dx(this->width / static_cast<double>(this->nx)),
          dz((this->bottom_depth - this->surface_depth) / static_cast<double>(this->nz)),
          inv_dx(1.0 / dx), inv_dz(1.0 / dz), cell_area(dx * dz), inv_cell_area(inv_dx * inv_dz) {}

    constexpr int cell_count() const noexcept {
        return nx * nz;
    }

private:
    static int validate_cell_count(int value) {
        if (value <= 0) {
            throw std::invalid_argument("Regular grid dimensions must be positive");
        }
        return value;
    }

    static double validate_width(double value) {
        if (!(value > 0.0) || !std::isfinite(value)) {
            throw std::invalid_argument("Regular grid width must be finite and positive");
        }
        return value;
    }

    static double validate_surface_depth(double value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Regular grid surface depth must be finite");
        }
        return value;
    }

    static double validate_bottom_depth(double value, double surface_depth) {
        if (!std::isfinite(value) || !(value > surface_depth)) {
            throw std::invalid_argument(
                "Regular grid bottom must be finite and below the finite surface");
        }
        return value;
    }
};
