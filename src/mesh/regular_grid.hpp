#pragma once

#include <cmath>
#include <stdexcept>

struct RegularGrid {
    const int nx;
    const int nz;
    const double W;
    const double z_surf;
    const double L;

    const double dx;
    const double dz;
    const double inv_dx;
    const double inv_dz;
    const double cell_area;
    const double inv_cell_area;

    RegularGrid(int nx, int nz, double W, double z_surf, double L)
        : nx(nx), nz(nz), W(W), z_surf(z_surf), L(L), dx(W / static_cast<double>(nx)),
          dz((L - z_surf) / static_cast<double>(nz)), inv_dx(1.0 / dx), inv_dz(1.0 / dz),
          cell_area(dx * dz), inv_cell_area(inv_dx * inv_dz) {
        if (nx <= 0 || nz <= 0) {
            throw std::invalid_argument("Regular grid dimensions must be positive");
        }
        if (!std::isfinite(W) || !(W > 0.0)) {
            throw std::invalid_argument("Regular grid width must be finite and positive");
        }
        if (!std::isfinite(z_surf) || !std::isfinite(L) || !(L > z_surf)) {
            throw std::invalid_argument(
                "Regular grid bottom must be finite and below the finite surface");
        }
    }

    int cell_count() const noexcept {
        return nx * nz;
    }
};
