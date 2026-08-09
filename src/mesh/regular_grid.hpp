#pragma once

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

    RegularGrid(int nx, int nz, double W, double z_surf, double L)
        : nx(nx), nz(nz), W(W), z_surf(z_surf), L(L), dx(W / static_cast<double>(nx)),
          dz((L - z_surf) / static_cast<double>(nz)), inv_dx(1.0 / dx), inv_dz(1.0 / dz) {}
};
