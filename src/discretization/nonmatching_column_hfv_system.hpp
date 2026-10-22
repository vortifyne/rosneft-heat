#pragma once

#include "basin/nonmatching_column_grid.hpp"
#include "discretization/nonmatching_column_flux.hpp"
#include "linear/sparse_matrix.hpp"
#include "linear/vector.hpp"

#include <functional>
#include <optional>
#include <span>
#include <vector>

struct NonmatchingHfvEnergyRates {
    double surface_outflow = 0.0;
    double basal_inflow = 0.0;
    double lateral_inflow = 0.0;
    double heat_production = 0.0;
};

struct NonmatchingHfvDiagnostics {
    std::size_t face_unknowns = 0;
    std::size_t maximum_cell_faces = 0;
    double internal_conservation_defect_linf = 0.0;
};

// Full hybrid finite-volume endpoint of the SUSHI family (B = empty in the
// notation of Eymard--Gallouet--Herbin). It is a verification operator: no
// face unknown is eliminated, so the implementation does not rely on a new
// barycentric reconstruction.
class NonmatchingColumnHfvSystem {
public:
    using PropertyUpdater =
        std::function<void(std::span<const double>, std::span<double>, std::span<double>)>;

    NonmatchingColumnHfvSystem(const NonmatchingColumnGrid& grid,
                               std::span<const double> thermal_conductivity,
                               std::span<const double> heat_production,
                               NonmatchingColumnBoundaryData boundary_data,
                               PropertyUpdater property_updater = {});

    [[nodiscard]] Vector::Index size() const noexcept;
    [[nodiscard]] std::size_t cell_count() const noexcept;
    [[nodiscard]] std::size_t face_unknown_count() const noexcept;
    [[nodiscard]] std::span<const double> thermal_conductivity() const noexcept;
    [[nodiscard]] std::span<const double> heat_production() const noexcept;

    void assemble_system(const Vector& linearization, SparseMatrix& matrix, Vector& rhs) const;
    void assemble_residual(const Vector& solution, Vector& residual) const;

    [[nodiscard]] Vector interpolate(const std::function<double(Point2D)>& function) const;
    [[nodiscard]] NonmatchingHfvEnergyRates energy_rates(const Vector& solution) const;
    [[nodiscard]] NonmatchingHfvDiagnostics diagnostics(const Vector& solution) const;
    [[nodiscard]] double material_interface_outflow(const Vector& solution, int layer) const;

private:
    enum class FaceKind {
        internal,
        surface_dirichlet,
        basal_neumann,
        left_neumann,
        right_neumann,
        basal_step_neumann,
    };

    struct Face {
        Point2D center{};
        double length = 0.0;
        FaceKind kind = FaceKind::internal;
        std::optional<std::size_t> unknown;
        std::vector<std::size_t> cells;
    };

    struct CellFace {
        std::size_t face = 0;
        double normal_x = 0.0;
        double normal_z = 0.0;
    };

    void update_properties(const Vector& solution) const;
    [[nodiscard]] double prescribed_face_value(const Face& face) const;
    [[nodiscard]] double prescribed_inward_flux(const Face& face) const;

    const NonmatchingColumnGrid* grid_ = nullptr;
    mutable std::vector<double> thermal_conductivity_;
    mutable std::vector<double> heat_production_;
    NonmatchingColumnBoundaryData boundary_data_;
    PropertyUpdater property_updater_;
    std::vector<Face> faces_;
    std::vector<std::vector<CellFace>> cell_faces_;
    std::size_t face_unknown_count_ = 0;
};
