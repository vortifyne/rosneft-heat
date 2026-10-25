#pragma once

#include "basin/nonmatching_column_grid.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

struct ColumnNtpfaAuditFailure {
    std::size_t cell = 0;
    Point2D face_center{};
    std::string reason;
};

struct ColumnNtpfaGeometryAudit {
    std::size_t internal_sides = 0;
    std::size_t dirichlet_cell_sides = 0;
    std::size_t dirichlet_boundary_sides = 0;
    std::size_t failed_sides = 0;
    std::size_t target_harmonic_points_outside_subfaces = 0;
    double maximum_selected_condition = 0.0;
    std::vector<ColumnNtpfaAuditFailure> examples;
};

// Check the 2D specialization of the positive co-normal decomposition in
// FVM_INMOST_2020 §2.3. Each nonmatching overlap segment is a separate face.
// This is a geometric applicability check; it does not solve the PDE.
[[nodiscard]] ColumnNtpfaGeometryAudit
audit_column_ntpfa_geometry(const NonmatchingColumnGrid& grid,
                            std::span<const double> conductivity);
