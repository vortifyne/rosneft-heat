#pragma once

#include "basin/geometry/basin_geometry.hpp"

// Simplify shared geological boundaries with RDP while retaining layer
// supports and zero-thickness contacts. tolerance is a distance in metres.
[[nodiscard]] BasinGeometry simplify_basin_geometry_rdp(const BasinGeometry& source,
                                                        double tolerance);
