#pragma once

#include "basin/basin_geometry.hpp"

#include <optional>
#include <span>
#include <vector>

// A pre-event contact may jump where the local uppermost old layer changes.
// Sample that geometry per column rather than represent it as BasinGeometry.
struct ColumnEpochGeometry {
    BasinGeometry start;
    BasinGeometry after_event;
    std::optional<int> born_layer_id;
};

struct ColumnEpochSection {
    double x_left = 0.0;
    double x_right = 0.0;
    std::vector<int> layer_ids;      // Top to bottom; old layers only.
    std::vector<double> boundaries; // Mean depths, including top and bottom.
};

[[nodiscard]] ColumnEpochGeometry
make_column_epoch_geometry(const BasinConfiguration& first,
                           const BasinConfiguration& second,
                           double rdp_tolerance = 0.0);

// Include contour knots and layer-support endpoints of both snapshots, plus
// required material-discontinuity coordinates supplied by the caller.
[[nodiscard]] std::vector<double> make_column_epoch_axis(const ColumnEpochGeometry& epoch,
                                                         double horizontal_step,
                                                         std::span<const double> required_x = {});

// fraction=0: start of epoch; fraction=1: immediately before layer birth.
[[nodiscard]] ColumnEpochSection sample_column_epoch(const ColumnEpochGeometry& epoch,
                                                      double x_left, double x_right,
                                                      double fraction);
