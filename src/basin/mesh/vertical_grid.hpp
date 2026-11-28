#pragma once
#include <cstddef>
#include <span>
#include <vector>

std::vector<std::size_t> vertical_epoch_rows(std::span<const double> start,
                                             std::span<const double> end, double dz);
std::vector<double> vertical_faces(std::span<const double> boundaries,
                                   std::span<const std::size_t> rows, double dz_min);
// Endpoint contacts include a collapsed top layer at the start of a deposition epoch.
std::vector<double> vertical_epoch_faces(std::span<const double> start, std::span<const double> end,
                                         std::span<const std::size_t> rows, bool birth,
                                         double theta, double dz_min);
