#include "basin/mesh/vertical_grid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

std::vector<std::size_t> vertical_epoch_rows(std::span<const double> start,
                                             std::span<const double> end, double dz) {
    std::vector<std::size_t> rows;
    for (std::size_t layer = 0; layer < end.size() - 1; ++layer) {
        const double thickness =
            std::max(start[layer + 1] - start[layer], end[layer + 1] - end[layer]);
        rows.push_back(
            thickness > 1.0e-12
                ? std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(thickness / dz)))
                : 0);
    }
    return rows;
}

std::vector<double> vertical_faces(std::span<const double> boundaries,
                                   std::span<const std::size_t> rows, double dz_min) {
    std::vector<double> thickness(rows.size(), 0.0);
    double required = 0.0, available = 0.0;
    for (std::size_t layer = 0; layer < rows.size(); ++layer) {
        if (rows[layer] == 0)
            continue;
        thickness[layer] = std::max(0.0, boundaries[layer + 1] - boundaries[layer]);
        const double floor = static_cast<double>(rows[layer]) * dz_min;
        required += std::max(0.0, floor - thickness[layer]);
        available += std::max(0.0, thickness[layer] - floor);
    }
    if (required > available + 1.0e-9)
        throw std::runtime_error("Minimum cell height exceeds column height");
    if (required > 0.0) {
        for (std::size_t layer = 0; layer < rows.size(); ++layer) {
            if (rows[layer] == 0)
                continue;
            const double floor = static_cast<double>(rows[layer]) * dz_min;
            if (thickness[layer] < floor)
                thickness[layer] = floor;
            else if (available > 0.0)
                thickness[layer] -= required * (thickness[layer] - floor) / available;
        }
    }
    std::vector<double> faces{boundaries.front()};
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const double top = faces.back();
        for (std::size_t j = 1; j <= rows[i]; ++j)
            faces.push_back(top +
                            thickness[i] * static_cast<double>(j) / static_cast<double>(rows[i]));
    }
    return faces;
}

std::vector<double> vertical_epoch_faces(std::span<const double> start, std::span<const double> end,
                                         std::span<const std::size_t> rows, bool birth,
                                         double theta, double dz_min) {
    std::vector<double> boundaries(start.size());
    for (std::size_t i = 0; i < start.size(); ++i)
        boundaries[i] = std::lerp(start[i], end[i], theta);
    if (!birth || rows.front() == 0)
        return vertical_faces(boundaries, rows, dz_min);

    // The new layer unfolds from bottom to top as its material reaches the surface.
    const auto count = rows.front();
    const double height = end[1] - end[0];
    std::vector<double> depths{boundaries.front()};
    std::vector<std::size_t> counts(count, 1);
    for (std::size_t row = 1; row <= count; ++row) {
        const double a = static_cast<double>(row) / static_cast<double>(count);
        depths.push_back(boundaries.front() + height * std::max(0.0, theta + a - 1.0));
    }
    depths.insert(depths.end(), boundaries.begin() + 2, boundaries.end());
    counts.insert(counts.end(), rows.begin() + 1, rows.end());
    return vertical_faces(depths, counts, dz_min);
}
