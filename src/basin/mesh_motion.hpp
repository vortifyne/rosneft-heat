#pragma once

#include "basin/basin_input.hpp"
#include "mesh/mesh2d.hpp"

#include <span>
#include <vector>

class BasinMeshMotion {
public:
    BasinMeshMotion(Mesh2D& mesh, const BasinConfiguration& first, const BasinConfiguration& second,
                    double duration, bool preserve_topology = false);

    void set_position(double fraction);
    void set_porosity(double fraction, std::span<double> porosity) const;

    [[nodiscard]] std::span<const double> velocity_z() const noexcept;

private:
    Mesh2D& mesh_;
    std::vector<Point2D> initial_;
    std::vector<Point2D> final_;
    std::vector<double> porosity_first_;
    std::vector<double> porosity_second_;
    std::vector<double> velocity_z_;
};
