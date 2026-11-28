#pragma once

#include "basin/forward/basin_forward_settings.hpp"
#include "basin/inverse/basin_heat_observation_data.hpp"
#include "basin/inverse/basin_heat_parameter_space.hpp"
#include "basin/io/basin_input.hpp"
#include "basin_1d/mesh/section_transfer.hpp"

class BasinForwardSolver1D {
public:
    BasinForwardSolver1D(const BasinInput& input, double x, BasinForwardSettings settings = {});
    const BasinSection1D& section() const noexcept {
        return section_;
    }
    // Projected observations in input order; optional raw d(T, Ro)/d(eta), before weighting.
    [[nodiscard]] Vector solve_at_wells(std::span<const BasinObservation> observations,
                                        const BasinHeatParameterSpace& space, const Vector& eta,
                                        Eigen::MatrixXd* jacobian = nullptr) const;

private:
    struct Stage {
        SectionEpoch1D mesh;
        SectionLayout layout;
        double age0, age1, surface0, surface1;
        struct Material {
            LithotypeProperties rock;
            SectionLayer1D start, end;
        };
        std::vector<Material> materials;
        std::vector<double> deposition_times;
        SectionTransfer transfer;
        void update_porosity(std::span<const double> z, double theta, std::span<double> phi) const;
    };
    BasinSection1D section_;
    BasinForwardSettings settings_;
    EasyRoModel easy_ro_;
    // Initial stationary grid, geological epochs, present-day sampling grid.
    std::vector<Stage> stages_;
};
