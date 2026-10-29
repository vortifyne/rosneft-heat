#pragma once

#include "discretization/multipoint_flux_types.hpp"
#include "mesh/point2d.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

class Mesh2D;

struct MultipointBoundaryData {
    std::function<double(Point2D)> surface_temperature;
    std::function<double(Point2D)> basal_inward_flux;
    std::function<double(Point2D)> left_inward_flux;
    std::function<double(Point2D)> right_inward_flux;
};

struct MultipointInternalFlux {
    std::size_t owner = 0;
    std::size_t neighbor = 0;
    MultipointFluxExpression owner_expression;
    MultipointFluxExpression neighbor_expression;
};

struct MultipointBoundaryFlux {
    std::size_t cell = 0;
    MultipointFluxExpression expression;
};

struct MultipointFluxes {
    std::vector<MultipointInternalFlux> internal;
    std::vector<MultipointBoundaryFlux> surface;
    std::size_t maximum_stencil_size = 0;
};

struct MultipointStencilAuditEntry {
    std::size_t cell = 0;
    std::size_t target_face = 0;
    bool target_is_internal = false;
    bool restricted_pair_found = false;
    bool general_pair_found = false;
    bool general_pair_uses_target = false;
    bool target_harmonic_point_on_face = false;
    bool general_harmonic_points_on_faces = false;
    double general_coefficient_sum = 0.0;
    double general_qtkq_condition = 0.0;
    double general_relative_reconstruction_defect = 0.0;
    double closest_normalized_negative_coefficient = 0.0;
};

struct MultipointStencilAudit {
    std::vector<MultipointStencilAuditEntry> entries;
    std::size_t restricted_failures = 0;
    std::size_t general_failures = 0;
    std::size_t general_pairs_without_target = 0;
    std::size_t target_harmonic_points_outside_faces = 0;
    std::size_t selected_harmonic_points_outside_faces = 0;
    double maximum_qtkq_condition = 0.0;
    double maximum_relative_reconstruction_defect = 0.0;
};

class MultipointFlux {
public:
    explicit MultipointFlux(const Mesh2D& mesh);

    [[nodiscard]] MultipointFluxes compute(MultipointFluxScheme scheme,
                                           std::span<const double> conductivity,
                                           std::span<const double> solution,
                                           const MultipointBoundaryData& boundary_data) const;

    // Compares the historical target+other restriction with the complete local
    // two-direction search used by the 2D specialization of the reference method.
    // It does not alter the flux used by compute().
    [[nodiscard]] MultipointStencilAudit
    audit_stencils(std::span<const double> conductivity,
                   const MultipointBoundaryData& boundary_data) const;

private:
    enum class BoundarySide {
        surface,
        basal,
        left,
        right,
    };

    struct InternalFace {
        std::size_t owner = 0;
        std::size_t neighbor = 0;
        Eigen::Vector2d center = Eigen::Vector2d::Zero();
        Eigen::Vector2d owner_normal = Eigen::Vector2d::Zero();
        double length = 0.0;
        double owner_distance = 0.0;
        double neighbor_distance = 0.0;
    };

    struct BoundaryFace {
        std::size_t cell = 0;
        Eigen::Vector2d center = Eigen::Vector2d::Zero();
        Eigen::Vector2d outward_normal = Eigen::Vector2d::Zero();
        double length = 0.0;
        BoundarySide side = BoundarySide::surface;
    };

    struct FaceReference {
        bool internal = false;
        std::size_t index = 0;
        bool owner_side = false;
    };

    struct Sample;
    struct OneSidedFlux;

    [[nodiscard]] Sample sample(std::size_t cell, const FaceReference& reference,
                                std::span<const double> conductivity,
                                const MultipointBoundaryData& boundary_data) const;
    [[nodiscard]] OneSidedFlux one_sided_flux(std::size_t cell, const FaceReference& target,
                                              std::span<const double> conductivity,
                                              const MultipointBoundaryData& boundary_data) const;

    std::vector<Eigen::Vector2d> cell_centers_;
    std::vector<InternalFace> internal_faces_;
    std::vector<BoundaryFace> boundary_faces_;
    std::vector<std::vector<FaceReference>> incident_faces_;
    std::vector<FaceReference> surface_faces_;
};
