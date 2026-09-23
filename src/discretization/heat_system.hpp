#pragma once

#include "discretization/least_squares_gradient.hpp"
#include "discretization/semi_discrete_system.hpp"
#include "mesh/mesh2d.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <vector>

struct HeatBoundaryConditions {
    std::function<double(Point2D, double)> surface_temperature;
    std::function<double(Point2D, double)> basal_heat_flux;
};

class HeatSystem final : public SemiDiscreteSystem {
public:
    HeatSystem(const Mesh2D& mesh, std::span<const double> thermal_conductivity,
               std::span<const double> volumetric_heat_capacity,
               std::span<const double> heat_production, HeatBoundaryConditions boundary_conditions);

    [[nodiscard]] Vector::Index size() const noexcept;

    void assemble_residual(double time, const Vector& solution, const Vector& solution_derivative,
                           Vector& residual) const override;

    void assemble_matrix(NonlinearMethod method, double time, const Vector& solution,
                         const Vector& solution_derivative, double derivative_shift,
                         SparseMatrix& matrix) const override;

private:
    struct InternalFace {
        std::size_t owner;
        std::size_t neighbor;
        double length;
        double normal_distance;
        double owner_distance;
        double neighbor_distance;
        Eigen::Vector2d correction;
    };

    struct BoundaryFace {
        std::size_t cell;
        Point2D center;
        double length;
        double normal_distance;
        Eigen::Vector2d correction;
    };

    struct ResidualAssembly;
    struct MatrixAssembly;

    template <typename Assembly>
    void assemble(Assembly& assembly, double time, const Vector& solution,
                  const Vector& solution_derivative) const;

    void assemble_accumulation(ResidualAssembly& assembly, double time, const Vector& solution,
                               const Vector& solution_derivative) const;
    void assemble_accumulation(MatrixAssembly& assembly, double time, const Vector& solution,
                               const Vector& solution_derivative) const;
    void assemble_internal_diffusion(ResidualAssembly& assembly, double time,
                                     const Vector& solution,
                                     const Vector& solution_derivative) const;
    void assemble_internal_diffusion(MatrixAssembly& assembly, double time, const Vector& solution,
                                     const Vector& solution_derivative) const;
    void assemble_source(ResidualAssembly& assembly, double time, const Vector& solution,
                         const Vector& solution_derivative) const;
    void assemble_source(MatrixAssembly& assembly, double time, const Vector& solution,
                         const Vector& solution_derivative) const;
    void assemble_surface_boundary(ResidualAssembly& assembly, double time, const Vector& solution,
                                   const Vector& solution_derivative) const;
    void assemble_surface_boundary(MatrixAssembly& assembly, double time, const Vector& solution,
                                   const Vector& solution_derivative) const;
    void assemble_basal_boundary(ResidualAssembly& assembly, double time, const Vector& solution,
                                 const Vector& solution_derivative) const;
    void assemble_basal_boundary(MatrixAssembly& assembly, double time, const Vector& solution,
                                 const Vector& solution_derivative) const;
    void assemble_lateral_boundaries(ResidualAssembly& assembly, double time,
                                     const Vector& solution,
                                     const Vector& solution_derivative) const;
    void assemble_lateral_boundaries(MatrixAssembly& assembly, double time, const Vector& solution,
                                     const Vector& solution_derivative) const;

    void assemble_orthogonal_diffusion_residual(ResidualAssembly& assembly,
                                                const Vector& solution) const;
    void assemble_nonorthogonal_correction_residual(ResidualAssembly& assembly) const;
    void assemble_orthogonal_diffusion_matrix(MatrixAssembly& assembly) const;
    void assemble_nonorthogonal_correction_matrix(MatrixAssembly& assembly) const;

    [[nodiscard]] double face_conductivity(const InternalFace& face) const noexcept;
    void check_vector_sizes(const Vector& solution, const Vector& solution_derivative) const;
    void prepare_gradients(double time, const Vector& solution) const;

    std::vector<double> thermal_conductivity_;
    std::vector<double> volumetric_heat_capacity_;
    std::vector<double> heat_production_;
    HeatBoundaryConditions boundary_conditions_;

    std::vector<double> inverse_cell_area_;
    std::vector<InternalFace> internal_faces_;
    std::vector<BoundaryFace> surface_faces_;
    std::vector<BoundaryFace> basal_faces_;
    std::size_t lateral_face_count_ = 0;

    std::optional<LeastSquaresGradient> gradient_reconstruction_;
    mutable std::vector<double> surface_temperatures_;
    mutable std::vector<LeastSquaresGradient::Gradient> gradients_;
};
