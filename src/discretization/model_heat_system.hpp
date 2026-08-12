#pragma once

#include "discretization/semi_discrete_system.hpp"
#include "mesh/regular_grid.hpp"
#include "model/model_heat_problem.hpp"

class ModelHeatSystem final : public SemiDiscreteSystem {
public:
    ModelHeatSystem(const RegularGrid& grid, const ModelHeatProblem& problem);

    Vector::Index size() const noexcept;

    void assemble_residual(double time, const Vector& solution, const Vector& solution_derivative,
                           Vector& residual) const override;

    void assemble_matrix(NonlinearMethod method, double time, const Vector& solution,
                         const Vector& solution_derivative, double derivative_shift,
                         SparseMatrix& matrix) const override;

private:
    Vector::Index cell_index(int ix, int iz) const noexcept;
    void check_vector_sizes(const Vector& solution, const Vector& solution_derivative) const;

    RegularGrid grid_;
    double thermal_conductivity_;
    double volumetric_heat_capacity_;
    double heat_production_;
    double surface_temperature_;
    double basal_heat_flux_;
};
