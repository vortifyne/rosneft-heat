#include "discretization/model_heat_system.hpp"
#include "linear/umfpack_linear_solver.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {

Lithotype make_lithotype(double thermal_conductivity = 2.0, double density = 2.0,
                         double specific_heat = 3.0, double heat_production = 5.0) {
    return {
        .thermal_conductivity = thermal_conductivity,
        .density = density,
        .specific_heat = specific_heat,
        .heat_production = heat_production,
    };
}

ModelHeatParameters make_parameters(double surface_temperature = 10.0,
                                    double basal_heat_flux = 3.0) {
    return {.surface_temperature = surface_temperature, .basal_heat_flux = basal_heat_flux};
}

void expect_vector_near(const Vector& actual, const Vector& expected, double tolerance = 1e-14) {
    ASSERT_EQ(actual.size(), expected.size());
    for (Vector::Index index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance);
    }
}

void expect_matrix_coefficient(const SparseMatrix& matrix, Vector::Index row, Vector::Index column,
                               double expected) {
    EXPECT_DOUBLE_EQ(matrix.coefficient(row, column), expected)
        << "coefficient (" << row << ", " << column << ')';
}

double realistically_scaled_steady_profile_error(int cell_count) {
    constexpr double kDepth = 5000.0;
    constexpr double kThermalConductivity = 2.0;
    constexpr double kHeatProduction = 1.5e-6;
    constexpr double kSurfaceTemperature = 288.15;
    constexpr double kBasalHeatFlux = 0.06;
    const RegularGrid grid(1, cell_count, 1000.0, 0.0, kDepth);
    const ModelHeatSystem system(
        grid, make_lithotype(kThermalConductivity, 2500.0, 800.0, kHeatProduction),
        make_parameters(kSurfaceTemperature, kBasalHeatFlux));
    Vector zero(system.size());
    zero.set_zero();
    Vector residual;
    system.assemble_residual(0.0, zero, zero, residual);
    SparseMatrix matrix(system.size(), system.size(), SparseStorageOrder::csc);
    system.assemble_matrix(NonlinearMethod::newton, 0.0, zero, zero, 0.0, matrix);
    UmfpackLinearSolver solver;
    Vector temperature;

    const LinearSolveResult solve_result = solver.solve(
        matrix, -residual, temperature,
        {.relative_tolerance = 1e-12, .absolute_tolerance = 1e-12, .max_iterations = 1});
    if (solve_result.status != LinearSolveStatus::converged) {
        throw std::runtime_error("Failed to solve the steady heat profile");
    }

    double maximum_error = 0.0;
    for (int iz = 0; iz < cell_count; ++iz) {
        const double depth = (static_cast<double>(iz) + 0.5) * grid.dz;
        const double expected =
            kSurfaceTemperature +
            (((kBasalHeatFlux + (kHeatProduction * kDepth)) / kThermalConductivity) * depth) -
            ((kHeatProduction / (2.0 * kThermalConductivity)) * depth * depth);
        maximum_error = std::max(maximum_error, std::abs(temperature[iz] - expected));
    }
    return maximum_error;
}

TEST(RegularGridTest, ProvidesCellGeometryForFiniteVolumeAssembly) {
    const RegularGrid grid(4, 5, 8.0, 2.0, 12.0);

    EXPECT_EQ(grid.cell_count(), 20);
    EXPECT_DOUBLE_EQ(grid.dx, 2.0);
    EXPECT_DOUBLE_EQ(grid.dz, 2.0);
    EXPECT_DOUBLE_EQ(grid.cell_area, 4.0);
    EXPECT_DOUBLE_EQ(grid.inv_cell_area, 0.25);
}

TEST(RegularGridTest, RejectsInvalidGeometry) {
    EXPECT_THROW(RegularGrid(0, 1, 1.0, 0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(RegularGrid(1, -1, 1.0, 0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(RegularGrid(1, 1, 0.0, 0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(RegularGrid(1, 1, 1.0, 1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(RegularGrid(1, 1, 1.0, 0.0, std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
}

TEST(ModelHeatSystemTest, AssemblesConservativeFaceResidualAndBoundaryTerms) {
    const RegularGrid grid(2, 2, 2.0, 0.0, 2.0);
    const ModelHeatSystem system(grid, make_lithotype(), make_parameters());
    const Vector solution{11.0, 13.0, 17.0, 19.0};
    const Vector solution_derivative{1.0, 2.0, 3.0, 4.0};
    Vector residual;

    system.assemble_residual(0.0, solution, solution_derivative, residual);

    expect_vector_near(residual, Vector{-11.0, 11.0, 18.0, 32.0});
}

TEST(ModelHeatSystemTest, KeepsConstantTemperatureAtSteadyStateWithoutSourcesOrFluxes) {
    const RegularGrid grid(3, 2, 6.0, 0.0, 4.0);
    const ModelHeatSystem system(grid, make_lithotype(2.0, 2500.0, 800.0, 0.0),
                                 make_parameters(300.0, 0.0));
    Vector solution(system.size());
    solution.set_constant(300.0);
    Vector solution_derivative(system.size());
    solution_derivative.set_zero();
    Vector residual;

    system.assemble_residual(1.0, solution, solution_derivative, residual);

    EXPECT_DOUBLE_EQ(residual.infinity_norm(), 0.0);
}

TEST(ModelHeatSystemTest, AssemblesExpectedMatrixDirectlyInEitherStorageOrder) {
    const RegularGrid grid(2, 2, 2.0, 0.0, 2.0);
    const ModelHeatSystem system(grid, make_lithotype(), make_parameters());
    const Vector solution{11.0, 13.0, 17.0, 19.0};
    const Vector solution_derivative{1.0, 2.0, 3.0, 4.0};

    for (const SparseStorageOrder storage_order :
         {SparseStorageOrder::csr, SparseStorageOrder::csc}) {
        SparseMatrix matrix(0, 0, storage_order);

        system.assemble_matrix(NonlinearMethod::newton, 0.0, solution, solution_derivative, 0.5,
                               matrix);

        EXPECT_EQ(matrix.rows(), 4);
        EXPECT_EQ(matrix.cols(), 4);
        EXPECT_EQ(matrix.storage_order(), storage_order);
        EXPECT_TRUE(matrix.is_compressed());
        EXPECT_EQ(matrix.nonzero_count(), 12);
        expect_matrix_coefficient(matrix, 0, 0, 11.0);
        expect_matrix_coefficient(matrix, 1, 1, 11.0);
        expect_matrix_coefficient(matrix, 2, 2, 7.0);
        expect_matrix_coefficient(matrix, 3, 3, 7.0);
        expect_matrix_coefficient(matrix, 0, 1, -2.0);
        expect_matrix_coefficient(matrix, 1, 0, -2.0);
        expect_matrix_coefficient(matrix, 0, 2, -2.0);
        expect_matrix_coefficient(matrix, 2, 0, -2.0);
        expect_matrix_coefficient(matrix, 1, 2, 0.0);
    }
}

TEST(ModelHeatSystemTest, PicardAndNewtonMatricesCoincideForConstantCoefficients) {
    const RegularGrid grid(2, 1, 2.0, 0.0, 1.0);
    const ModelHeatSystem system(grid, make_lithotype(), make_parameters());
    const Vector solution{11.0, 13.0};
    const Vector solution_derivative{1.0, 2.0};
    SparseMatrix picard(0, 0, SparseStorageOrder::csc);
    SparseMatrix newton(0, 0, SparseStorageOrder::csc);

    system.assemble_matrix(NonlinearMethod::picard, 0.0, solution, solution_derivative, 0.25,
                           picard);
    system.assemble_matrix(NonlinearMethod::newton, 0.0, solution, solution_derivative, 0.25,
                           newton);

    for (Vector::Index row = 0; row < system.size(); ++row) {
        for (Vector::Index column = 0; column < system.size(); ++column) {
            EXPECT_DOUBLE_EQ(picard.coefficient(row, column), newton.coefficient(row, column));
        }
    }
}

TEST(ModelHeatSystemTest, MatchesARealisticallyScaledSteadyTemperatureProfile) {
    EXPECT_LT(realistically_scaled_steady_profile_error(100), 3e-4);
}

TEST(ModelHeatSystemTest, HasSecondOrderSpatialConvergenceWithNeumannBoundaryFlux) {
    // This steady manufactured profile removes temporal error. For uniform refinements the
    // observed order is p = log(error_h/error_h2) / log(2), so an error ratio of four means p = 2.
    const double coarse_error = realistically_scaled_steady_profile_error(25);
    const double medium_error = realistically_scaled_steady_profile_error(50);
    const double fine_error = realistically_scaled_steady_profile_error(100);

    EXPECT_NEAR(coarse_error / medium_error, 4.0, 1e-6);
    EXPECT_NEAR(medium_error / fine_error, 4.0, 1e-5);
}

TEST(ModelHeatSystemTest, RejectsInvalidMaterialAndVectorData) {
    const RegularGrid grid(1, 1, 1.0, 0.0, 1.0);
    EXPECT_THROW(ModelHeatSystem(grid, make_lithotype(0.0), make_parameters()),
                 std::invalid_argument);
    EXPECT_THROW(ModelHeatSystem(grid, make_lithotype(1.0, 0.0), make_parameters()),
                 std::invalid_argument);
    EXPECT_THROW(ModelHeatSystem(grid, make_lithotype(1.0, 1.0, 0.0), make_parameters()),
                 std::invalid_argument);
    EXPECT_THROW(ModelHeatSystem(
                     grid, make_lithotype(1.0, 1.0, 1.0, std::numeric_limits<double>::infinity()),
                     make_parameters()),
                 std::invalid_argument);

    const ModelHeatSystem system(grid, make_lithotype(), make_parameters());
    Vector residual;
    EXPECT_THROW(system.assemble_residual(0.0, Vector{}, Vector{}, residual),
                 std::invalid_argument);
}

} // namespace
