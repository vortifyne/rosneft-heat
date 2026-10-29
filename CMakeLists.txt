cmake_minimum_required(VERSION 3.28)

project(heat_solver VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HEAT_BUILD_MODEL_INVERSE "Build the model inverse problem and PETSc tests" OFF)

find_package(Eigen3 CONFIG REQUIRED)
find_package(UMFPACK CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(PETSc REQUIRED IMPORTED_TARGET PETSc petsc)
find_package(MPI REQUIRED)

add_library(heat_numerics STATIC
  src/linear/umfpack_linear_solver.cpp
  src/linear/petsc_krylov_linear_solver.cpp
  src/nonlinear/nonlinear_solver.cpp
  src/time/bdf1.cpp
  src/time/time_integrator.cpp
)

target_include_directories(heat_numerics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(heat_numerics
  PUBLIC Eigen3::Eigen
  PRIVATE SuiteSparse::UMFPACK_static PkgConfig::PETSc MPI::MPI_CXX
)

if(MSVC)
  target_compile_options(heat_numerics PRIVATE /W4 /permissive-)
else()
  target_compile_options(heat_numerics PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

add_library(model_heat STATIC
  src/model/model_heat_system.cpp
  src/model/model_heat_forward_solver.cpp
)

target_include_directories(model_heat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(model_heat PUBLIC heat_numerics)

if(MSVC)
  target_compile_options(model_heat PRIVATE /W4 /permissive-)
else()
  target_compile_options(model_heat PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

add_library(heat_physics STATIC
  src/physics/easy_ro.cpp
  src/physics/thermophysical_properties.cpp
)
target_include_directories(heat_physics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(basin_data STATIC
  src/basin/basin_input.cpp
  src/basin/basin_geometry.cpp
)
target_include_directories(basin_data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(basin_data PUBLIC heat_physics)

if(MSVC)
  target_compile_options(heat_physics PRIVATE /W4 /permissive-)
  target_compile_options(basin_data PRIVATE /W4 /permissive-)
else()
  target_compile_options(heat_physics PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
  target_compile_options(basin_data PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

add_library(basin_column STATIC
  src/basin/basin_geometry_rdp.cpp
  src/basin/column_epoch_geometry.cpp
  src/basin/column_epoch_grid.cpp
  src/basin/column_field_transfer.cpp
  src/basin/column_basin_solver.cpp
  src/basin/column_basin_binary.cpp
  src/basin/nonmatching_column_grid.cpp
  src/discretization/multipoint_flux_expression.cpp
  src/discretization/nonmatching_column_flux.cpp
  src/discretization/nonmatching_column_heat_system.cpp
  src/discretization/nonmatching_column_transient_system.cpp
)

target_include_directories(basin_column PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(basin_column PUBLIC heat_numerics basin_data)

if(MSVC)
  target_compile_options(basin_column PRIVATE /W4 /permissive-)
else()
  target_compile_options(basin_column PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

if(HEAT_BUILD_MODEL_INVERSE)
  add_library(heat_optimization STATIC
    src/optimization/pounders_optimizer.cpp
  )

  target_include_directories(heat_optimization PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(heat_optimization
    PUBLIC Eigen3::Eigen
    PRIVATE PkgConfig::PETSc MPI::MPI_CXX
  )

  if(MSVC)
    target_compile_options(heat_optimization PRIVATE /W4 /permissive-)
  else()
    target_compile_options(heat_optimization PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
  endif()

  add_library(model_heat_inverse STATIC
    src/inverse/model_heat_inverse_solver.cpp
    src/inverse/model_heat_observation_data.cpp
    src/inverse/model_heat_parameter_space.cpp
    src/inverse/model_heat_residual_evaluator.cpp
  )

  target_include_directories(model_heat_inverse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(model_heat_inverse PUBLIC model_heat heat_optimization)

  if(MSVC)
    target_compile_options(model_heat_inverse PRIVATE /W4 /permissive-)
  else()
    target_compile_options(model_heat_inverse PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
  endif()
endif()

add_executable(column_heat_forward src/basin/column_main.cpp)
target_link_libraries(column_heat_forward PRIVATE basin_column PkgConfig::PETSc MPI::MPI_CXX)

if(MSVC)
  target_compile_options(column_heat_forward PRIVATE /W4 /permissive-)
else()
  target_compile_options(column_heat_forward PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

add_executable(nonmatching_discretization_verification
  tools/nonmatching_discretization_verification.cpp)
target_link_libraries(nonmatching_discretization_verification PRIVATE basin_column)

add_executable(nonmatching_geometry_audit tools/nonmatching_geometry_audit.cpp)
target_link_libraries(nonmatching_geometry_audit PRIVATE basin_column)

add_executable(nonmatching_fixed_epoch_verification
  tools/nonmatching_fixed_epoch_verification.cpp)
target_link_libraries(nonmatching_fixed_epoch_verification
  PRIVATE basin_column PkgConfig::PETSc MPI::MPI_CXX)

add_executable(compare_nonmatching_cell_fields
  tools/compare_nonmatching_cell_fields.cpp)

if(MSVC)
  target_compile_options(nonmatching_discretization_verification PRIVATE /W4 /permissive-)
  target_compile_options(nonmatching_geometry_audit PRIVATE /W4 /permissive-)
  target_compile_options(nonmatching_fixed_epoch_verification PRIVATE /W4 /permissive-)
  target_compile_options(compare_nonmatching_cell_fields PRIVATE /W4 /permissive-)
else()
  target_compile_options(nonmatching_discretization_verification
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
  target_compile_options(nonmatching_geometry_audit PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
  target_compile_options(nonmatching_fixed_epoch_verification
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
  target_compile_options(compare_nonmatching_cell_fields
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

enable_testing()
include(GoogleTest)

add_executable(model_unit_tests
  tests/test_bdf1.cpp
  tests/test_model_heat_forward_solver.cpp
  tests/test_model_heat_system.cpp
  tests/test_nonlinear_solver.cpp
  tests/test_sparse_matrix.cpp
  tests/test_time_integrator.cpp
  tests/test_umfpack_linear_solver.cpp
  tests/test_petsc_krylov_linear_solver.cpp
  tests/test_umfpack_smoke.cpp
  tests/test_vector.cpp
)

target_link_libraries(model_unit_tests PRIVATE
  heat_numerics
  model_heat
  SuiteSparse::UMFPACK_static
  GTest::gtest_main
  PkgConfig::PETSc MPI::MPI_CXX
)

if(HEAT_BUILD_MODEL_INVERSE)
  target_sources(model_unit_tests PRIVATE
    tests/test_model_heat_inverse_recovery.cpp
    tests/test_model_heat_inverse_solver.cpp
    tests/test_model_heat_observation_data.cpp
    tests/test_model_heat_parameter_space.cpp
    tests/test_model_heat_residual_evaluator.cpp
    tests/test_petsc_tao_punders.cpp
    tests/test_pounders_optimizer.cpp
  )
  target_link_libraries(model_unit_tests PRIVATE
    model_heat_inverse
    heat_optimization
    PkgConfig::PETSc
    MPI::MPI_CXX
  )
endif()

if(NOT MSVC)
  target_compile_options(model_unit_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

gtest_discover_tests(model_unit_tests)

add_executable(heat_unit_tests
  tests/test_easy_ro.cpp
  tests/test_basin_geometry.cpp
  tests/test_column_epoch_geometry.cpp
  tests/test_column_field_transfer.cpp
  tests/test_column_basin_solver.cpp
  tests/test_manufactured_heat_solution.cpp
  tests/test_nonmatching_column_grid.cpp
  tests/test_nonmatching_column_transient_system.cpp
  tests/test_thermophysical_properties.cpp
)

target_link_libraries(heat_unit_tests PRIVATE
  basin_column
  GTest::gtest_main
  PkgConfig::PETSc MPI::MPI_CXX
)

if(NOT MSVC)
  target_compile_options(heat_unit_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

gtest_discover_tests(heat_unit_tests)

add_executable(benchmarks
  benchmarks/linear_search.cpp
)

target_link_libraries(benchmarks PRIVATE
  benchmark::benchmark
  benchmark::benchmark_main
)

if(NOT MSVC)
  target_compile_options(benchmarks PRIVATE -Wall -Wextra -Wpedantic)
endif()
