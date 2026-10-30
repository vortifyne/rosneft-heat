cmake_minimum_required(VERSION 3.28)

option(BUILD_TESTING "Build automated checks" OFF)
option(HEAT_BUILD_MODEL "Build the model forward and inverse problems" OFF)
option(HEAT_BUILD_BENCHMARKS "Build performance benchmarks" OFF)

if(BUILD_TESTING)
  list(APPEND VCPKG_MANIFEST_FEATURES tests)
endif()
if(HEAT_BUILD_BENCHMARKS)
  list(APPEND VCPKG_MANIFEST_FEATURES benchmarks)
endif()

project(heat_solver VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 CONFIG REQUIRED)
find_package(UMFPACK CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSc REQUIRED IMPORTED_TARGET PETSc petsc)
find_package(MPI REQUIRED COMPONENTS CXX)

function(heat_configure_target target)
  target_include_directories(${target} PUBLIC ${PROJECT_SOURCE_DIR}/src)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
  endif()
endfunction()

add_library(heat_numerics STATIC
  src/numerics/linear/umfpack_linear_solver.cpp
  src/numerics/linear/petsc_krylov_linear_solver.cpp
  src/numerics/nonlinear/nonlinear_solver.cpp
  src/numerics/time/bdf1.cpp
  src/numerics/time/time_integrator.cpp
)
heat_configure_target(heat_numerics)
target_link_libraries(heat_numerics
  PUBLIC Eigen3::Eigen
  PRIVATE SuiteSparse::UMFPACK_static PkgConfig::PETSc MPI::MPI_CXX
)

add_library(heat_opt STATIC EXCLUDE_FROM_ALL src/numerics/optimization/pounders_optimizer.cpp)
heat_configure_target(heat_opt)
target_link_libraries(heat_opt PUBLIC Eigen3::Eigen PRIVATE PkgConfig::PETSc MPI::MPI_CXX)

add_library(basin_physics STATIC
  src/basin/physics/easy_ro.cpp
  src/basin/physics/thermal_properties.cpp
)
heat_configure_target(basin_physics)

add_library(basin_data STATIC
  src/basin/basin_problem.cpp
  src/basin/io/basin_input.cpp
  src/basin/geometry/basin_geometry.cpp
  src/basin/geometry/basin_geometry_rdp.cpp
  src/basin/mesh/col_epoch_grid.cpp
  src/basin/mesh/col_field_transfer.cpp
  src/basin/mesh/col_grid.cpp
  src/basin/io/basin_binary.cpp
)
heat_configure_target(basin_data)
target_link_libraries(basin_data PUBLIC basin_physics)

add_library(basin_heat_forward STATIC
  src/basin/forward/basin_forward_solver.cpp
  src/basin/io/basin_forward_options.cpp
  src/basin/io/basin_forward_output.cpp
  src/basin/io/basin_statistics.cpp
  src/basin/discretization/col_flux.cpp
  src/basin/discretization/col_heat_system.cpp
  src/basin/postprocessing/energy_rates.cpp
)
heat_configure_target(basin_heat_forward)
target_link_libraries(basin_heat_forward PUBLIC heat_numerics basin_data)

add_executable(basin_forward_app src/apps/basin_forward/main.cpp)
set_target_properties(basin_forward_app PROPERTIES OUTPUT_NAME basin_forward_solver)
heat_configure_target(basin_forward_app)
target_link_libraries(basin_forward_app PRIVATE basin_heat_forward PkgConfig::PETSc MPI::MPI_CXX)

if(HEAT_BUILD_MODEL)
  add_library(model_heat_forward STATIC
    src/model/forward/model_heat_system.cpp
    src/model/forward/model_heat_forward_solver.cpp
  )
  heat_configure_target(model_heat_forward)
  target_link_libraries(model_heat_forward PUBLIC heat_numerics)

  add_library(model_heat_inverse STATIC
    src/model/inverse/model_heat_inverse_solver.cpp
    src/model/inverse/model_heat_observation_data.cpp
    src/model/inverse/model_heat_parameter_space.cpp
    src/model/inverse/model_heat_residual_evaluator.cpp
  )
  heat_configure_target(model_heat_inverse)
  target_link_libraries(model_heat_inverse PUBLIC model_heat_forward heat_opt)
endif()

if(BUILD_TESTING)
  enable_testing()
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)
  add_executable(heat_numerics_tests
    tests/numerics/algebra/test_sparse_matrix.cpp
    tests/numerics/algebra/test_vector.cpp
    tests/numerics/linear/test_umfpack_linear_solver.cpp
    tests/numerics/linear/test_petsc_krylov_linear_solver.cpp
    tests/numerics/nonlinear/test_nonlinear_solver.cpp
    tests/numerics/time/test_bdf1.cpp
    tests/numerics/time/test_time_integrator.cpp
    tests/numerics/optimization/test_pounders_optimizer.cpp
  )
  heat_configure_target(heat_numerics_tests)
  target_include_directories(heat_numerics_tests PRIVATE ${PROJECT_SOURCE_DIR}/tests)
  target_link_libraries(heat_numerics_tests PRIVATE
    heat_numerics heat_opt
    GTest::gtest_main PkgConfig::PETSc MPI::MPI_CXX
  )
  gtest_discover_tests(heat_numerics_tests PROPERTIES LABELS numerics)

  add_executable(basin_heat_tests
    tests/basin/geometry/test_basin_geometry.cpp
    tests/basin/mesh/test_col_epoch_grid.cpp
    tests/basin/mesh/test_col_field_transfer.cpp
    tests/basin/mesh/test_col_grid.cpp
    tests/basin/discretization/test_col_flux.cpp
    tests/basin/discretization/test_col_heat_system.cpp
    tests/basin/discretization/test_col_accuracy.cpp
    tests/basin/discretization/test_col_time.cpp
    tests/basin/physics/test_easy_ro.cpp
    tests/basin/physics/test_thermal_properties.cpp
    tests/basin/io/test_basin_forward_options.cpp
    tests/basin/io/test_basin_statistics.cpp
    tests/basin/forward/test_basin_forward_solver.cpp
  )
  heat_configure_target(basin_heat_tests)
  target_include_directories(basin_heat_tests PRIVATE ${PROJECT_SOURCE_DIR}/tests)
  target_link_libraries(basin_heat_tests PRIVATE
    basin_heat_forward
    GTest::gtest_main PkgConfig::PETSc MPI::MPI_CXX
  )
  gtest_discover_tests(basin_heat_tests PROPERTIES LABELS basin)

  if(HEAT_BUILD_MODEL)
    add_executable(model_heat_tests
      tests/model/forward/test_model_heat_forward_solver.cpp
      tests/model/forward/test_model_heat_system.cpp
      tests/model/inverse/test_model_heat_inverse_recovery.cpp
      tests/model/inverse/test_model_heat_inverse_solver.cpp
      tests/model/inverse/test_model_heat_observation_data.cpp
      tests/model/inverse/test_model_heat_parameter_space.cpp
      tests/model/inverse/test_model_heat_residual_evaluator.cpp
    )
    heat_configure_target(model_heat_tests)
    target_include_directories(model_heat_tests PRIVATE ${PROJECT_SOURCE_DIR}/tests)
    target_link_libraries(model_heat_tests PRIVATE
      model_heat_inverse GTest::gtest_main PkgConfig::PETSc MPI::MPI_CXX
    )
    gtest_discover_tests(model_heat_tests PROPERTIES LABELS model)
  endif()
endif()

if(HEAT_BUILD_BENCHMARKS)
  find_package(benchmark CONFIG REQUIRED)
  add_executable(benchmarks benchmarks/solvers.cpp)
  heat_configure_target(benchmarks)
  target_link_libraries(benchmarks PRIVATE basin_heat_forward benchmark::benchmark PkgConfig::PETSc MPI::MPI_CXX)
  target_compile_definitions(benchmarks PRIVATE
    "HEAT_BENCHMARK_COMPILER=\"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\""
  )
endif()
