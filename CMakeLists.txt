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

if(HEAT_BUILD_MODEL_INVERSE)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(PETSc REQUIRED IMPORTED_TARGET PETSc petsc)
  find_package(MPI REQUIRED)
endif()

set(TQMesh_VERSION_MAJOR 1)
set(TQMesh_VERSION_MINOR 4)
set(TQMESH_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/tqmesh")
file(MAKE_DIRECTORY "${TQMESH_GENERATED_INCLUDE_DIR}")
configure_file(
  thirdparty/TQMesh/auxiliary/TQMeshConfig.h.in
  "${TQMESH_GENERATED_INCLUDE_DIR}/TQMeshConfig.h"
)

add_library(tqmesh_headers INTERFACE)
add_library(TQMesh::TQMesh ALIAS tqmesh_headers)
target_include_directories(tqmesh_headers SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/TQMesh/src/algorithm
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/TQMesh/src/utils
  ${TQMESH_GENERATED_INCLUDE_DIR}
)

add_library(heat_numerics STATIC
  src/linear/umfpack_linear_solver.cpp
  src/nonlinear/nonlinear_solver.cpp
  src/time/bdf1.cpp
  src/time/time_integrator.cpp
)

target_include_directories(heat_numerics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(heat_numerics
  PUBLIC Eigen3::Eigen
  PRIVATE SuiteSparse::UMFPACK_static
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

add_library(basin_heat STATIC
  src/basin/basin_forward_solver.cpp
  src/basin/basin_input.cpp
  src/basin/basin_state.cpp
  src/basin/forward_output.cpp
  src/basin/layered_mesh.cpp
  src/basin/mesh_motion.cpp
  src/basin/vtu_output.cpp
  src/discretization/heat_system.cpp
  src/discretization/least_squares_gradient.cpp
  src/mesh/mesh2d.cpp
  src/mesh/mesh_transfer.cpp
  src/physics/easy_ro.cpp
  src/physics/thermophysical_properties.cpp
)

target_include_directories(basin_heat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(basin_heat PUBLIC heat_numerics TQMesh::TQMesh)

if(MSVC)
  target_compile_options(basin_heat PRIVATE /W4 /permissive-)
else()
  target_compile_options(basin_heat PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
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

add_executable(heat_forward src/basin/main.cpp)
target_link_libraries(heat_forward PRIVATE basin_heat)

if(MSVC)
  target_compile_options(heat_forward PRIVATE /W4 /permissive-)
else()
  target_compile_options(heat_forward PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
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
  tests/test_umfpack_smoke.cpp
  tests/test_vector.cpp
)

target_link_libraries(model_unit_tests PRIVATE
  heat_numerics
  model_heat
  SuiteSparse::UMFPACK_static
  GTest::gtest_main
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
  tests/test_basin_forward_solver.cpp
  tests/test_easy_ro.cpp
  tests/test_heat_system.cpp
  tests/test_layered_mesh.cpp
  tests/test_least_squares_gradient.cpp
  tests/test_manufactured_heat_solution.cpp
  tests/test_mesh2d.cpp
  tests/test_mesh_transfer.cpp
  tests/test_thermophysical_properties.cpp
)

target_link_libraries(heat_unit_tests PRIVATE
  basin_heat
  GTest::gtest_main
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
