cmake_minimum_required(VERSION 3.28)

project(heat_solver VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(fmt CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(UMFPACK CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)

# Find petsc via pkg-config
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSc REQUIRED IMPORTED_TARGET PETSc petsc)
find_package(MPI REQUIRED)

target_include_directories(heat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(
  heat_core PUBLIC Eigen3::Eigen PRIVATE SuiteSparse::UMFPACK_static
)

if(MSVC)
  target_compile_options(heat_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(
  heat_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion
  )
endif()

add_library(heat_core STATIC
    src/discretization/model_heat_system.cpp
    src/forward/model_heat_forward_solver.cpp
    src/linear/umfpack_linear_solver.cpp
    src/nonlinear/nonlinear_solver.cpp
    src/time/bdf1.cpp
    src/time/time_integrator.cpp
)

target_include_directories(heat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(heat_core PUBLIC Eigen3::Eigen PRIVATE SuiteSparse::UMFPACK_static)

if (MSVC)
    target_compile_options(heat_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(heat_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

add_executable(heat_solver src/main.cpp)
target_link_libraries(heat_solver PRIVATE heat_core fmt::fmt)

if(MSVC)
  target_compile_options(heat_solver PRIVATE /W4 /permissive-)
else()
  target_compile_options(
  heat_solver PRIVATE -Wall -Wextra -Wpedantic -Wconversion
)
endif()

enable_testing()
include(GoogleTest)

add_executable(unit_tests
    tests/test_bdf1.cpp
    tests/test_model_heat_forward_solver.cpp
    tests/test_model_heat_system.cpp
    tests/test_nonlinear_solver.cpp
    tests/test_sparse_matrix.cpp
    tests/test_time_integrator.cpp
    tests/test_umfpack_linear_solver.cpp
    tests/test_umfpack_smoke.cpp
    tests/test_vector.cpp
    tests/test_petsc_tao_punders.cpp
)

target_link_libraries(unit_tests PRIVATE
    heat_core
    SuiteSparse::UMFPACK_static
    GTest::gtest_main
    PkgConfig::PETSc
    MPI::MPI_CXX
)

if(NOT MSVC)
  target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

gtest_discover_tests(unit_tests)

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
