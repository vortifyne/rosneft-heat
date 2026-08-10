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

add_executable(heat_solver src/main.cpp)
target_include_directories(heat_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(heat_solver PRIVATE fmt::fmt Eigen3::Eigen)

if (MSVC)
    target_compile_options(heat_solver PRIVATE /W4 /permissive-)
else()
    target_compile_options(heat_solver PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

enable_testing()
include(GoogleTest)

add_executable(unit_tests
    tests/test_umfpack.cpp
)

target_link_libraries(unit_tests PRIVATE
    Eigen3::Eigen
    SuiteSparse::UMFPACK_static
    GTest::gtest_main
)

if (NOT MSVC)
    target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

gtest_discover_tests(unit_tests)
