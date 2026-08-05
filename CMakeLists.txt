cmake_minimum_required(VERSION 3.28)

project(heat_solver
    VERSION 0.1.0
    LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(heat_solver src/main.cpp)

target_include_directories(heat_solver
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (MSVC)
    target_compile_options(heat_solver PRIVATE /W4 /permissive-)
else()
    target_compile_options(heat_solver PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()