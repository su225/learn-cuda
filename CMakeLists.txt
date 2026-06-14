cmake_minimum_required(VERSION 3.28)

project(cuda-learning LANGUAGES CXX CUDA)

set(CMAKE_NINJA_FORCE_RESPONSE_FILE OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_COMPILER "nvcc")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 120)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDAToolkit 13.1 REQUIRED)
message(STATUS "Found CUDA version: ${CUDAToolkit_VERSION}")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
)
add_subdirectory(src)

include(cmake/FetchGoogleTest.cmake)
enable_testing()
add_subdirectory(test)
