cmake_minimum_required(VERSION 3.18)
project(IDP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED 20)

option(USE_APPROX_RESHARE "Enable APPROX RESHARE" OFF)
option(TRIPLE_VERIFY "Enable verification" ON)
option(TRIPLE_COLOR "Enable colored output" ON)
option(TRIPLE_ZERO "Process all-zero weight blocks like any other (OFF: skip them, reject all-zero filters)" ON)
option(TRIPLE_GPU "Enable GPU support" OFF)
option(TRIPLE_GPU_REVERSE "GPU uses reverse AB2" OFF)

set(TRIPLE_LOG_LEVEL 0 CACHE STRING "Set log level (0: All, 1: Debug and Error, 2: Error)")
set(TRIPLE_SEED 42 CACHE STRING "Sets the seed (-1: no seed)")
set(TRIPLE_BITLEN 32 CACHE STRING "Sets the BITLEN (default: 32)")
set(CONV_MAX_BATCH_SIZE 16 CACHE STRING "Sets the maximum batch size for GPU convolutions (default: 16)")

add_compile_options(-fPIC)
# add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

## Dependencies
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/deps;${CMAKE_PREFIX_PATH}")

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(emp-tool REQUIRED)
find_package(emp-ot REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(SEAL 4.1.2 REQUIRED)

if(TRIPLE_GPU)
    if(NOT CMAKE_CUDA_COMPILER)
        find_program(CMAKE_CUDA_COMPILER nvcc)
        message(STATUS "Using ${CMAKE_CUDA_COMPILER}")
    else()
        message(STATUS ${CMAKE_CUDA_COMPILER})
    endif()
    enable_language(CUDA)
    find_package(Troy REQUIRED)
    find_package(CUDAToolkit REQUIRED)

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")
endif()
# include_directories(${EIGEN3_INCLUDE_DIR}
#                     ${EMP-OT_INCLUDE_DIRS}
#                     ${EMP-TOOL_INCLUDE_DIRS})


set(ALL_SOURCE_FILES "")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)

add_custom_target(
    format
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMAND clang-format
        -i ${ALL_SOURCE_FILES}
        --verbose
)
