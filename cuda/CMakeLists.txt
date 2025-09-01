cmake_minimum_required(VERSION 3.18)
project(raytracer-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architectures (adjust based on your GPU)
# Common ones: 50, 52, 60, 61, 70, 75, 80, 86, 89, 90
set(CMAKE_CUDA_ARCHITECTURES 75) # Change this to match your GPU architecture

# Find SDL2 via vcpkg
find_package(SDL2 CONFIG REQUIRED)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Build executable
add_executable(raytracer
    src/main.cpp
    src/renderer.cu
)

# Set properties for CUDA compilation
set_target_properties(raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link SDL2
target_link_libraries(raytracer PRIVATE
    SDL2::SDL2main
    SDL2::SDL2
)

# Link CUDA libraries
target_link_libraries(raytracer PRIVATE
    CUDA::cudart
    CUDA::curand  # Add curand for random number generation
)

# Add include directories
# The src directory contains all your CUDA headers (vec3.cuh, ray.cuh, etc.)
target_include_directories(raytracer PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    src
)

# Set CUDA flags for better debugging and optimization
target_compile_options(raytracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
    >
)