cmake_minimum_required(VERSION 3.18)

# Options to control what gets built
option(BUILD_SERVER "Build CUDA server" ON)
option(BUILD_CLIENT "Build client" ON)
option(BUILD_LOCAL "Build local raytracer" ON)

# Determine if we need CUDA
if(BUILD_SERVER OR BUILD_LOCAL)
    project(raytracer-cuda LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 17)
else()
    project(raytracer-cuda LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)

# Find SDL2 - try CONFIG first (vcpkg), then MODULE (system)
find_package(SDL2 CONFIG QUIET)
if(NOT SDL2_FOUND)
    find_package(SDL2 MODULE REQUIRED)
    # For system SDL2, we need to set up the targets manually
    if(NOT TARGET SDL2::SDL2)
        add_library(SDL2::SDL2 INTERFACE IMPORTED)
        set_target_properties(SDL2::SDL2 PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${SDL2_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${SDL2_LIBRARIES}"
        )
    endif()
    if(NOT TARGET SDL2::SDL2main)
        add_library(SDL2::SDL2main INTERFACE IMPORTED)
        set_target_properties(SDL2::SDL2main PROPERTIES
            INTERFACE_LINK_LIBRARIES "${SDL2MAIN_LIBRARY}"
        )
    endif()
endif()

# Find zstd - try CONFIG first (vcpkg), then system libraries
find_package(zstd CONFIG QUIET)
if(NOT zstd_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ZSTD libzstd)
        if(ZSTD_FOUND)
            add_library(zstd::libzstd_shared INTERFACE IMPORTED)
            set_target_properties(zstd::libzstd_shared PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIRS}"
                INTERFACE_LINK_LIBRARIES "${ZSTD_LIBRARIES}"
                INTERFACE_LINK_DIRECTORIES "${ZSTD_LIBRARY_DIRS}"
            )
        endif()
    endif()
    
    # Fallback: try to find system zstd
    if(NOT ZSTD_FOUND)
        find_library(ZSTD_LIBRARY zstd)
        find_path(ZSTD_INCLUDE_DIR zstd.h)
        
        if(ZSTD_LIBRARY AND ZSTD_INCLUDE_DIR)
            add_library(zstd::libzstd_shared INTERFACE IMPORTED)
            set_target_properties(zstd::libzstd_shared PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}"
                INTERFACE_LINK_LIBRARIES "${ZSTD_LIBRARY}"
            )
            set(ZSTD_FOUND TRUE)
        endif()
    endif()
    
    if(NOT ZSTD_FOUND)
        message(FATAL_ERROR "zstd library not found. On Ubuntu, install with: sudo apt install libzstd-dev")
    endif()
endif()

# Common source files (non-CUDA)
set(COMMON_SOURCES
    src/network_protocol.cpp
)

# Server-specific build (requires CUDA)
if(BUILD_SERVER)
    # Set CUDA architectures (adjust based on your GPU)
    set(CMAKE_CUDA_ARCHITECTURES 75) # Change this to match your GPU architecture
    
    # Find CUDA toolkit
    find_package(CUDAToolkit REQUIRED)
    
    # Build server executable
    add_executable(raytracer-server
        src/server_main.cpp
        src/renderer.cu
        ${COMMON_SOURCES}
    )
    
    # Set properties for CUDA compilation
    set_target_properties(raytracer-server PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # Link libraries for server
    target_link_libraries(raytracer-server PRIVATE
        SDL2::SDL2main
        SDL2::SDL2
        CUDA::cudart
        CUDA::curand
        zstd::libzstd_shared
    )
    
    # Add include directories for server
    target_include_directories(raytracer-server PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        src
    )
    
    # Set CUDA flags
    target_compile_options(raytracer-server PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --extended-lambda
            --expt-relaxed-constexpr
        >
    )
    
    # Add server-specific preprocessor definition
    target_compile_definitions(raytracer-server PRIVATE SERVER_BUILD)
endif()

# Client build (no CUDA required)
if(BUILD_CLIENT)
    add_executable(raytracer-client
        src/client_main.cpp
        ${COMMON_SOURCES}
    )
    
    # Link libraries for client
    target_link_libraries(raytracer-client PRIVATE
        SDL2::SDL2main
        SDL2::SDL2
        zstd::libzstd_shared
    )
    
    # Add include directories for client
    target_include_directories(raytracer-client PRIVATE
        src
    )
    
    # Add client-specific preprocessor definition
    target_compile_definitions(raytracer-client PRIVATE CLIENT_BUILD)
endif()

# Local raytracer build (original, requires CUDA)
if(BUILD_LOCAL)
    # Set CUDA architectures
    set(CMAKE_CUDA_ARCHITECTURES 75)
    
    # Find CUDA toolkit
    find_package(CUDAToolkit REQUIRED)
    
    # Build local executable (original)
    add_executable(raytracer-local
        src/main.cpp
        src/renderer.cu
    )
    
    # Set properties for CUDA compilation
    set_target_properties(raytracer-local PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # Link libraries
    target_link_libraries(raytracer-local PRIVATE
        SDL2::SDL2main
        SDL2::SDL2
        CUDA::cudart
        CUDA::curand
    )
    
    # Add include directories
    target_include_directories(raytracer-local PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        src
    )
    
    # Set CUDA flags
    target_compile_options(raytracer-local PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --extended-lambda
            --expt-relaxed-constexpr
        >
    )
endif()

# Platform-specific socket libraries
if(WIN32)
    if(BUILD_SERVER)
        target_link_libraries(raytracer-server PRIVATE ws2_32)
    endif()
    if(BUILD_CLIENT)
        target_link_libraries(raytracer-client PRIVATE ws2_32)
    endif()
else()
    # Unix-like systems don't need extra libraries for sockets
endif()

# Usage instructions
message(STATUS "Build configuration:")
message(STATUS "  BUILD_SERVER: ${BUILD_SERVER}")
message(STATUS "  BUILD_CLIENT: ${BUILD_CLIENT}")
message(STATUS "  BUILD_LOCAL: ${BUILD_LOCAL}")
message(STATUS "")
message(STATUS "To build only client on Ubuntu laptop:")
message(STATUS "  cmake -DBUILD_SERVER=OFF -DBUILD_LOCAL=OFF ..")
message(STATUS "")
message(STATUS "To build only server on Windows PC:")
message(STATUS "  cmake -DBUILD_CLIENT=OFF -DBUILD_LOCAL=OFF ..")
message(STATUS "")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "On Ubuntu/Debian, install dependencies with:")
    message(STATUS "  sudo apt update")
    message(STATUS "  sudo apt install build-essential cmake libsdl2-dev libzstd-dev pkg-config")
endif()