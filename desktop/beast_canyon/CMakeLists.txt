# Shannon-Prime Beast Canyon Backend
# Heterogeneous MoE Orchestrator: Optane + AVX-512 + dual-GPU + sidecar
#
# Build options:
#   -DSP_WITH_CUDA=ON       — Enable NVIDIA CUDA dispatch
#   -DSP_WITH_VULKAN=ON     — Enable Vulkan dispatch (Intel Xe, AMD, etc.)
#   -DSP_WITH_LEVEL_ZERO=ON — Enable Intel Level Zero (UHD iGPU via Ring Bus)
#   -DSP_BEAST_STANDALONE=ON — Build standalone test binary

cmake_minimum_required(VERSION 3.18)

# Source files
set(SP_BEAST_SOURCES
    sp_optane.c
    sp_avx512_shredder.c
    sp_hetero_sync.c
    sp_beast_canyon.c
    sp_diagnostics_bc.c
    sp_shadow_steal.c
    sp_thermal_throttle.c
    sp_prefetch_telemetry.c
    sp_shredder_crt.c
    sp_level_zero.c
)

set(SP_BEAST_HEADERS
    sp_optane.h
    sp_avx512_shredder.h
    sp_hetero_sync.h
    sp_beast_canyon.h
    sp_diagnostics_bc.h
    sp_shadow_steal.h
    sp_thermal_throttle.h
    sp_prefetch_telemetry.h
    sp_shredder_crt.h
    sp_level_zero.h
)

# Library target
add_library(sp_beast_canyon STATIC ${SP_BEAST_SOURCES})

target_include_directories(sp_beast_canyon
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../core
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../crt
)

# Platform flags
if(MSVC)
    target_compile_options(sp_beast_canyon PRIVATE /arch:AVX512 /O2 /openmp)
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sp_beast_canyon PRIVATE
        -mavx512f -mavx512bw -mavx512dq -mavx512vl -mf16c -mfma
        -O3 -march=native -fopenmp
    )
    target_link_options(sp_beast_canyon PRIVATE -fopenmp)
endif()

# Windows-specific: link ws2_32 for sidecar socket
if(WIN32)
    target_link_libraries(sp_beast_canyon PRIVATE ws2_32)
endif()

# Linux-specific: link rt for clock_gettime, pthread
if(UNIX AND NOT APPLE)
    target_link_libraries(sp_beast_canyon PRIVATE rt pthread)
endif()

# CUDA runtime support — beast_canyon only uses cudart (no .cu compilation).
# This does NOT call enable_language(CUDA) which would require nvcc+MSVC.
# On MinGW, MSVC .lib files can't link directly (security cookie symbols),
# so we generate a MinGW import lib from the DLL at configure time.
option(SP_WITH_CUDA "Enable NVIDIA CUDA dispatch" OFF)
if(SP_WITH_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_compile_definitions(sp_beast_canyon PUBLIC SP_WITH_CUDA)
        target_include_directories(sp_beast_canyon PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

        if(MINGW)
            # MinGW can't link MSVC .lib — generate import lib from DLL
            set(_cudart_dll "${CUDAToolkit_BIN_DIR}/cudart64_${CUDAToolkit_VERSION_MAJOR}.dll")
            if(NOT EXISTS "${_cudart_dll}")
                # Try bin/x64/ (CUDA 13+ layout)
                set(_cudart_dll "${CUDAToolkit_BIN_DIR}/../bin/x64/cudart64_${CUDAToolkit_VERSION_MAJOR}.dll")
            endif()
            set(_mingw_cudart "${CMAKE_BINARY_DIR}/libcudart_mingw.a")
            if(EXISTS "${_cudart_dll}" AND NOT EXISTS "${_mingw_cudart}")
                set(_def "${CMAKE_BINARY_DIR}/cudart64.def")
                execute_process(COMMAND gendef "${_cudart_dll}"
                                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                                RESULT_VARIABLE _gendef_rc)
                file(GLOB _def_files "${CMAKE_BINARY_DIR}/cudart64*.def")
                if(_def_files)
                    list(GET _def_files 0 _def)
                    execute_process(COMMAND dlltool -d "${_def}" -l "${_mingw_cudart}"
                                    RESULT_VARIABLE _dlltool_rc)
                endif()
            endif()
            if(EXISTS "${_mingw_cudart}")
                target_link_libraries(sp_beast_canyon PUBLIC "${_mingw_cudart}")
                message(STATUS "Beast Canyon CUDA: ON (MinGW import lib from ${_cudart_dll})")
            else()
                message(WARNING "Beast Canyon: could not generate MinGW cudart import lib")
                target_link_libraries(sp_beast_canyon PUBLIC CUDA::cudart)
            endif()
        else()
            target_link_libraries(sp_beast_canyon PUBLIC CUDA::cudart)
            message(STATUS "Beast Canyon CUDA: ON (cudart from ${CUDAToolkit_LIBRARY_DIR})")
        endif()
    else()
        message(WARNING "Beast Canyon: SP_WITH_CUDA=ON but CUDAToolkit not found — CUDA disabled")
    endif()
endif()

# Vulkan support
option(SP_WITH_VULKAN "Enable Vulkan dispatch" OFF)
if(SP_WITH_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        target_compile_definitions(sp_beast_canyon PUBLIC SP_WITH_VULKAN)
        target_link_libraries(sp_beast_canyon PUBLIC Vulkan::Vulkan)
        message(STATUS "Beast Canyon Vulkan: ON (${Vulkan_LIBRARY})")
    else()
        message(WARNING "Beast Canyon: SP_WITH_VULKAN=ON but Vulkan not found — Vulkan disabled")
    endif()
endif()

# Level Zero support — dynamic loading, no SDK required at compile time.
# ze_loader.dll is loaded at runtime via LoadLibrary/dlopen.
# All L0 types are defined in sp_level_zero.h (vendored minimal subset).
# This option just sets the compile definition; no find_package needed.
option(SP_WITH_LEVEL_ZERO "Enable Intel Level Zero (UHD iGPU via Ring Bus)" OFF)
if(SP_WITH_LEVEL_ZERO)
    target_compile_definitions(sp_beast_canyon PUBLIC SP_WITH_LEVEL_ZERO)
    message(STATUS "Beast Canyon Level Zero: ON (dynamic loading at runtime)")
endif()

# Standalone test binary
option(SP_BEAST_STANDALONE "Build standalone test binary" OFF)
if(SP_BEAST_STANDALONE)
    add_executable(sp_beast_test test_beast_canyon.c)
    target_link_libraries(sp_beast_test PRIVATE sp_beast_canyon)
    if(WIN32)
        target_link_libraries(sp_beast_test PRIVATE ws2_32)
    endif()
endif()
