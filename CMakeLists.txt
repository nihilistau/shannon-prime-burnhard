# Shannon-Prime BurnHard — Distributed Inference Engine
#
# Mono-repo build system. Consolidates:
#   core/      — SP math core (VHT2, Möbius, CRT, band quant)
#   desktop/   — Engine + Beast Canyon + CRT multi-GPU + CUDA + Vulkan + L0
#   mobile/    — Adreno + Hexagon + QNN HTP + Halide + freethedsp
#   bridge/    — llama.cpp + ComfyUI integration
#   vendor/    — ggml + cpp-httplib
#
# Paths adapted from shannon-prime-engine/CMakeLists.txt for the
# burnhard mono-repo directory layout.

cmake_minimum_required(VERSION 3.14)
project(shannon-prime-burnhard
    VERSION 0.1.0
    DESCRIPTION "Shannon-Prime BurnHard distributed inference engine"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --------------------------------------------------------------------
# Options
# --------------------------------------------------------------------
option(SP_BUILD_TESTS           "Build tests"                           ON)
option(SP_BUILD_CLI             "Build the sp-engine CLI binary"        ON)
option(SP_WITH_CUDA             "Enable CUDA backend for KV kernels"    OFF)
option(SP_WITH_VULKAN           "Enable Vulkan backend"                 OFF)
option(SP_WITH_HEXAGON          "Enable Hexagon DSP backend"            ON)
option(SP_HEXAGON_FASTRPC       "Link real Qualcomm FastRPC libs"       OFF)
option(SP_WITH_QNN              "Enable Qualcomm QNN HTP backend"       OFF)
option(SP_FP8                   "Enable FP8 (E4M3) KV cache quant"     OFF)
option(SP_FP4                   "Enable FP4 (Blackwell) KV cache quant" OFF)
option(SP_WITH_BEAST            "Enable Beast Canyon heterogeneous MoE" OFF)
option(SP_BEAST_CUDA            "Beast Canyon: CUDA GPU detect"         OFF)
option(SP_BEAST_VULKAN          "Beast Canyon: Vulkan iGPU detect"      OFF)
option(SP_BEAST_LEVEL_ZERO      "Beast Canyon: Intel Level Zero iGPU"   OFF)
option(SP_WITH_BRIDGE           "Build bridge: heartbeat + residue codec" ON)
option(SP_TARGET_MOBILE         "Build mobile-only target (LONE_WOLF)"  OFF)

# --------------------------------------------------------------------
# Core shannon-prime math library (platform-independent)
# --------------------------------------------------------------------
set(SP_CORE_SRC
    core/src/shannon_prime.c
    core/src/shannon_prime_sqfree.c
    core/src/shannon_prime_cauchy.c
    core/src/shannon_prime_modelpack.c
    core/src/shannon_prime_pe.c
    core/src/sp_runmode.c
    core/src/burnhard_config.c
    core/src/shredder_q4k_inflight.c
    core/src/burnhard_residue_matmul.c
    desktop/crt/sp_crt.c
    desktop/crt/sp_crt_dispatch.c
    desktop/crt/sp_theoretical.c)

if(NOT SP_WITH_CUDA)
    list(APPEND SP_CORE_SRC desktop/crt/sp_crt_cuda_stub.c)
endif()
if(SP_WITH_VULKAN OR SP_BEAST_VULKAN)
    list(APPEND SP_CORE_SRC desktop/crt/sp_crt_vulkan.c)
else()
    list(APPEND SP_CORE_SRC desktop/crt/sp_crt_vulkan_stub.c)
endif()

add_library(shannon_prime_core STATIC ${SP_CORE_SRC})
# AVX-512 for the in-flight Shredder (and any future AVX-512 paths in core).
# The Shredder gates its vector kernel on __AVX512F__ and falls back to a
# scalar path otherwise, so this flag is for performance, not correctness.
if(MSVC)
    target_compile_options(shannon_prime_core PRIVATE /arch:AVX512)
else()
    target_compile_options(shannon_prime_core PRIVATE -mavx512f)
endif()
target_include_directories(shannon_prime_core PUBLIC
    core/include
    desktop/crt
    desktop/beast_canyon
    vendor/ggml/include)
target_compile_features(shannon_prime_core PUBLIC c_std_11)
# sp_runmode probes use TCP sockets and (on Linux) dlopen for CUDA detection.
if(WIN32)
    target_link_libraries(shannon_prime_core PUBLIC ws2_32)
elseif(NOT ANDROID)
    target_link_libraries(shannon_prime_core PUBLIC dl)
endif()
if(SP_WITH_CUDA)
    target_compile_definitions(shannon_prime_core PUBLIC SP_WITH_CUDA)
endif()
if(SP_TARGET_MOBILE)
    # Short-circuits sp_probe_hardware() to LONE_WOLF; skip desktop-only
    # backends (CUDA/Beast/CRT) further down in this file.
    target_compile_definitions(shannon_prime_core PUBLIC SP_TARGET_MOBILE=1)
endif()

if(SP_WITH_VULKAN OR SP_BEAST_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        target_include_directories(shannon_prime_core PUBLIC ${Vulkan_INCLUDE_DIRS})
        target_link_libraries(shannon_prime_core PUBLIC Vulkan::Vulkan)
        target_compile_definitions(shannon_prime_core PUBLIC SP_WITH_VULKAN)
    endif()
endif()
if(MSVC)
    target_compile_options(shannon_prime_core PRIVATE /W3)
else()
    target_compile_options(shannon_prime_core PRIVATE -Wall -Wextra -Wno-unused-function)
endif()

# --------------------------------------------------------------------
# CUDA backend
# --------------------------------------------------------------------
if(SP_WITH_CUDA)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(_SP_CUDA_ARCHS "75")
        if(SP_FP8)
            list(APPEND _SP_CUDA_ARCHS "89")
        endif()
        if(SP_FP4)
            list(APPEND _SP_CUDA_ARCHS "120")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES ${_SP_CUDA_ARCHS})
    endif()

    set(_SP_CUDA_SOURCES
        desktop/cuda/shannon_prime_cuda.cu
        desktop/cuda/shannon_prime_sqfree.cu
        desktop/cuda/shannon_prime_hier.cu
        desktop/crt/sp_crt_cuda.cu
        desktop/crt/sp_shor_kernel.cu)
    if(SP_FP8 AND EXISTS ${CMAKE_SOURCE_DIR}/desktop/cuda/shannon_prime_fp8.cu)
        list(APPEND _SP_CUDA_SOURCES desktop/cuda/shannon_prime_fp8.cu)
    endif()

    add_library(shannon_prime_cuda_backend STATIC ${_SP_CUDA_SOURCES})
    target_include_directories(shannon_prime_cuda_backend PUBLIC
        desktop/cuda
        desktop/crt
        core/include)
    target_link_libraries(shannon_prime_cuda_backend PUBLIC shannon_prime_core)
    set_target_properties(shannon_prime_cuda_backend PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON)
    if(SP_FP8)
        target_compile_definitions(shannon_prime_cuda_backend PUBLIC SP_FP8=1)
    endif()
    if(SP_FP4)
        target_compile_definitions(shannon_prime_cuda_backend PUBLIC SP_FP4=1)
    endif()
endif()

# --------------------------------------------------------------------
# Hexagon DSP backend
# --------------------------------------------------------------------
if(SP_WITH_HEXAGON)
    set(_SP_HEXAGON_SOURCES mobile/hexagon/shannon_prime_hexagon.c)
    add_library(shannon_prime_hexagon_backend STATIC ${_SP_HEXAGON_SOURCES})
    target_include_directories(shannon_prime_hexagon_backend PUBLIC
        mobile/hexagon
        core/include)
    target_link_libraries(shannon_prime_hexagon_backend PUBLIC shannon_prime_core)
    target_compile_features(shannon_prime_hexagon_backend PUBLIC c_std_11)
    if(SP_HEXAGON_FASTRPC)
        target_compile_definitions(shannon_prime_hexagon_backend PUBLIC SP_HEXAGON_FASTRPC=1)
    endif()
endif()

# --------------------------------------------------------------------
# QNN HTP backend
# --------------------------------------------------------------------
if(SP_WITH_QNN)
    if(NOT QAIRT_ROOT)
        message(FATAL_ERROR "SP_WITH_QNN=ON requires QAIRT_ROOT")
    endif()
    set(_SP_QNN_SOURCES
        mobile/qnn/sp_qnn.c
        mobile/qnn/sp_llama_qnn.c)
    add_library(shannon_prime_qnn_backend STATIC ${_SP_QNN_SOURCES})
    target_include_directories(shannon_prime_qnn_backend PUBLIC
        mobile/qnn
        ${QAIRT_ROOT}/include/QNN
        ${QAIRT_ROOT}/include
        core/include)
    target_link_libraries(shannon_prime_qnn_backend PUBLIC shannon_prime_core)
    if(NOT WIN32)
        target_link_libraries(shannon_prime_qnn_backend PUBLIC dl)
    endif()
endif()

# --------------------------------------------------------------------
# Beast Canyon: heterogeneous MoE orchestrator
# --------------------------------------------------------------------
if(SP_WITH_BEAST)
    # Beast Canyon has its own CMakeLists.txt
    if(EXISTS ${CMAKE_SOURCE_DIR}/desktop/beast_canyon/CMakeLists.txt)
        add_subdirectory(desktop/beast_canyon ${CMAKE_BINARY_DIR}/beast_canyon)
        target_link_libraries(sp_beast_canyon PUBLIC shannon_prime_core)
    else()
        # Build from source list
        file(GLOB _BC_SOURCES desktop/beast_canyon/*.c desktop/beast_canyon/*.h)
        add_library(sp_beast_canyon STATIC ${_BC_SOURCES})
        target_include_directories(sp_beast_canyon PUBLIC
            desktop/beast_canyon
            desktop/crt
            core/include
            vendor/ggml/include)
        target_link_libraries(sp_beast_canyon PUBLIC shannon_prime_core)
        if(SP_BEAST_CUDA)
            target_compile_definitions(sp_beast_canyon PUBLIC SP_WITH_CUDA)
        endif()
        if(SP_BEAST_VULKAN)
            target_compile_definitions(sp_beast_canyon PUBLIC SP_WITH_VULKAN)
        endif()
        if(SP_BEAST_LEVEL_ZERO)
            target_compile_definitions(sp_beast_canyon PUBLIC SP_WITH_LEVEL_ZERO)
        endif()
    endif()
endif()

# --------------------------------------------------------------------
# Bridge: heartbeat + residue migration codec (desktop ↔ phone fabric)
# --------------------------------------------------------------------
if(SP_WITH_BRIDGE)
    add_library(sp_bridge STATIC
        bridge/protocol/sp_bridge_codec.c
        bridge/heartbeat/sp_bridge_heartbeat.c)
    target_include_directories(sp_bridge PUBLIC
        bridge/protocol
        bridge/heartbeat
        desktop/beast_canyon
        core/include)
    target_link_libraries(sp_bridge PUBLIC shannon_prime_core)
    if(WIN32)
        target_link_libraries(sp_bridge PUBLIC ws2_32)
    elseif(NOT ANDROID)
        target_link_libraries(sp_bridge PUBLIC pthread)
    endif()
    target_compile_features(sp_bridge PUBLIC c_std_11)
endif()

# --------------------------------------------------------------------
# ggml (vendor submodule)
# --------------------------------------------------------------------
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(SP_WITH_CUDA)
    set(GGML_CUDA TRUE CACHE BOOL "" FORCE)
endif()
if(SP_WITH_VULKAN)
    set(GGML_VULKAN TRUE CACHE BOOL "" FORCE)
endif()
if(EXISTS ${CMAKE_SOURCE_DIR}/vendor/ggml/CMakeLists.txt)
    add_subdirectory(vendor/ggml EXCLUDE_FROM_ALL)
endif()

# --------------------------------------------------------------------
# Engine library
# --------------------------------------------------------------------
add_library(sp_engine STATIC
    desktop/engine/src/engine.cpp
    desktop/engine/src/kv_cache.cpp
    desktop/engine/src/gdn_state.cpp
    desktop/engine/src/gguf_loader.cpp
    desktop/engine/src/vocab.cpp
    desktop/engine/src/tokenizer.cpp
    desktop/engine/src/llama_weights.cpp
    desktop/engine/src/forward.cpp
    desktop/engine/src/burn_moe_op.cpp
    desktop/engine/src/prime_pe.cpp
    desktop/engine/src/http_server.cpp
    desktop/engine/src/sp_tensor.cpp
    core/src/sp_quant.cpp
    desktop/engine/src/sp_kernels_cpu.cpp
    desktop/engine/src/sp_threadpool.cpp
    desktop/engine/src/forward_native.cpp
    desktop/engine/src/forward_native_context.cpp
    desktop/engine/src/speculative_oracle.cpp)

# cpp-httplib (if present in vendor)
if(EXISTS ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib/httplib.cpp)
    target_sources(sp_engine PRIVATE vendor/cpp-httplib/httplib.cpp)
    target_include_directories(sp_engine PUBLIC vendor/cpp-httplib)
endif()

target_include_directories(sp_engine PUBLIC
    desktop/engine/src
    core/include
    desktop/crt
    desktop/beast_canyon
    desktop/cuda
    mobile/hexagon
    mobile/qnn)
if(WIN32)
    target_link_libraries(sp_engine PUBLIC ws2_32 crypt32)
endif()
target_link_libraries(sp_engine PUBLIC shannon_prime_core)
if(TARGET ggml)
    target_link_libraries(sp_engine PUBLIC ggml)
endif()

# Conditional backend linkage
if(SP_WITH_HEXAGON)
    target_link_libraries(sp_engine PUBLIC shannon_prime_hexagon_backend)
    target_compile_definitions(sp_engine PUBLIC SP_ENGINE_WITH_HEXAGON=1)
endif()
if(SP_WITH_QNN)
    target_link_libraries(sp_engine PUBLIC shannon_prime_qnn_backend)
    target_compile_definitions(sp_engine PUBLIC SP_ENGINE_WITH_QNN=1)
    target_sources(sp_engine PRIVATE desktop/engine/src/qnn_bin_driver.cpp)
endif()
if(SP_WITH_BEAST)
    target_link_libraries(sp_engine PUBLIC sp_beast_canyon)
    target_compile_definitions(sp_engine PUBLIC SP_ENGINE_WITH_BEAST=1)
    if(SP_WITH_BRIDGE)
        # Bridge headers wrap the sidecar transport; only needed when both
        # are on. The compile def gates the heartbeat hook in sp_beast_canyon.c.
        target_link_libraries(sp_beast_canyon PUBLIC sp_bridge)
        target_compile_definitions(sp_beast_canyon PUBLIC SP_WITH_BRIDGE=1)
    endif()
endif()
if(SP_WITH_CUDA)
    target_link_libraries(sp_engine PUBLIC shannon_prime_cuda_backend)
    target_compile_definitions(sp_engine PUBLIC SP_ENGINE_WITH_CUDA=1)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(sp_engine PUBLIC CUDA::cudart)
endif()
target_compile_features(sp_engine PUBLIC cxx_std_17)

# --------------------------------------------------------------------
# CLI binary
# --------------------------------------------------------------------
if(SP_BUILD_CLI)
    add_executable(sp-engine desktop/engine/src/cli/main.cpp)
    target_link_libraries(sp-engine PRIVATE sp_engine)
    if(MSVC)
        target_link_options(sp-engine PRIVATE "/STACK:33554432")
    endif()
endif()

# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------
if(SP_BUILD_TESTS)
    enable_testing()
    # Core math tests
    if(EXISTS ${CMAKE_SOURCE_DIR}/tests/core/test_core.c)
        add_executable(test_core tests/core/test_core.c)
        target_link_libraries(test_core PRIVATE shannon_prime_core)
        add_test(NAME core COMMAND test_core)
    endif()
    # Session-1 BurnHard Shredder + Garner round-trip verifier.
    if(EXISTS ${CMAKE_SOURCE_DIR}/tools/shred_verify.c)
        add_executable(shred_verify tools/shred_verify.c)
        target_link_libraries(shred_verify PRIVATE shannon_prime_core)
        target_include_directories(shred_verify PRIVATE
            ${CMAKE_SOURCE_DIR}/core/include)
        if(MSVC)
            target_compile_options(shred_verify PRIVATE /arch:AVX512)
        else()
            target_compile_options(shred_verify PRIVATE -mavx512f)
        endif()
        add_test(NAME shred_verify COMMAND shred_verify)
    endif()
    # Session-2 residue matmul verifier — proves the residue path matches
    # stock fp32 matmul to within 1e-5 relative error per cell.
    if(EXISTS ${CMAKE_SOURCE_DIR}/tools/residue_matmul_verify.c)
        add_executable(residue_matmul_verify tools/residue_matmul_verify.c)
        target_link_libraries(residue_matmul_verify PRIVATE shannon_prime_core)
        target_include_directories(residue_matmul_verify PRIVATE
            ${CMAKE_SOURCE_DIR}/core/include)
        if(MSVC)
            target_compile_options(residue_matmul_verify PRIVATE /arch:AVX512)
        else()
            target_compile_options(residue_matmul_verify PRIVATE -mavx512f -lm)
        endif()
        add_test(NAME residue_matmul_verify COMMAND residue_matmul_verify)
    endif()
    # Session-2.5 BURN_MOE expert FFN verifier — exercises the residue path
    # against a real Qwen3.6 expert tensor (gate / up / down dequant from
    # the Q4_K GGUF). The last math gate before the chat-path wiring.
    if(EXISTS ${CMAKE_SOURCE_DIR}/tools/burn_moe_verify.c)
        add_executable(burn_moe_verify tools/burn_moe_verify.c)
        target_link_libraries(burn_moe_verify PRIVATE shannon_prime_core ggml)
        target_include_directories(burn_moe_verify PRIVATE
            ${CMAKE_SOURCE_DIR}/core/include
            ${CMAKE_SOURCE_DIR}/vendor/ggml/include)
        if(MSVC)
            target_compile_options(burn_moe_verify PRIVATE /arch:AVX512)
        else()
            target_compile_options(burn_moe_verify PRIVATE -mavx512f -lm)
        endif()
    endif()
endif()

# --------------------------------------------------------------------
# Config summary
# --------------------------------------------------------------------
message(STATUS "")
message(STATUS "Shannon-Prime BurnHard ${PROJECT_VERSION}")
message(STATUS "  build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA         : ${SP_WITH_CUDA}")
message(STATUS "  Vulkan       : ${SP_WITH_VULKAN}")
message(STATUS "  Hexagon      : ${SP_WITH_HEXAGON} (FastRPC=${SP_HEXAGON_FASTRPC})")
message(STATUS "  QNN HTP      : ${SP_WITH_QNN}")
message(STATUS "  FP8          : ${SP_FP8}")
message(STATUS "  Beast Canyon : ${SP_WITH_BEAST}")
message(STATUS "    CUDA       : ${SP_BEAST_CUDA}")
message(STATUS "    Vulkan     : ${SP_BEAST_VULKAN}")
message(STATUS "    Level Zero : ${SP_BEAST_LEVEL_ZERO}")
message(STATUS "  CLI          : ${SP_BUILD_CLI}")
message(STATUS "  Tests        : ${SP_BUILD_TESTS}")
message(STATUS "")
