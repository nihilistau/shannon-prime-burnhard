# Shannon-Prime VHT2 - Hexagon DSP FastRPC scaffold (CMake).
# Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
#
# Forked from the Hexagon SDK 5.5.6.0 S22U sample CMakeLists. Topology
# (HLOS+Hexagon split, build_idl, link_custom_library, copy_binaries,
# runMainOnHexagonSim) preserved verbatim because it's the canonical
# Qualcomm-blessed FastRPC project shape; renaming targets is enough.

cmake_minimum_required(VERSION 3.14.3)

project(sp_hex C CXX ASM)

enable_language(ASM)

set(CMAKE_DEBUG_TARGET_PROPERTIES
    INCLUDE_DIRECTORIES
    COMPILE_DEFINITIONS
    POSITION_INDEPENDENT_CODE
    CONTAINER_SIZE_REQUIRED
    LIB_VERSION
)

include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)

set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/src_app/
    ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/
    ${CMAKE_CURRENT_BINARY_DIR}/
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
    ${HEXAGON_SDK_ROOT}/rtos/qurt/
    ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc/
    ${HEXAGON_SDK_ROOT}/utils/examples
)

if(${OS_TYPE} MATCHES "HLOS")

    # ---------------------------------------------------------------------
    # HLOS Dynamic Stub Library (libsp_hex.so on the ARM side)
    # ---------------------------------------------------------------------
    add_library(sp_hex SHARED ${CMAKE_CURRENT_BINARY_DIR}/sp_hex_stub)

    # SP math core path (needed for the engine-side stub's fp16 helpers
    # and band config). Matches the DSP-side definition below.
    set(SP_CORE ${CMAKE_CURRENT_SOURCE_DIR}/../../../core)

    # ---------------------------------------------------------------------
    # HLOS Executable (sp_hex on the device, built from the *_device CMake target)
    # ---------------------------------------------------------------------
    add_executable(sp_hex_device
        ${CMAKE_CURRENT_SOURCE_DIR}/src_app/sp_hex_main
        ${CMAKE_CURRENT_SOURCE_DIR}/src_app/sp_hex_ext
        ${CMAKE_CURRENT_SOURCE_DIR}/../shannon_prime_hexagon
        ${SP_CORE}/shannon_prime
        ${SP_CORE}/shannon_prime_pe
        ${SP_CORE}/shannon_prime_sqfree
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils
    )

    set_target_properties(sp_hex_device PROPERTIES OUTPUT_NAME "sp_hex")

    target_compile_definitions(sp_hex_device PUBLIC VERIFY_PRINT_ERROR
                                                    SP_HEXAGON_FASTRPC)
    target_include_directories(sp_hex_device PRIVATE ${SP_CORE}
                                                     ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_compile_options(sp_hex_device PRIVATE -Wno-unused-function)

    link_options(sp_hex_device)

    include_directories(${common_incs})

    # qaic generates sp_hex_stub.{c,h} and sp_hex.h from the IDL.
    build_idl(inc/sp_hex.idl sp_hex)

    choose_dsprpc(${DSP_TYPE} dsprpc)

    link_custom_library(sp_hex_device rpcmem)
    link_custom_library(sp_hex_device getopt_custom)

    link_custom_library(sp_hex        ${dsprpc})
    link_custom_library(sp_hex_device ${dsprpc})

    add_dependencies(sp_hex_device sp_hex)

    target_link_libraries(sp_hex_device sp_hex)

    copy_binaries(sp_hex_device)

else()

    # ---------------------------------------------------------------------
    # Hexagon Dynamic Skel Library (libsp_hex_skel.so on the DSP side)
    # ---------------------------------------------------------------------
    # SP math core sources (relative to shannon-prime repo root, two levels up
    # from this CMakeLists.txt). Linked into the DSP skel so IDL methods can
    # call sp_band_quantize / sp_band_dequantize / sp_vht2_forward_f32 directly
    # — no scalar reference duplication on the DSP side.
    set(SP_CORE ${CMAKE_CURRENT_SOURCE_DIR}/../../../core)

    set(srcs
        ${CMAKE_CURRENT_BINARY_DIR}/sp_hex_skel
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_hex_imp
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_hex_kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_hex_kernels_hvx
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_hvx_logits
        ${SP_CORE}/shannon_prime
        ${SP_CORE}/shannon_prime_pe
        ${SP_CORE}/shannon_prime_sqfree
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils
    )

    add_library(sp_hex_skel SHARED ${srcs})

    # SP core needs its own header dir on the include path.
    # compat/ holds the dmaWrapper shim headers for Phase 7B DMA probe.
    target_include_directories(sp_hex_skel PRIVATE ${SP_CORE}
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/compat)

    # The SP core has static-inline helpers that are unused when only
    # shannon_prime.c + _pe.c + _sqfree.c are linked (the cauchy and
    # modelpack TUs that consume them are host-only). Suppress the
    # resulting -Wunused-function-as-error so the DSP build stays green.
    #
    # -mhvx is required for the HVX intrinsics in sp_hex_kernels_hvx.c.
    # The default v69 build doesn't enable HVX automatically; with -mhvx
    # the toolchain defines __HVX__ and the HVX-gated code paths activate.
    target_compile_options(sp_hex_skel PRIVATE -Wno-unused-function -mhvx -mhvx-ieee-fp)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")

    include_directories(${common_incs})

    build_idl("inc/sp_hex.idl" sp_hex_skel incs)

    # ---------------------------------------------------------------------
    # Hexagon Simulator Test Executable
    # ---------------------------------------------------------------------
    add_library(sp_hex_q SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src_app/sp_hex_main
        ${CMAKE_CURRENT_SOURCE_DIR}/src_app/sp_hex_ext
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_hex_imp
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_hex_kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_hex_kernels_hvx
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/sp_hvx_logits
        ${SP_CORE}/shannon_prime
        ${SP_CORE}/shannon_prime_pe
        ${SP_CORE}/shannon_prime_sqfree
    )
    target_include_directories(sp_hex_q PRIVATE ${SP_CORE}
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/compat)
    target_compile_options(sp_hex_q PRIVATE -Wno-unused-function -mhvx -mhvx-ieee-fp)

    add_library(sp_hex_skel_static STATIC ${srcs})
    target_include_directories(sp_hex_skel_static PRIVATE ${SP_CORE}
        ${CMAKE_CURRENT_SOURCE_DIR}/src_dsp/compat)
    target_compile_options(sp_hex_skel_static PRIVATE -Wno-unused-function -mhvx -mhvx-ieee-fp)

    include_directories(${common_incs})

    link_custom_library(sp_hex_q rpcmem)

    add_dependencies(sp_hex_skel sp_hex_q)
    add_dependencies(sp_hex_q sp_hex_skel_static)
    target_link_libraries(sp_hex_q sp_hex_skel_static)

    copy_binaries(sp_hex_q sp_hex_skel)

    runMainOnHexagonSim(sp_hex_q)

endif()

# vim: set noet fenc=utf-8 ff=unix ft=cmake :
