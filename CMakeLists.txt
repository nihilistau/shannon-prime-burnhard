cmake_minimum_required(VERSION 3.16)
project(burnhard C)

set(CMAKE_C_STANDARD 11)

# ── Core library ──────────────────────────────────────────────────────
add_library(burnhard_core STATIC
    core/src/burnhard_config.c
    core/src/burnhard_reservoir.c
    core/src/burnhard_init.c
)
target_include_directories(burnhard_core PUBLIC
    core/include
    bridge/include
)

# ── Bridge library ────────────────────────────────────────────────────
add_library(burnhard_bridge STATIC
    bridge/src/bridge_telemetry.c
)
target_include_directories(burnhard_bridge PUBLIC
    bridge/include
    core/include
)
if(WIN32)
    target_link_libraries(burnhard_bridge PRIVATE ws2_32)
else()
    target_link_libraries(burnhard_bridge PRIVATE pthread)
endif()

# Core depends on bridge (init starts the heartbeat)
target_link_libraries(burnhard_core PRIVATE burnhard_bridge)

# ── Test harness ──────────────────────────────────────────────────────
add_executable(burnhard_test tests/test_init.c)
target_link_libraries(burnhard_test PRIVATE burnhard_core burnhard_bridge)
if(WIN32)
    target_link_libraries(burnhard_test PRIVATE ws2_32)
endif()
