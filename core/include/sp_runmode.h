// Shannon-Prime: Runtime mode dispatcher
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Probes available silicon at startup and selects one of four operating
// modes that the rest of the engine branches on. See README.md for the
// product-level definitions.
//
//   FULL_PULSE     — every silicon path online (i9 + RTX + UHD + Optane + S22)
//   DESKTOP_SOLO   — phone detached, desktop handles drafting + residue
//   LEGACY         — no Optane reservoir, fall back to NVMe-tier prefetch
//   LONE_WOLF      — mobile build, runs the engine standalone on the phone
//
// The dispatcher is intentionally pure-C with no dependency on the engine
// core or the beast canyon orchestrator — both of those *consume* it.
// Probes themselves may delegate to platform-specific detectors that
// already exist (sp_hetero_detect_gpus, sp_optane_*).

#ifndef SP_RUNMODE_H
#define SP_RUNMODE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SP_MODE_LONE_WOLF    = 0,   // mobile only, no desktop fabric
    SP_MODE_LEGACY       = 1,   // desktop CPU/GPU, no Optane
    SP_MODE_DESKTOP_SOLO = 2,   // desktop with Optane, no phone
    SP_MODE_FULL_PULSE   = 3,   // everything online
} sp_run_mode_t;

typedef struct {
    bool has_cuda;          // discrete NVIDIA GPU present + CUDA runtime loadable
    bool has_l0_igpu;       // Intel iGPU detected (Level Zero or Vulkan vendor 0x8086)
    bool has_optane;        // Optane DAX device, or env var override
    bool has_sidecar;       // ADB heartbeat ACK from S22 within probe window
    bool is_mobile_host;    // build-time SP_TARGET_MOBILE — short-circuits to LONE_WOLF
    char optane_path[256];  // resolved reservoir path (empty if none)
} sp_hw_caps_t;

// Probe every detectable subsystem. Cheap (sub-second) — safe to call on
// every engine start. Side-effect-free except for log lines.
sp_hw_caps_t sp_probe_hardware(void);

// Cascade caps → mode. Honours the SP_RUN_MODE env var if set to one of:
//   "FULL_PULSE", "DESKTOP_SOLO", "LEGACY", "LONE_WOLF"
// Useful for forcing a specific path during test/bench.
sp_run_mode_t sp_select_mode(const sp_hw_caps_t *caps);

// Stable string for logs.
const char *sp_mode_name(sp_run_mode_t m);

// Print the mode + populated caps to stderr in a one-line summary.
void sp_log_mode(sp_run_mode_t m, const sp_hw_caps_t *caps);

// Standalone probes. Each is also useful for the bridge / health endpoint.
bool sp_probe_cuda(void);
bool sp_probe_l0_igpu(void);
bool sp_probe_optane(char *out_path, size_t out_path_len);
bool sp_probe_sidecar(int timeout_ms);

#ifdef __cplusplus
}
#endif

#endif // SP_RUNMODE_H
