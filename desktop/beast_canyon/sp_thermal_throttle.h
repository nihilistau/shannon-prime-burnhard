// Shannon-Prime Beast Canyon: Thermal-Aware Homeostatic Throttle
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// ============================================================================
//  THERMAL-ENTROPY BRIDGE
// ============================================================================
//
// This module links the mathematical entropy of the model to the literal
// kinetic energy of the hardware. Shannon entropy in the KV cache drives
// speculation aggressiveness (tau), and silicon temperature is the physical
// manifestation of that computational work. When the hardware heats up,
// the math must back off — not because of arbitrary policy, but because
// the system is approaching a thermodynamic limit.
//
// The homeostatic feedback loop:
//
//   1. Shannon-Prime compresses KV cache with tau controlling quality.
//   2. Lower tau = more aggressive compression = more speculative decoding.
//   3. More speculation = more compute = more heat.
//   4. More heat = higher tau = back off speculation.
//   5. System finds equilibrium: maximum throughput at sustainable temp.
//
// This is not a thermostat. A thermostat is bang-bang (on/off). This is a
// homeostatic regulator — continuous, proportional, with three operating
// zones that map directly to the information-theoretic quality ladder.
//
// ============================================================================
//  THREE-ZONE STRATEGY
// ============================================================================
//
//  ZONE_COOL   (<70C)  — Full send. Hardware is cold, tau=0.50 baseline.
//                         Maximum speculation depth, maximum throughput.
//                         The silicon is begging for work.
//
//  ZONE_NOMINAL (70-85C) — Balanced. tau=0.75-0.85.
//                         The system is warm but sustainable. The homeostatic
//                         formula applies: tau = tau_base + sigma * max(0, T - T_limit).
//                         This is the steady-state operating regime — most of
//                         the time, the engine lives here.
//
//  ZONE_HOT    (>85C)  — Conservation mode. tau=0.95+.
//                         Speculation is nearly eliminated. The engine preserves
//                         quality at the cost of throughput. At T_critical (90C),
//                         tau snaps to 1.0 — no compression, no speculation,
//                         pure safety. The silicon lives to fight another token.
//
// The sigma parameter (0.02) controls the slope of the back-off curve.
// At T_limit (75C), each degree above costs 0.02 tau. By 85C, that's
// +0.20 tau on top of the base — enough to substantially reduce speculation
// without killing throughput entirely.
//
// Sensor sources:
//   - NVML (nvidia-ml) for GPU junction temperature + fan RPM
//   - Level-Zero Sysman for Intel iGPU / CPU package temperature
//   - Both are optional; graceful degradation if neither is available
//

#ifndef SP_THERMAL_THROTTLE_H
#define SP_THERMAL_THROTTLE_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>  // fmaxf, fminf

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
//  Constants
// ---------------------------------------------------------------------------

// Comfort zone threshold (Celsius). Below this, no tau adjustment.
#define SP_THERMAL_T_LIMIT       75.0f

// Survival mode threshold (Celsius). Above this, tau snaps to 1.0.
#define SP_THERMAL_T_CRITICAL    90.0f

// Back-off intensity: tau penalty per degree above T_LIMIT.
// At 85C: penalty = 0.02 * 10 = 0.20 tau added to base.
#define SP_THERMAL_SIGMA         0.02f

// Sensor polling interval in milliseconds.
// 500ms is fast enough to track thermal transients without
// burning cycles on sensor reads.
#define SP_THERMAL_POLL_INTERVAL_MS  500

// ---------------------------------------------------------------------------
//  Enums
// ---------------------------------------------------------------------------

// Thermal operating zone — maps directly to speculation strategy.
typedef enum {
    // < 70C. Full send. Aggressive speculation, tau ~0.50.
    SP_THERMAL_ZONE_COOL = 0,

    // 70-85C. Balanced. Homeostatic formula active, tau ~0.75-0.85.
    SP_THERMAL_ZONE_NOMINAL = 1,

    // > 85C. Conservation. Minimal speculation, tau ~0.95-1.00.
    SP_THERMAL_ZONE_HOT = 2
} sp_thermal_zone_t;

// ---------------------------------------------------------------------------
//  Structs
// ---------------------------------------------------------------------------

// Snapshot of thermal sensor readings at a point in time.
typedef struct {
    float    cpu_temp;       // CPU package temperature (Celsius), -1 if unavailable
    float    gpu_temp;       // GPU junction temperature (Celsius), -1 if unavailable
    float    max_temp;       // max(cpu_temp, gpu_temp) — the constraint that matters
    uint32_t fan_rpm;        // GPU fan RPM, 0 if unavailable or passive cooling
    uint64_t timestamp_us;   // microsecond timestamp of this reading
} sp_thermal_reading_t;

// Homeostatic regulator state. One per engine instance.
typedef struct {
    // Configuration — set at init, stable thereafter
    float    t_limit;        // Temperature threshold for back-off (default: 75C)
    float    t_critical;     // Temperature threshold for survival mode (default: 90C)
    float    sigma;          // Back-off slope: tau penalty per degree (default: 0.02)
    float    tau_base;       // Base tau when cool — the "resting heart rate"

    // Live state — updated by sp_thermal_poll()
    sp_thermal_zone_t       current_zone;
    sp_thermal_reading_t    last_reading;

    // Telemetry
    uint64_t total_polls;        // Number of sensor polls since init
    uint64_t zone_transitions;   // Number of zone changes (cool→nominal, etc.)

    // Sensor availability flags — detected at init, checked before reads
    bool     nvml_available;         // NVIDIA Management Library (GPU temp + fan)
    bool     l0_sysman_available;    // Intel Level-Zero System Management (CPU/iGPU)
} sp_thermal_homeostat_t;

// ---------------------------------------------------------------------------
//  Function declarations
// ---------------------------------------------------------------------------

// Initialize the thermal homeostat. Probes for NVML and L0 Sysman.
// Sets t_limit, t_critical, sigma to defaults. tau_base is the caller's
// chosen resting tau (typically 0.50 for aggressive, 0.75 for balanced).
// Returns 0 on success, -1 if no sensor backend is available (non-fatal;
// the engine can still run, it just won't have thermal awareness).
int sp_thermal_init(sp_thermal_homeostat_t* ctx, float tau_base);

// Release any sensor handles (NVML shutdown, L0 Sysman cleanup).
void sp_thermal_free(sp_thermal_homeostat_t* ctx);

// Poll current temperatures from available sensor backends.
// Updates ctx->last_reading and ctx->current_zone.
// Returns 0 on success, -1 if all sensor reads failed.
int sp_thermal_poll(sp_thermal_homeostat_t* ctx);

// Calculate the thermally-adjusted tau given a base tau.
// This is the full-path version that may re-poll sensors if the last
// reading is stale (older than SP_THERMAL_POLL_INTERVAL_MS).
//
// The homeostatic formula:
//   tau = tau_base + sigma * max(0, T_current - T_limit)
//   clamped to [tau_base, 0.98] in normal operation.
//   Returns 1.0 if T_current > T_critical (survival mode).
float sp_thermal_calculate_tau(sp_thermal_homeostat_t* ctx, float base_tau);

// Return the current thermal zone based on the last sensor reading.
sp_thermal_zone_t sp_thermal_get_zone(sp_thermal_homeostat_t* ctx);

// Log current thermal status to stdout.
// Format: "[SP-THERMAL] zone=NOMINAL cpu=72.3C gpu=68.1C max=72.3C fan=1850rpm tau_adj=0.77"
void sp_thermal_log_status(sp_thermal_homeostat_t* ctx);

// ---------------------------------------------------------------------------
//  Hot-path inline: sp_thermal_dynamic_tau
// ---------------------------------------------------------------------------
//
// This is the fast version for the inner decode loop. No sensor poll —
// uses whatever last_reading contains. The caller is responsible for
// ensuring sp_thermal_poll() is called at SP_THERMAL_POLL_INTERVAL_MS.
//
// The math:
//   if T > T_critical → tau = 1.0 (full stop, no speculation)
//   else → tau = base_tau + (T - T_limit) * sigma, clamped to [base_tau, 0.98]
//
// The 0.98 cap ensures we never fully eliminate compression even in
// ZONE_HOT — going to 1.0 is reserved for survival mode only.
//
static inline float sp_thermal_dynamic_tau(const sp_thermal_homeostat_t* ctx,
                                           float base_tau) {
    float max_t = ctx->last_reading.max_temp;

    // Survival mode: silicon is in danger, shut down speculation entirely
    if (max_t > ctx->t_critical) return 1.0f;

    // Homeostatic formula: proportional back-off above T_limit
    float heat_delta = max_t > ctx->t_limit ? max_t - ctx->t_limit : 0.0f;
    float dynamic_tau = base_tau + (heat_delta * ctx->sigma);

    // Clamp: never exceed 0.98 outside survival mode
    return dynamic_tau < 0.98f ? dynamic_tau : 0.98f;
}

#ifdef __cplusplus
}
#endif

#endif // SP_THERMAL_THROTTLE_H
