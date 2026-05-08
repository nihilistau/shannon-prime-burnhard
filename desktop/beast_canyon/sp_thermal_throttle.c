// Shannon-Prime Beast Canyon: Thermal-Aware Homeostatic Throttle
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "sp_thermal_throttle.h"

#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#  include <windows.h>
static uint64_t sp_thermal_time_us(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (uint64_t)(c.QuadPart * 1000000ULL / f.QuadPart);
}
#else
#  include <time.h>
static uint64_t sp_thermal_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000;
}
#endif

// ---------------------------------------------------------------------------
//  sp_thermal_init
// ---------------------------------------------------------------------------
int sp_thermal_init(sp_thermal_homeostat_t* ctx, float tau_base) {
    if (!ctx) return -1;
    memset(ctx, 0, sizeof(*ctx));

    ctx->t_limit    = SP_THERMAL_T_LIMIT;
    ctx->t_critical = SP_THERMAL_T_CRITICAL;
    ctx->sigma      = SP_THERMAL_SIGMA;
    ctx->tau_base   = tau_base;

    ctx->current_zone = SP_THERMAL_ZONE_COOL;

    // Set initial reading to unknown (sensor not yet polled).
    ctx->last_reading.cpu_temp     = -1.0f;
    ctx->last_reading.gpu_temp     = -1.0f;
    ctx->last_reading.max_temp     = -1.0f;
    ctx->last_reading.fan_rpm      = 0;
    ctx->last_reading.timestamp_us = 0;

    // Probe NVML (NVIDIA Management Library).
    // TODO: nvmlInit_v2() → nvmlDeviceGetHandleByIndex(0) → available
    // This requires linking against nvml.dll / libnvidia-ml.so at runtime
    // (dlopen/LoadLibrary). For now, mark unavailable.
    ctx->nvml_available = false;

    // Probe Level-Zero Sysman (Intel iGPU / CPU package).
    // TODO: zesInit() → zesDeviceEnumTemperatureSensors() → available
    // Same approach: runtime dlopen. For now, mark unavailable.
    ctx->l0_sysman_available = false;

    // Return -1 to indicate no sensor backend is available.
    // This is non-fatal — the engine runs without thermal awareness,
    // sp_thermal_dynamic_tau() returns base_tau unchanged because
    // last_reading.max_temp == -1 which is < t_limit.
    if (!ctx->nvml_available && !ctx->l0_sysman_available) {
        return -1;
    }

    return 0;
}

// ---------------------------------------------------------------------------
//  sp_thermal_free
// ---------------------------------------------------------------------------
void sp_thermal_free(sp_thermal_homeostat_t* ctx) {
    if (!ctx) return;
    // TODO: nvmlShutdown() if nvml_available
    // TODO: L0 Sysman cleanup if l0_sysman_available
    memset(ctx, 0, sizeof(*ctx));
}

// ---------------------------------------------------------------------------
//  sp_thermal_poll
// ---------------------------------------------------------------------------
int sp_thermal_poll(sp_thermal_homeostat_t* ctx) {
    if (!ctx) return -1;

    float cpu_temp = -1.0f;
    float gpu_temp = -1.0f;
    uint32_t fan_rpm = 0;

    // --- NVML path (GPU temperature + fan) ---
    if (ctx->nvml_available) {
        // TODO: nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &gpu_temp_u)
        // TODO: nvmlDeviceGetFanSpeed(device, &fan_pct) → estimate RPM
        // gpu_temp = (float)gpu_temp_u;
    }

    // --- Level-Zero Sysman path (CPU/iGPU temperature) ---
    if (ctx->l0_sysman_available) {
        // TODO: zesDeviceEnumTemperatureSensors() → zesTemperatureGetState()
        // cpu_temp = reading.current;
    }

    // --- Fallback: no sensors available ---
    if (cpu_temp < 0.0f && gpu_temp < 0.0f) {
        return -1;  // No readings obtained
    }

    // Update the reading snapshot.
    sp_thermal_zone_t old_zone = ctx->current_zone;

    ctx->last_reading.cpu_temp     = cpu_temp;
    ctx->last_reading.gpu_temp     = gpu_temp;
    ctx->last_reading.fan_rpm      = fan_rpm;
    ctx->last_reading.timestamp_us = sp_thermal_time_us();

    // max_temp is the constraint that matters.
    float max_t = cpu_temp;
    if (gpu_temp > max_t) max_t = gpu_temp;
    ctx->last_reading.max_temp = max_t;

    // Update zone.
    ctx->current_zone = sp_thermal_get_zone(ctx);
    ctx->total_polls++;
    if (ctx->current_zone != old_zone) {
        ctx->zone_transitions++;
    }

    return 0;
}

// ---------------------------------------------------------------------------
//  sp_thermal_calculate_tau
// ---------------------------------------------------------------------------
float sp_thermal_calculate_tau(sp_thermal_homeostat_t* ctx, float base_tau) {
    if (!ctx) return base_tau;

    // Re-poll if the last reading is stale.
    uint64_t now = sp_thermal_time_us();
    uint64_t age_ms = (now - ctx->last_reading.timestamp_us) / 1000;
    if (age_ms > SP_THERMAL_POLL_INTERVAL_MS || ctx->last_reading.timestamp_us == 0) {
        sp_thermal_poll(ctx);
    }

    // If no reading available, return base tau unchanged.
    if (ctx->last_reading.max_temp < 0.0f) return base_tau;

    // Use the fast inline path which implements the homeostatic formula.
    return sp_thermal_dynamic_tau(ctx, base_tau);
}

// ---------------------------------------------------------------------------
//  sp_thermal_get_zone
// ---------------------------------------------------------------------------
sp_thermal_zone_t sp_thermal_get_zone(sp_thermal_homeostat_t* ctx) {
    if (!ctx || ctx->last_reading.max_temp < 0.0f) return SP_THERMAL_ZONE_COOL;

    float t = ctx->last_reading.max_temp;
    if (t > 85.0f) return SP_THERMAL_ZONE_HOT;
    if (t >= 70.0f) return SP_THERMAL_ZONE_NOMINAL;
    return SP_THERMAL_ZONE_COOL;
}

// ---------------------------------------------------------------------------
//  sp_thermal_log_status
// ---------------------------------------------------------------------------
void sp_thermal_log_status(sp_thermal_homeostat_t* ctx) {
    if (!ctx) return;

    const char* zone_str = "COOL";
    if (ctx->current_zone == SP_THERMAL_ZONE_NOMINAL) zone_str = "NOMINAL";
    else if (ctx->current_zone == SP_THERMAL_ZONE_HOT) zone_str = "HOT";

    float adj_tau = sp_thermal_dynamic_tau(ctx, ctx->tau_base);

    fprintf(stderr, "[SP-THERMAL] zone=%s cpu=%.1fC gpu=%.1fC max=%.1fC fan=%urpm tau_adj=%.2f\n",
            zone_str,
            ctx->last_reading.cpu_temp,
            ctx->last_reading.gpu_temp,
            ctx->last_reading.max_temp,
            (unsigned)ctx->last_reading.fan_rpm,
            adj_tau);
}
