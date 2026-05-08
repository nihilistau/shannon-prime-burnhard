/*
 * burnhard_config.h — Hardware Capability & Feature Gating
 *
 * Shannon-Prime BurnHard Engine
 * The "rules of existence" for the distributed inference pipeline.
 *
 * Four operating modes, detected at runtime:
 *   FULL_PULSE      — All silicon active (CPU + Optane + iGPU + dGPU + S22 sidecar)
 *   DESKTOP_SOLO    — S22 detached; desktop runs its own speculative logic
 *   LEGACY          — No Optane; reservoir falls back to NVMe/system RAM
 *   LONE_WOLF       — Mobile-only; S22 runs standalone with local GGUF
 */

#ifndef BURNHARD_CONFIG_H
#define BURNHARD_CONFIG_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Hardware capabilities (probed once at init) ──────────────────── */

typedef struct {
    /* Storage tier */
    bool     has_optane;          /* Intel Optane DCPMM / persistent memory     */
    bool     has_nvme;            /* NVMe SSD available for reservoir fallback   */

    /* Compute */
    bool     has_avx512;          /* AVX-512 on host CPU                        */
    bool     has_cuda;            /* NVIDIA GPU (RTX 2060 or better)            */
    int      cuda_sm;             /* SM version (e.g. 75 for Turing)            */
    bool     has_l0;              /* Intel Level Zero / UHD 750 iGPU            */

    /* Sidecar (the S22 Ultra) */
    bool     has_sidecar;         /* ADB link to phone detected                 */
    bool     sidecar_responsive;  /* Last heartbeat was within timeout window   */

    /* Memory */
    uint64_t optane_capacity_mb;  /* 0 if absent                                */
    uint64_t system_ram_mb;
    uint64_t vram_mb;             /* dGPU VRAM; 0 if no CUDA                    */
} burnhard_capabilities_t;


/* ── Operating mode (derived from capabilities) ───────────────────── */

typedef enum {
    BURNHARD_MODE_FULL_PULSE   = 0,  /* Everything online                    */
    BURNHARD_MODE_DESKTOP_SOLO = 1,  /* No sidecar; desktop handles draft    */
    BURNHARD_MODE_LEGACY       = 2,  /* No Optane; reservoir on NVMe/RAM     */
    BURNHARD_MODE_LONE_WOLF    = 3,  /* Mobile standalone                    */
} burnhard_mode_t;


/* ── Runtime configuration ────────────────────────────────────────── */

typedef struct {
    burnhard_capabilities_t caps;
    burnhard_mode_t         mode;

    /* Bridge config (only meaningful when has_sidecar) */
    uint16_t bridge_port;            /* ADB-forwarded TCP port (default 8090) */
    uint32_t heartbeat_interval_ms;  /* How often to ping (default 500)       */
    uint32_t heartbeat_timeout_ms;   /* Declare dead after this (default 2000)*/

    /* Reservoir config */
    const char *optane_dev_path;     /* e.g. "/dev/dax0.0" or NULL           */
    const char *nvme_cache_path;     /* fallback mmap path                   */
    uint64_t    reservoir_size_mb;   /* requested reservoir size             */

    /* Speculative decoding */
    int      spec_draft_tokens;      /* draft length (default 8)             */
    bool     spec_use_sidecar;       /* true = phone drafts; false = local   */
} burnhard_config_t;


/* ── API ──────────────────────────────────────────────────────────── */

/*
 * Probe hardware, populate caps, select mode.
 * Returns 0 on success. Safe to call multiple times (re-probes sidecar).
 */
int burnhard_detect(burnhard_config_t *cfg);

/*
 * Return human-readable mode string (for logging).
 */
const char *burnhard_mode_str(burnhard_mode_t mode);

/*
 * Log full capability snapshot to stderr.
 */
void burnhard_dump_caps(const burnhard_config_t *cfg);

#ifdef __cplusplus
}
#endif

#endif /* BURNHARD_CONFIG_H */
