// Shannon-Prime BurnHard — Heterogeneous Reservoir Configuration
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Defines how the engine maps the physical Optane / NVMe / RAM tiers
// into the BurnHard "Dual-Chamber" Reservoir.  The Beast Canyon target
// rig pairs a 32 GB Optane stick (Primary Expert Reservoir) with a
// 16 GB stick (High-Speed Scratchpad / KV-overflow / draft model).
// Either chamber can be replaced by an NVMe-backed file or a RAM
// allocation; the engine probes at startup and binds whichever tier
// each chamber resolves to.

#ifndef BURNHARD_CONFIG_H
#define BURNHARD_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// Tier identification — what kind of medium each chamber is bound to.
// Latency floors (typical, 4 KB random read):
//   OPTANE:  ~10 us   (Memory M10 Persistent Memory, 3D XPoint)
//   NVME:    ~100 us  (consumer M.2 SSD; depends on controller + queue depth)
//   RAM:     ~0.1 us  (DDR4 LLC-miss read)
// The engine uses these tags to pick prefetch depth and queue strategy.
// ----------------------------------------------------------------------------
typedef enum {
    BURNHARD_TIER_NONE   = 0,   // Chamber unmapped; engine must skip / fallback
    BURNHARD_TIER_OPTANE = 1,   // 3D XPoint, byte-addressable when DAX-mounted
    BURNHARD_TIER_NVME   = 2,   // Standard M.2 NVMe, page-cache backed mmap
    BURNHARD_TIER_RAM    = 3,   // Heap allocation (last-resort fallback)
} burnhard_tier_t;

// What each chamber is *for*.  The engine consults this when deciding
// where to mmap the GGUF, where to spill KV-cache pages, etc.  Multiple
// roles may be packed into one chamber when capacity allows.
typedef enum {
    BURNHARD_ROLE_NONE          = 0,
    BURNHARD_ROLE_EXPERT_RESV   = 1 << 0,  // Primary expert weights (Q4_K MoE)
    BURNHARD_ROLE_KV_OVERFLOW   = 1 << 1,  // Compressed KV pages spilled here
    BURNHARD_ROLE_RoPE_TABLES   = 1 << 2,  // Pre-computed mRoPE freq tables
    BURNHARD_ROLE_DRAFT_MODEL   = 1 << 3,  // Speculative draft (paired w/ S22)
    BURNHARD_ROLE_KV_SCRATCH    = 1 << 4,  // Per-token K/V staging buffer
} burnhard_role_t;

// One physical chamber.  Path is platform-specific:
//   Windows: "E:\\reservoir.bin" or "\\\\.\\PhysicalDrive1"
//   Linux:   "/mnt/burnhard_m1/reservoir.bin" or "/dev/pmem0"
typedef struct {
    burnhard_tier_t tier;
    uint32_t        roles;          // Bitmask of burnhard_role_t
    char            path[260];      // Full filesystem path or device node
    size_t          capacity_bytes; // Total addressable bytes in this chamber
    size_t          used_bytes;     // Currently mapped / written bytes
    void           *mapping;        // mmap base pointer (NULL until bound)
    int             dax_enabled;    // 1 = byte-addressable mode, 0 = page cache
} burnhard_chamber_t;

// The full dual-chamber descriptor.  Lives on the engine.  primary is
// always populated (even if it falls back to NVMe); secondary is
// optional (NULL tier = single-chamber mode).
typedef struct {
    burnhard_chamber_t primary;     // Larger reservoir (32 GB target)
    burnhard_chamber_t secondary;   // Scratchpad (16 GB target)

    // Auto-discovered system context (read once at sp_burnhard_probe).
    int    has_optane_total_count;  // # Optane sticks visible to OS
    size_t optane_total_bytes;      // Sum of Optane capacities
    size_t nvme_free_bytes;         // Available NVMe scratch space
    size_t ram_free_bytes;          // Available RAM (last-resort)

    // Effective active mode after probe.  Copied to logs for debugging.
    enum {
        BURNHARD_MODE_DUAL_OPTANE   = 0,  // Primary=Optane, Secondary=Optane
        BURNHARD_MODE_OPTANE_NVME   = 1,  // Primary=Optane, Secondary=NVMe
        BURNHARD_MODE_OPTANE_ONLY   = 2,  // Primary=Optane, Secondary=NONE
        BURNHARD_MODE_NVME_FALLBACK = 3,  // Primary=NVMe (no Optane present)
        BURNHARD_MODE_RAM_FALLBACK  = 4,  // No persistent storage; RAM only
    } mode;
} burnhard_reservoir_t;

// ----------------------------------------------------------------------------
// API.  All functions return 0 on success, negative errno-like code on
// failure.  Probe is idempotent; bind/unbind acquire / release the mmaps.
// ----------------------------------------------------------------------------

// Discover what storage is available.  Populates the *_total_count /
// *_free_bytes fields.  Does NOT mmap anything yet.  After this call
// you can inspect r->has_optane_total_count etc. and decide config.
int  burnhard_probe(burnhard_reservoir_t *r);

// Bind a chamber to a specific path + roles.  If the path is on Optane
// and DAX is supported, dax_enabled=1.  Otherwise standard mmap.  When
// path is NULL and tier=BURNHARD_TIER_RAM, allocates `capacity_bytes`
// of pageable memory (no persistence).
int  burnhard_chamber_bind(burnhard_chamber_t *c,
                           burnhard_tier_t tier,
                           const char *path,
                           uint32_t roles,
                           size_t requested_bytes);

// Tear down one chamber's mapping.  Idempotent.
void burnhard_chamber_release(burnhard_chamber_t *c);

// Convenience: probe + auto-bind for the Beast Canyon dev rig.  Walks
// Optane → NVMe → RAM in order, fills primary with the largest Optane
// stick (or NVMe if none) and secondary with the second-largest tier.
// Logs the resolved mode to stderr.
int  burnhard_reservoir_init_auto(burnhard_reservoir_t *r,
                                  const char *gguf_path);

// Tear down both chambers.
void burnhard_reservoir_free(burnhard_reservoir_t *r);

// Helpers for log + debug output.
const char *burnhard_tier_name(burnhard_tier_t t);
const char *burnhard_mode_name(int mode);

#ifdef __cplusplus
}
#endif

#endif // BURNHARD_CONFIG_H
