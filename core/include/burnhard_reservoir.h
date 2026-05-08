/*
 * burnhard_reservoir.h — Tiered memory-mapped reservoir
 *
 * Provides a unified mmap'd address space for KV-cache and model weights,
 * backed by the best available storage tier:
 *
 *   Tier 0: Intel Optane DCPMM (DAX, byte-addressable, ~10 GB/s)
 *   Tier 1: NVMe SSD (mmap'd file, ~3-5 GB/s seq)
 *   Tier 2: System RAM (malloc fallback, capacity-limited)
 *
 * The caller gets a flat pointer. The reservoir handles tier selection,
 * fallback, and (optionally) async prefetch from lower tiers.
 */

#ifndef BURNHARD_RESERVOIR_H
#define BURNHARD_RESERVOIR_H

#include "burnhard_config.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RESERVOIR_TIER_OPTANE = 0,
    RESERVOIR_TIER_NVME   = 1,
    RESERVOIR_TIER_RAM    = 2,
} reservoir_tier_t;

typedef struct {
    void            *base;       /* mmap'd / malloc'd base pointer */
    size_t           size;       /* allocated bytes                */
    reservoir_tier_t tier;       /* which tier we landed on        */
    int              fd;         /* backing fd (-1 for RAM)        */
} reservoir_t;

/*
 * Open a reservoir of `size_bytes`. Tries tiers in order based on cfg.
 * Returns 0 on success. reservoir_t is caller-owned, freed with reservoir_close.
 */
int reservoir_open(reservoir_t *res, const burnhard_config_t *cfg, size_t size_bytes);

/*
 * Close and unmap.
 */
void reservoir_close(reservoir_t *res);

/*
 * Return human-readable tier name.
 */
const char *reservoir_tier_str(reservoir_tier_t tier);

/*
 * Prefetch a range into CPU cache. No-op on RAM tier.
 */
void reservoir_prefetch(const reservoir_t *res, size_t offset, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* BURNHARD_RESERVOIR_H */
