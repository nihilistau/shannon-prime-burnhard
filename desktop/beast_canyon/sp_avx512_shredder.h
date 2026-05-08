// Shannon-Prime Beast Canyon: AVX-512 "Shredder" — Weight Dequantization
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// The Shredder reads quantized weights directly from Optane mmap pointers
// and inflates them to fp16 in a staging buffer (LLC-resident via SVM or
// pinned allocation). This is the CPU-side pump that feeds both GPUs.
//
// Supported formats: Q4_0, Q4_1, Q8_0, Q4_K, Q6_K, F16 (passthrough).
// AVX-512F + AVX-512BW + F16C required (i9-11900 and later).
//
// Design invariants:
//   1. Input: raw pointer into Optane mmap (no file I/O).
//   2. Output: fp16 staging buffer aligned to 64 bytes (cache line).
//   3. Prefetch hints issued AHEAD of the read cursor.
//   4. Entire operation runs in user space — no kernel transitions.

#ifndef SP_AVX512_SHREDDER_H
#define SP_AVX512_SHREDDER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Shredder Configuration
// ============================================================================

typedef struct {
    int      prefetch_pages;     // Pages to prefetch ahead (default 8)
    bool     use_streaming;      // Use NT stores for staging buffer (default false)
    bool     force_f32_staging;  // Stage as f32 instead of f16 (for CPU-only path)
} sp_shredder_config_t;

// Initialize config with defaults.
static inline void sp_shredder_config_init(sp_shredder_config_t *cfg) {
    cfg->prefetch_pages = 8;
    cfg->use_streaming = false;
    cfg->force_f32_staging = false;
}

// ============================================================================
// Shredder Context
// ============================================================================

typedef struct {
    sp_shredder_config_t config;

    // Staging buffer (fp16 or f32 depending on config)
    void        *staging_buf;     // Aligned to 64 bytes
    size_t       staging_capacity; // In bytes
    size_t       staging_elements; // Number of fp16/f32 values

    // CPU feature detection
    bool         has_avx512f;
    bool         has_avx512bw;
    bool         has_avx512_vnni;
    bool         has_f16c;
    bool         has_avx512_fp16; // Native fp16 arithmetic (Sapphire Rapids+)

    // Performance counters
    uint64_t     total_bytes_shredded;
    uint64_t     total_elements_produced;
    uint64_t     total_shred_calls;
    uint64_t     total_shred_us;     // Microseconds spent in shred calls
} sp_shredder_t;

// ============================================================================
// Public API
// ============================================================================

// Initialize the shredder. Detects CPU features and allocates staging buffer.
// staging_elements: max number of fp16 values the staging buffer can hold.
//                   For a single expert MLP: hidden_dim * intermediate_dim.
//                   For the ping-pong system: 2x this.
int sp_shredder_init(sp_shredder_t *shred, const sp_shredder_config_t *cfg,
                     size_t staging_elements);

void sp_shredder_free(sp_shredder_t *shred);

// ============================================================================
// Core Shred Operations — dequantize from Optane pointer to staging buffer
// ============================================================================

// Shred Q4_0: 32 elements per block, 18 bytes per block (2 scale + 16 data).
// src: raw pointer into Optane mmap (Q4_0 blocks).
// dst: fp16 staging buffer (must hold n_elements * sizeof(uint16_t)).
// n_elements: number of fp16 values to produce (must be multiple of 32).
void sp_shredder_q4_0(const sp_shredder_t *shred,
                      const void *src, uint16_t *dst, size_t n_elements);

// Shred Q4_1: like Q4_0 but with min value (20 bytes/block).
void sp_shredder_q4_1(const sp_shredder_t *shred,
                      const void *src, uint16_t *dst, size_t n_elements);

// Shred Q8_0: 32 elements per block, 34 bytes per block.
void sp_shredder_q8_0(const sp_shredder_t *shred,
                      const void *src, uint16_t *dst, size_t n_elements);

// Shred Q4_K: K-quant format (256 elements per superblock, 144 bytes).
void sp_shredder_q4_k(const sp_shredder_t *shred,
                      const void *src, uint16_t *dst, size_t n_elements);

// Shred Q6_K: K-quant format (256 elements per superblock, 210 bytes).
void sp_shredder_q6_k(const sp_shredder_t *shred,
                      const void *src, uint16_t *dst, size_t n_elements);

// F16 passthrough: just memcpy with prefetch (Optane → LLC).
void sp_shredder_f16(const sp_shredder_t *shred,
                     const void *src, uint16_t *dst, size_t n_elements);

// ============================================================================
// Auto-dispatch: picks the right shredder based on ggml type.
// ============================================================================

// Shred any supported quantized tensor into fp16 staging.
// ggml_type: SP_GGML_TYPE_* enum value.
// Returns 0 on success, -1 if type unsupported.
int sp_shredder_auto(const sp_shredder_t *shred,
                     uint32_t ggml_type,
                     const void *src, uint16_t *dst, size_t n_elements);

// ============================================================================
// Staging buffer management — for the ping-pong system
// ============================================================================

// Get pointer to the staging buffer (or a specific offset).
// The Beast Canyon orchestrator uses two halves for ping-pong.
static inline void *sp_shredder_staging(sp_shredder_t *shred) {
    return shred->staging_buf;
}

// Get staging buffer half for double-buffering.
// half: 0 or 1.
static inline void *sp_shredder_staging_half(sp_shredder_t *shred, int half) {
    size_t half_bytes = shred->staging_capacity / 2;
    return (uint8_t *)shred->staging_buf + (half * half_bytes);
}

// ============================================================================
// Diagnostics
// ============================================================================

void sp_shredder_print_status(const sp_shredder_t *shred);

// Returns average throughput in GB/s for shred operations.
double sp_shredder_throughput_gbps(const sp_shredder_t *shred);

#ifdef __cplusplus
}
#endif

#endif // SP_AVX512_SHREDDER_H
