// Shannon-Prime VHT2: Predictive Expert Prefetch — Zero-Bubble Inference
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#ifndef SP_PREFETCH_ENGINE_H
#define SP_PREFETCH_ENGINE_H

// ============================================================================
// Top-2 Speculative Prefetch with Confidence Gate
// ============================================================================
//
// The i9 acts as a "scout": while the GPUs grind through Layer L, the
// Prefetcher thread looks ahead to Layer L+1 and pre-shreds the TWO
// hottest experts into a Shadow Buffer pair. When the Router's actual
// pick matches either prediction, we do a pointer swap (~10ns) instead
// of a full shred (~2ms).
//
// Architecture:
//   Thread 1 (Executor)   — manages current GPU kernels, merges CRT residues
//   Thread 2 (Prefetcher) — scouts L+1, shreds primary+secondary to shadow
//
// Top-2 Dual Shadow Buffers:
//   Each shadow slot holds TWO pre-shredded experts:
//     primary   — the #1 hottest expert from the curriculum heatmap
//     secondary — the #2 hottest (the "Second-Guess" buffer)
//
// Confidence Gate:
//   Before speculating, the prefetcher checks the combined confidence
//   of the top-K experts for the target layer. If confidence < tau
//   (default 0.75), the heatmap is too flat / uncertain — speculation
//   is skipped entirely to avoid wasting shred bandwidth on a coin flip.
//
// Hit/Miss logic:
//   If Router picks primary   → pointer swap (PRIMARY HIT, ~10ns)
//   If Router picks secondary → pointer swap (SECONDARY HIT, ~10ns)
//   Otherwise                 → standard shred (MISS, ~2ms)
//
// Adaptive alpha: if hit rate drops below 70%, the EWMA decay alpha in
// the curriculum is increased to make the heatmap more responsive.

#include "sp_moe_curriculum.h"
#include "sp_crt.h"

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Shadow Buffer — pre-shredded expert residues (one per expert)
// ============================================================================

typedef struct {
    // Which layer/expert this shadow holds. -1 = empty/invalid.
    int layer_id;
    int expert_id;

    // Pre-shredded CRT residues ready for GPU dispatch.
    // Sized for max_M × max_K (expert weight matrix).
    uint32_t* res_m1;           // Ring M1 residues (RTX 2060 / CUDA)
    uint32_t* res_m2;           // Ring M2 residues (Intel UHD / Level Zero)

    // The fp32 dequantized weights (intermediate — used by Shredder).
    float*    fp32_scratch;

    // CRT quantization params used for this shred.
    sp_crt_quant_t quant;

    // Dimensions of the pre-shredded data.
    int M;                       // rows (output features)
    int K;                       // cols (input features)

    // Validity flag. Set to 1 by the prefetch thread after shredding
    // completes; read by the executor thread.
    volatile int ready;
} sp_shadow_buffer_t;

// ============================================================================
// Shadow Slot — holds primary + secondary shadow for one double-buffer index
// ============================================================================

typedef struct {
    sp_shadow_buffer_t primary;
    sp_shadow_buffer_t secondary;
} sp_shadow_slot_t;

// ============================================================================
// Prefetch Engine state
// ============================================================================

typedef struct {
    // Double-buffered shadow slots — while one slot is being consumed by
    // the executor, the other can be filled by the prefetcher.
    // Each slot holds TWO shadows: primary (#1 expert) + secondary (#2).
    sp_shadow_slot_t slot[2];
    int active_slot;             // Index of the slot currently being filled

    // Pointer to the curriculum (not owned — the engine owns it)
    sp_moe_curriculum_t* curriculum;

    // Maximum dimensions (for buffer allocation)
    int max_M;
    int max_K;

    // Confidence gate — minimum combined top-K confidence to speculate.
    // If the heatmap is too flat (uniform distribution), speculation is
    // a coin flip and we skip it to avoid wasting shred bandwidth.
    float    confidence_threshold;  // tau (default 0.75)
    int      confidence_k;          // K for the confidence check (default 2)

    // Telemetry
    uint64_t primary_hits;         // Router matched the #1 predicted expert
    uint64_t secondary_hits;       // Router matched the #2 predicted expert
    uint64_t misses;               // Neither prediction matched
    uint64_t gated_skips;          // Speculation skipped by confidence gate
    uint64_t total_prefetches;     // Total dual-prefetch operations started

    // Adaptive alpha threshold — if hit_rate < this, increase curriculum alpha
    float    min_hit_rate;         // default 0.70

    int      initialized;
} sp_prefetch_engine_t;

// ============================================================================
// Internal helpers
// ============================================================================

static inline int sp_shadow_alloc(sp_shadow_buffer_t* s, size_t buf_size) {
    s->res_m1       = (uint32_t*)calloc(buf_size, sizeof(uint32_t));
    s->res_m2       = (uint32_t*)calloc(buf_size, sizeof(uint32_t));
    s->fp32_scratch = (float*)calloc(buf_size, sizeof(float));
    if (!s->res_m1 || !s->res_m2 || !s->fp32_scratch) return -1;
    s->layer_id  = -1;
    s->expert_id = -1;
    s->ready     = 0;
    return 0;
}

static inline void sp_shadow_dealloc(sp_shadow_buffer_t* s) {
    free(s->res_m1);       s->res_m1       = NULL;
    free(s->res_m2);       s->res_m2       = NULL;
    free(s->fp32_scratch); s->fp32_scratch = NULL;
}

// ============================================================================
// Lifecycle
// ============================================================================

static inline int sp_prefetch_init(sp_prefetch_engine_t* pe,
                                    sp_moe_curriculum_t* curriculum,
                                    int max_M, int max_K) {
    if (!pe || !curriculum) return -1;

    memset(pe, 0, sizeof(*pe));
    pe->curriculum            = curriculum;
    pe->max_M                 = max_M;
    pe->max_K                 = max_K;
    pe->min_hit_rate          = 0.70f;
    pe->confidence_threshold  = 0.75f;
    pe->confidence_k          = 2;
    pe->active_slot           = 0;

    // Allocate shadow buffers — 2 slots × 2 shadows (primary + secondary)
    size_t buf_size = (size_t)max_M * (size_t)max_K;
    for (int i = 0; i < 2; ++i) {
        if (sp_shadow_alloc(&pe->slot[i].primary, buf_size) != 0 ||
            sp_shadow_alloc(&pe->slot[i].secondary, buf_size) != 0) {
            // Cleanup on failure
            for (int j = 0; j <= i; ++j) {
                sp_shadow_dealloc(&pe->slot[j].primary);
                sp_shadow_dealloc(&pe->slot[j].secondary);
            }
            return -1;
        }
    }

    pe->initialized = 1;
    return 0;
}

static inline void sp_prefetch_free(sp_prefetch_engine_t* pe) {
    if (!pe) return;
    for (int i = 0; i < 2; ++i) {
        sp_shadow_dealloc(&pe->slot[i].primary);
        sp_shadow_dealloc(&pe->slot[i].secondary);
    }
    pe->initialized = 0;
}

// ============================================================================
// Shred a single expert into a shadow buffer (internal)
// ============================================================================

static inline void sp_shadow_shred(sp_shadow_buffer_t* shadow,
                                    int layer_id,
                                    int expert_id,
                                    const float* weight_fp32,
                                    int M, int K) {
    shadow->ready = 0;
    shadow->layer_id  = layer_id;
    shadow->expert_id = expert_id;
    shadow->M = M;
    shadow->K = K;

    // Calibrate quantization from the weight data
    float min_val = weight_fp32[0], max_val = weight_fp32[0];
    int n = M * K;
    for (int i = 1; i < n; ++i) {
        if (weight_fp32[i] < min_val) min_val = weight_fp32[i];
        if (weight_fp32[i] > max_val) max_val = weight_fp32[i];
    }

    sp_crt_quant_t q;
    sp_crt_quant_calibrate_k(&q, min_val, max_val, K);

    // Quantize to both residue rings
    for (int i = 0; i < n; ++i) {
        shadow->res_m1[i] = sp_crt_quantize_f32(&q, weight_fp32[i], SP_CRT_M1);
        shadow->res_m2[i] = sp_crt_quantize_f32(&q, weight_fp32[i], SP_CRT_M2);
    }

    shadow->quant = q;
    shadow->ready = 1;
}

// ============================================================================
// Prefetch request — called by the Prefetcher thread
// ============================================================================
//
// Single-expert shred (legacy API — still usable for non-MoE or fallback).

typedef void (*sp_dequant_fn)(const void* src, float* dst, int n);

static inline void sp_prefetch_shred(sp_prefetch_engine_t* pe,
                                      int layer_id,
                                      int expert_id,
                                      const float* weight_fp32,
                                      int M, int K) {
    if (!pe || !pe->initialized) return;

    sp_shadow_slot_t* slot = &pe->slot[pe->active_slot];
    sp_shadow_shred(&slot->primary, layer_id, expert_id, weight_fp32, M, K);

    // Invalidate secondary for single-expert shred
    slot->secondary.ready = 0;
    slot->secondary.layer_id  = -1;
    slot->secondary.expert_id = -1;

    pe->total_prefetches++;
}

// ============================================================================
// Top-2 Speculative Prefetch — dual-expert shred with confidence gate
// ============================================================================
//
// Shreds BOTH the primary (#1) and secondary (#2) predicted experts for
// `target_layer` into the active shadow slot. The confidence gate checks
// the curriculum heatmap first — if the combined top-K probability is
// below tau, speculation is skipped entirely.
//
// `get_weight_fn` is a callback that returns the fp32 weight pointer for
// a given (layer, expert) pair. The caller owns the weight memory.
//
// Returns:  1 = dual shred performed
//           0 = gated (confidence too low, speculation skipped)
//          -1 = error

typedef const float* (*sp_get_expert_weight_fn)(void* ctx,
                                                 int layer_id,
                                                 int expert_id,
                                                 int* out_M,
                                                 int* out_K);

static inline int sp_prefetch_shred_dual(sp_prefetch_engine_t* pe,
                                          int target_layer,
                                          sp_get_expert_weight_fn get_weight,
                                          void* weight_ctx) {
    if (!pe || !pe->initialized || !pe->curriculum || !get_weight) return -1;
    if (target_layer < 0 || target_layer >= pe->curriculum->n_layers) return -1;

    // ── Confidence gate ──────────────────────────────────────────
    float conf = sp_moe_top_k_confidence(pe->curriculum,
                                          target_layer,
                                          pe->confidence_k);
    if (conf < pe->confidence_threshold) {
        pe->gated_skips++;
        return 0;  // Heatmap too flat — skip speculation
    }

    // ── Get the top-2 predicted experts from the curriculum ──────
    sp_moe_expert_confidence_t top2[2];
    int n_got = sp_moe_get_top_k_confidence(pe->curriculum,
                                             target_layer,
                                             top2, 2);
    if (n_got < 1) return -1;

    sp_shadow_slot_t* slot = &pe->slot[pe->active_slot];

    // ── Shred primary (#1 hottest) ──────────────────────────────
    int M1 = 0, K1 = 0;
    const float* w1 = get_weight(weight_ctx, target_layer,
                                  top2[0].expert_id, &M1, &K1);
    if (!w1 || M1 <= 0 || K1 <= 0) return -1;
    sp_shadow_shred(&slot->primary, target_layer,
                     top2[0].expert_id, w1, M1, K1);

    // ── Shred secondary (#2 hottest, the "Second-Guess") ────────
    if (n_got >= 2) {
        int M2 = 0, K2 = 0;
        const float* w2 = get_weight(weight_ctx, target_layer,
                                      top2[1].expert_id, &M2, &K2);
        if (w2 && M2 > 0 && K2 > 0) {
            sp_shadow_shred(&slot->secondary, target_layer,
                             top2[1].expert_id, w2, M2, K2);
        } else {
            // Secondary weight unavailable — invalidate
            slot->secondary.ready = 0;
            slot->secondary.layer_id  = -1;
            slot->secondary.expert_id = -1;
        }
    } else {
        slot->secondary.ready = 0;
        slot->secondary.layer_id  = -1;
        slot->secondary.expert_id = -1;
    }

    pe->total_prefetches++;
    return 1;
}

// ============================================================================
// Hit check — called by the Executor before dispatching Layer L+1
// ============================================================================
//
// Checks BOTH primary and secondary shadow buffers for a match.
// Returns a pointer to the matching shadow if hit, NULL on miss.
// On hit, flips the active slot so the prefetcher can start filling
// the other one.
//
// `out_is_secondary` is set to 0 for primary hit, 1 for secondary hit.

static inline const sp_shadow_buffer_t* sp_prefetch_check(
        sp_prefetch_engine_t* pe,
        int layer_id,
        int expert_id,
        int* out_is_secondary) {
    if (!pe || !pe->initialized) return NULL;
    if (out_is_secondary) *out_is_secondary = 0;

    sp_shadow_slot_t* slot = &pe->slot[pe->active_slot];

    // Check primary first (most likely match)
    if (slot->primary.ready &&
        slot->primary.layer_id == layer_id &&
        slot->primary.expert_id == expert_id) {
        pe->primary_hits++;
        const sp_shadow_buffer_t* result = &slot->primary;
        pe->active_slot = 1 - pe->active_slot;
        return result;
    }

    // Check secondary ("Second-Guess" buffer)
    if (slot->secondary.ready &&
        slot->secondary.layer_id == layer_id &&
        slot->secondary.expert_id == expert_id) {
        pe->secondary_hits++;
        if (out_is_secondary) *out_is_secondary = 1;
        const sp_shadow_buffer_t* result = &slot->secondary;
        pe->active_slot = 1 - pe->active_slot;
        return result;
    }

    // MISS — neither prediction matched
    pe->misses++;
    return NULL;
}

// Backward-compatible 3-arg check (ignores secondary distinction).
static inline const sp_shadow_buffer_t* sp_prefetch_check_simple(
        sp_prefetch_engine_t* pe,
        int layer_id,
        int expert_id) {
    return sp_prefetch_check(pe, layer_id, expert_id, NULL);
}

// ============================================================================
// Adaptive alpha — adjust curriculum responsiveness
// ============================================================================

// Called periodically (e.g., every rebalance) to check if the prefetch
// hit rate is too low and the curriculum needs to be more reactive.
static inline void sp_prefetch_adapt(sp_prefetch_engine_t* pe) {
    if (!pe || !pe->initialized || !pe->curriculum) return;

    uint64_t total = pe->primary_hits + pe->secondary_hits + pe->misses;
    if (total < 64) return;  // Not enough data yet

    float hit_rate = (float)(pe->primary_hits + pe->secondary_hits)
                   / (float)total;

    if (hit_rate < pe->min_hit_rate) {
        // Curriculum is too slow to track changing patterns.
        // Increase alpha (more weight on recent data).
        float new_alpha = pe->curriculum->alpha * 1.5f;
        if (new_alpha > 0.5f) new_alpha = 0.5f;
        pe->curriculum->alpha = new_alpha;
    } else if (hit_rate > 0.90f && pe->curriculum->alpha > 0.03f) {
        // Very high hit rate — curriculum is tracking well, relax alpha
        // to reduce noise sensitivity.
        float new_alpha = pe->curriculum->alpha * 0.8f;
        if (new_alpha < 0.02f) new_alpha = 0.02f;
        pe->curriculum->alpha = new_alpha;
    }
}

// ============================================================================
// Telemetry
// ============================================================================

typedef struct {
    uint64_t primary_hits;
    uint64_t secondary_hits;
    uint64_t misses;
    uint64_t gated_skips;
    uint64_t total_prefetches;
    float    hit_rate;           // (primary + secondary) / (hits + misses)
    float    primary_rate;       // primary / (hits + misses)
    float    secondary_rate;     // secondary / (hits + misses)
    float    gate_skip_rate;     // gated_skips / (prefetches + skips)
    float    confidence_threshold;
    float    current_alpha;
} sp_prefetch_stats_t;

static inline sp_prefetch_stats_t sp_prefetch_get_stats(
        const sp_prefetch_engine_t* pe) {
    sp_prefetch_stats_t s;
    memset(&s, 0, sizeof(s));
    if (!pe || !pe->initialized) return s;

    s.primary_hits         = pe->primary_hits;
    s.secondary_hits       = pe->secondary_hits;
    s.misses               = pe->misses;
    s.gated_skips          = pe->gated_skips;
    s.total_prefetches     = pe->total_prefetches;
    s.confidence_threshold = pe->confidence_threshold;

    uint64_t check_total = pe->primary_hits + pe->secondary_hits + pe->misses;
    if (check_total > 0) {
        s.hit_rate       = (float)(pe->primary_hits + pe->secondary_hits)
                         / (float)check_total;
        s.primary_rate   = (float)pe->primary_hits   / (float)check_total;
        s.secondary_rate = (float)pe->secondary_hits / (float)check_total;
    }

    uint64_t gate_total = pe->total_prefetches + pe->gated_skips;
    if (gate_total > 0) {
        s.gate_skip_rate = (float)pe->gated_skips / (float)gate_total;
    }

    s.current_alpha = pe->curriculum ? pe->curriculum->alpha : 0.0f;

    return s;
}

#ifdef __cplusplus
}
#endif

#endif // SP_PREFETCH_ENGINE_H
