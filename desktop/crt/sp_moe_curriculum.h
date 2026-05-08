// Shannon-Prime VHT2: MoE Expert Curriculum — Homeostatic Expert Balancer
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#ifndef SP_MOE_CURRICULUM_H
#define SP_MOE_CURRICULUM_H

// ============================================================================
// MoE Expert Curriculum — Beast Canyon heterogeneous dispatch
// ============================================================================
//
// In a Top-K MoE model, different experts see wildly different activation
// rates. The "hot" experts that the router keeps selecting for core logic
// should never be stuck behind the PCIe bus bottleneck. The "cool" experts
// that fire infrequently should stay LLC-resident for the Intel UHD to
// handle opportunistically.
//
// This module implements:
//
//   1. A per-layer EWMA heatmap that tracks expert usage over time.
//   2. A periodic rebalancing step ("Curriculum Pulse") that re-ranks
//      experts and assigns them to GPU tiers.
//   3. An assignment table that the CRT Shredder consults before dispatch.
//
// The scoring formula uses exponentially weighted moving average:
//
//   S_e = (1 - alpha) * S_e + alpha * I_active
//
// Where I_active is 1 if the expert was selected for the current token,
// 0 otherwise. alpha controls the decay (default 0.05).
//
// Every N tokens (the "Curriculum Pulse", default 128), the i9 re-ranks:
//   - Tier 1 (Powerhouse): Top 50% by score → RTX 2060 (raw compute)
//   - Tier 2 (Proximity):  Bottom 50%       → Intel UHD (LLC-resident)
//
// The assignment table is what the Shredder checks before dispatching
// residues to the appropriate GPU's pinned memory or SVM buffer.

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define SP_MOE_MAX_LAYERS   128
#define SP_MOE_MAX_EXPERTS  256

// GPU destination tiers
typedef enum {
    SP_GPU_RTX_2060  = 0,   // Tier 1: PCIe-attached high-compute GPU
    SP_GPU_INTEL_UHD = 1,   // Tier 2: Ring Bus LLC-resident iGPU
    SP_GPU_TIER_COUNT
} sp_gpu_destination_t;

// ============================================================================
// Expert Heatmap — per-layer EWMA scoring
// ============================================================================

typedef struct {
    // EWMA scores: [n_layers][n_experts]. Higher = hotter.
    float scores[SP_MOE_MAX_LAYERS][SP_MOE_MAX_EXPERTS];

    // Raw usage counters for the current pulse window.
    // Reset to 0 after each rebalance.
    uint32_t usage[SP_MOE_MAX_LAYERS][SP_MOE_MAX_EXPERTS];

    // Per-layer expert → GPU assignment table.
    // Updated by rebalance_curriculum().
    sp_gpu_destination_t assignment[SP_MOE_MAX_LAYERS][SP_MOE_MAX_EXPERTS];

    // Per-layer sorted expert indices (hottest first).
    // Populated by rebalance so the prefetcher can look up the #1 expert
    // for any layer in O(1).
    int sorted_experts[SP_MOE_MAX_LAYERS][SP_MOE_MAX_EXPERTS];

    // Configuration
    float    alpha;             // EWMA decay (default 0.05)
    int      pulse_interval;    // Rebalance every N tokens (default 128)
    float    tier1_fraction;    // Fraction assigned to RTX (default 0.5)

    // Model dimensions
    int      n_layers;
    int      n_experts;
    int      n_expert_used;     // Top-K (e.g., 2 for Top-2 MoE)

    // Telemetry
    uint64_t total_tokens;      // Tokens processed since init
    uint64_t total_rebalances;  // Number of curriculum pulses fired
    uint64_t tier1_dispatches;  // Matmuls sent to RTX 2060
    uint64_t tier2_dispatches;  // Matmuls sent to Intel UHD

    int      initialized;
} sp_moe_curriculum_t;

// ============================================================================
// Lifecycle
// ============================================================================

// Initialise the curriculum. All experts start assigned to RTX_2060
// (conservative: everything goes to the fast GPU until the heatmap
// accumulates enough signal to tier).
static inline int sp_moe_curriculum_init(sp_moe_curriculum_t* c,
                                          int n_layers,
                                          int n_experts,
                                          int n_expert_used) {
    if (!c) return -1;
    if (n_layers > SP_MOE_MAX_LAYERS || n_experts > SP_MOE_MAX_EXPERTS)
        return -1;

    memset(c, 0, sizeof(*c));
    c->n_layers      = n_layers;
    c->n_experts     = n_experts;
    c->n_expert_used = n_expert_used;
    c->alpha          = 0.05f;
    c->pulse_interval = 128;
    c->tier1_fraction = 0.5f;

    // Default assignment: all experts → RTX 2060 (Tier 1)
    for (int l = 0; l < n_layers; ++l) {
        for (int e = 0; e < n_experts; ++e) {
            c->assignment[l][e] = SP_GPU_RTX_2060;
            c->sorted_experts[l][e] = e;
            c->scores[l][e] = 1.0f / (float)n_experts;  // uniform prior
        }
    }

    c->initialized = 1;
    return 0;
}

static inline void sp_moe_curriculum_free(sp_moe_curriculum_t* c) {
    if (c) {
        c->initialized = 0;
    }
}

// ============================================================================
// Usage recording — called by the router after expert selection
// ============================================================================

// Record that `expert_ids[0..k-1]` were selected for a token at `layer_id`.
// This is called from the sampler/router path — must be very fast.
static inline void sp_moe_record_usage(sp_moe_curriculum_t* c,
                                        int layer_id,
                                        const int* expert_ids,
                                        int k) {
    if (!c || !c->initialized) return;
    if (layer_id < 0 || layer_id >= c->n_layers) return;

    for (int i = 0; i < k; ++i) {
        int eid = expert_ids[i];
        if (eid >= 0 && eid < c->n_experts) {
            c->usage[layer_id][eid]++;
        }
    }
}

// ============================================================================
// EWMA update + rebalance ("Curriculum Pulse")
// ============================================================================

// Simple insertion sort for the expert ranking (n_experts is small, O(E²) fine).
static inline void sp_moe_sort_experts(const float* scores, int* sorted, int n) {
    for (int i = 0; i < n; ++i) sorted[i] = i;
    for (int i = 1; i < n; ++i) {
        int key = sorted[i];
        float key_score = scores[key];
        int j = i - 1;
        while (j >= 0 && scores[sorted[j]] < key_score) {
            sorted[j + 1] = sorted[j];
            --j;
        }
        sorted[j + 1] = key;
    }
}

// Called every token. Checks if the pulse interval has been reached,
// and if so, updates EWMA scores and re-ranks/re-assigns experts.
// Returns 1 if a rebalance occurred, 0 otherwise.
static inline int sp_moe_tick(sp_moe_curriculum_t* c) {
    if (!c || !c->initialized) return 0;

    c->total_tokens++;

    if ((c->total_tokens % (uint64_t)c->pulse_interval) != 0)
        return 0;

    // ── EWMA update ──────────────────────────────────────────────
    const float alpha = c->alpha;
    const float one_minus_alpha = 1.0f - alpha;

    for (int l = 0; l < c->n_layers; ++l) {
        for (int e = 0; e < c->n_experts; ++e) {
            // Normalise usage count over the pulse window to [0, 1].
            // Maximum possible usage = pulse_interval (selected every token).
            float activity = (float)c->usage[l][e] / (float)c->pulse_interval;
            if (activity > 1.0f) activity = 1.0f;

            // EWMA update
            c->scores[l][e] = one_minus_alpha * c->scores[l][e]
                             + alpha * activity;
        }

        // Sort experts by score (descending — hottest first)
        sp_moe_sort_experts(c->scores[l], c->sorted_experts[l], c->n_experts);

        // Assign tiers: top tier1_fraction → RTX, rest → UHD
        int n_tier1 = (int)(c->tier1_fraction * (float)c->n_experts + 0.5f);
        if (n_tier1 < 1) n_tier1 = 1;
        if (n_tier1 > c->n_experts) n_tier1 = c->n_experts;

        for (int rank = 0; rank < c->n_experts; ++rank) {
            int eid = c->sorted_experts[l][rank];
            c->assignment[l][eid] = (rank < n_tier1) ? SP_GPU_RTX_2060
                                                     : SP_GPU_INTEL_UHD;
        }

        // Reset usage counters for the next pulse window
        memset(c->usage[l], 0, sizeof(uint32_t) * (size_t)c->n_experts);
    }

    c->total_rebalances++;
    return 1;
}

// ============================================================================
// Query interface — used by the Shredder and Prefetcher
// ============================================================================

// Which GPU should handle expert `eid` at `layer_id`?
static inline sp_gpu_destination_t sp_moe_get_destination(
        const sp_moe_curriculum_t* c, int layer_id, int expert_id) {
    if (!c || !c->initialized) return SP_GPU_RTX_2060;
    if (layer_id < 0 || layer_id >= c->n_layers) return SP_GPU_RTX_2060;
    if (expert_id < 0 || expert_id >= c->n_experts) return SP_GPU_RTX_2060;
    return c->assignment[layer_id][expert_id];
}

// Get the hottest expert for a given layer (sorted_experts[layer][0]).
// Used by the prefetcher to predict the most likely expert for layer L+1.
static inline int sp_moe_get_hottest(const sp_moe_curriculum_t* c,
                                      int layer_id) {
    if (!c || !c->initialized) return 0;
    if (layer_id < 0 || layer_id >= c->n_layers) return 0;
    return c->sorted_experts[layer_id][0];
}

// Get the top-K hottest experts for a layer (for multi-expert prefetch).
static inline void sp_moe_get_hottest_k(const sp_moe_curriculum_t* c,
                                          int layer_id,
                                          int* out_experts,
                                          int k) {
    if (!c || !c->initialized || !out_experts) return;
    if (layer_id < 0 || layer_id >= c->n_layers) return;
    if (k > c->n_experts) k = c->n_experts;
    for (int i = 0; i < k; ++i)
        out_experts[i] = c->sorted_experts[layer_id][i];
}

// ============================================================================
// Confidence query — normalised scores for Top-K (used by prefetch gate)
// ============================================================================

typedef struct {
    int   expert_id;
    float score;            // raw EWMA score
    float confidence;       // normalised: score / sum_of_all_scores for this layer
} sp_moe_expert_confidence_t;

// Return the top-K experts for `layer_id` with their normalised confidence.
// `out` must have room for `k` entries. Returns actual count written.
static inline int sp_moe_get_top_k_confidence(
        const sp_moe_curriculum_t* c,
        int layer_id,
        sp_moe_expert_confidence_t* out,
        int k) {
    if (!c || !c->initialized || !out) return 0;
    if (layer_id < 0 || layer_id >= c->n_layers) return 0;
    if (k > c->n_experts) k = c->n_experts;

    // Sum all scores for this layer (for normalisation).
    float total = 0.0f;
    for (int e = 0; e < c->n_experts; ++e)
        total += c->scores[layer_id][e];
    if (total <= 0.0f) total = 1.0f;  // avoid div-by-zero on cold start

    for (int i = 0; i < k; ++i) {
        int eid = c->sorted_experts[layer_id][i];
        out[i].expert_id  = eid;
        out[i].score      = c->scores[layer_id][eid];
        out[i].confidence = c->scores[layer_id][eid] / total;
    }
    return k;
}

// Convenience: combined confidence of top-K experts for a layer.
// Used by the confidence gate to decide whether to speculate.
static inline float sp_moe_top_k_confidence(
        const sp_moe_curriculum_t* c,
        int layer_id,
        int k) {
    if (!c || !c->initialized) return 0.0f;
    if (layer_id < 0 || layer_id >= c->n_layers) return 0.0f;
    if (k > c->n_experts) k = c->n_experts;

    float total = 0.0f;
    for (int e = 0; e < c->n_experts; ++e)
        total += c->scores[layer_id][e];
    if (total <= 0.0f) return 0.0f;

    float top_sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        int eid = c->sorted_experts[layer_id][i];
        top_sum += c->scores[layer_id][eid];
    }
    return top_sum / total;
}

// ============================================================================
// Telemetry
// ============================================================================

static inline void sp_moe_record_dispatch(sp_moe_curriculum_t* c,
                                           sp_gpu_destination_t dest) {
    if (!c) return;
    if (dest == SP_GPU_RTX_2060)  c->tier1_dispatches++;
    else                          c->tier2_dispatches++;
}

typedef struct {
    uint64_t total_tokens;
    uint64_t total_rebalances;
    uint64_t tier1_dispatches;
    uint64_t tier2_dispatches;
    float    avg_tier1_score;   // mean score of Tier-1 experts (last pulse)
    float    avg_tier2_score;   // mean score of Tier-2 experts (last pulse)
} sp_moe_curriculum_stats_t;

static inline sp_moe_curriculum_stats_t sp_moe_get_stats(
        const sp_moe_curriculum_t* c) {
    sp_moe_curriculum_stats_t s;
    memset(&s, 0, sizeof(s));
    if (!c || !c->initialized) return s;

    s.total_tokens     = c->total_tokens;
    s.total_rebalances = c->total_rebalances;
    s.tier1_dispatches = c->tier1_dispatches;
    s.tier2_dispatches = c->tier2_dispatches;

    // Compute mean scores for each tier across all layers
    float sum1 = 0, sum2 = 0;
    int   cnt1 = 0, cnt2 = 0;
    for (int l = 0; l < c->n_layers; ++l) {
        for (int e = 0; e < c->n_experts; ++e) {
            if (c->assignment[l][e] == SP_GPU_RTX_2060) {
                sum1 += c->scores[l][e]; cnt1++;
            } else {
                sum2 += c->scores[l][e]; cnt2++;
            }
        }
    }
    if (cnt1 > 0) s.avg_tier1_score = sum1 / (float)cnt1;
    if (cnt2 > 0) s.avg_tier2_score = sum2 / (float)cnt2;

    return s;
}

#ifdef __cplusplus
}
#endif

#endif // SP_MOE_CURRICULUM_H
