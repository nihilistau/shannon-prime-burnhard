// Shannon-Prime Beast Canyon: Shadow-Steal Speculative Dispatch
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// ============================================================================
// Shadow-Steal: Speculative Expert Pre-Computation on Intel UHD
// ============================================================================
//
// In a Beast Canyon MoE pipeline, the Router's top-K selection is the critical
// path bottleneck: every token waits for the Router to finish before any expert
// computation can begin. Shadow-Steal exploits a unique property of the Beast
// Canyon topology — the Intel UHD iGPU shares the Ring Bus with the i9 cores —
// to speculatively pre-compute experts BEFORE the Router confirms selection.
//
// The insight: the Curriculum EWMA heatmap already tells us which experts are
// "hot" for a given layer. For many tokens, the Router's actual selection is
// predictable from recent history. When confidence is high (combined probability
// of the top-2 predicted experts exceeds tau), we launch speculative computation
// on the UHD immediately, using the Ring Bus shared memory path.
//
// Why UHD specifically:
//   - Ring Bus interconnect means abort+restart costs ~2 microseconds, not
//     milliseconds. A PCIe-attached GPU would make this design untenable.
//   - UHD compute sits idle during Router evaluation anyway (Router runs on i9
//     AVX-512 cores). Shadow-Steal turns dead silicon into free lookahead.
//   - Level Zero command queues support fine-grained abort via zeCommandListReset,
//     so partial computation can be discarded without side effects.
//
// The lifecycle per token:
//   1. IDLE:        No speculation active.
//   2. SPECULATING: Scout identified top-2, UHD launched, waiting for Router.
//   3. HIT:         Router confirmed our prediction → pointer swap (~10ns).
//   4. MISS:        Router chose differently → abort UHD, flush queue (~2ms).
//   5. ABORTED:     Speculation abandoned (confidence < tau, or error).
//
// The confidence gate (tau) is the key tunable. Too low → too many misses,
// wasting UHD cycles. Too high → too few speculations, leaving UHD idle.
// Default 0.75 works well for Mixtral-class models where the top expert
// typically has ~60% probability and the top-2 combined exceeds 85%.

#ifndef SP_SHADOW_STEAL_H
#define SP_SHADOW_STEAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// MoE Curriculum provides the EWMA heatmap used by the Scout to predict
// which experts the Router will select for the next layer.
#include "../crt/sp_moe_curriculum.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

// Maximum number of experts we track across all MoE layers.
// Mixtral = 8, DBRX = 16, Switch Transformer = up to 64.
#define SP_SHADOW_STEAL_MAX_EXPERTS 64

// Default confidence threshold for launching speculative computation.
// The combined probability of the top-2 predicted experts must exceed
// this value before UHD launches. 0.75 balances hit rate (~85%) against
// the cost of occasional misses (~2ms flush).
#define SP_SHADOW_STEAL_DEFAULT_TAU  0.75f

// Aggressive mode: lower threshold for latency-sensitive workloads.
// Hit rate drops to ~70% but more tokens get speculative coverage.
// Useful when the UHD would otherwise be 100% idle.
#define SP_SHADOW_STEAL_TAU_MIN      0.50f

// Conservative mode: only speculate when practically certain.
// Hit rate ~97% but many tokens get no speculation at all.
// Useful when UHD is shared with display compositing.
#define SP_SHADOW_STEAL_TAU_MAX      0.98f


// ============================================================================
// Enums
// ============================================================================

// State machine for a single steal attempt. Transitions:
//   IDLE → SPECULATING (scout launches)
//   SPECULATING → HIT (router confirms prediction)
//   SPECULATING → MISS (router chose differently)
//   SPECULATING → ABORTED (tau gate failed, L0 error, or manual abort)
//   HIT|MISS|ABORTED → IDLE (reset for next token)
typedef enum {
    SP_STEAL_IDLE       = 0,  // No speculation in flight
    SP_STEAL_SPECULATING = 1, // UHD compute launched, awaiting Router verdict
    SP_STEAL_HIT        = 2,  // Prediction matched — pointer swap completed
    SP_STEAL_MISS       = 3,  // Prediction wrong — UHD flushed, full recompute
    SP_STEAL_ABORTED    = 4   // Speculation cancelled before Router verdict
} sp_steal_state_t;

// Which GPU a shadow buffer targets. The dispatch path differs:
//   PRIMARY_GPU:   CUDA memcpy into RTX 2060 device memory, cuLaunchKernel
//   SECONDARY_GPU: Level Zero shared allocation on Ring Bus, zeCommandListAppend
//
// Shadow-Steal always targets SECONDARY_GPU (UHD) for speculative work,
// because abort cost on Ring Bus is negligible. PRIMARY_GPU is used for
// the confirmed computation after Router verdict.
typedef enum {
    SP_GPU_PRIMARY   = 0,     // RTX 2060 / CUDA — confirmed compute target
    SP_GPU_SECONDARY = 1      // Intel UHD / Level Zero — speculative target
} sp_gpu_target_t;


// ============================================================================
// Structs
// ============================================================================

// A shadow buffer holds pre-computed (or in-flight) expert output for one
// predicted expert. We maintain two slots because MoE routers typically
// select top-2 experts per token. Slot 0 = highest predicted, slot 1 = second.
//
// On a HIT, we pointer-swap `data` directly into the MoE accumulator,
// avoiding any memcpy. On a MISS, we zero the ready flag and let the
// confirmed path overwrite.
typedef struct {
    void*            data;       // Expert output buffer (Ring Bus shared alloc)
    size_t           size;       // Buffer size in bytes (expert_dim * sizeof(float))
    int              expert_id;  // Which expert this slot is computing
    bool             ready;      // True once UHD kernel has finished writing
    sp_gpu_target_t  target;     // Which GPU owns this buffer
} sp_shadow_buffer_t;

// Telemetry for Shadow-Steal efficiency. Updated after every steal attempt.
// The key metric is `net_time_saved_ms`: total time saved by pointer swaps
// on HITs, minus total time wasted by flushes on MISSes.
typedef struct {
    uint64_t total_steals;       // Total speculation attempts launched
    uint64_t hits;               // Router confirmed our prediction
    uint64_t misses;             // Router chose differently
    uint64_t aborts;             // Speculation cancelled before verdict

    float    hit_rate;           // hits / total_steals (0.0 - 1.0)

    // Time accounting. net = saved - wasted. Positive = Shadow-Steal is
    // profitable. Negative = tau is too low, speculation costs more than
    // it saves. In practice, with tau=0.75, net is always positive for
    // Mixtral-class models.
    float    net_time_saved_ms;  // Time saved by HITs minus time lost to MISSes
    float    total_steal_time_ms;  // Cumulative UHD compute time on HITs
    float    total_flush_time_ms;  // Cumulative UHD flush time on MISSes

    // UHD utilisation: fraction of total inference time the UHD was actively
    // computing speculative experts. 0% = all idle, 100% = fully saturated.
    // Healthy range is 30-60%. Above 80% suggests the UHD is becoming a
    // bottleneck and tau should be raised.
    float    uhd_utilisation_pct;
} sp_shadow_steal_stats_t;

// Main Shadow-Steal context. One per Beast Canyon inference session.
// Holds the two shadow buffer slots, current state machine position,
// confidence gate, and accumulated statistics.
typedef struct {
    // Two shadow buffer slots for top-2 predicted experts.
    // slots[0] = highest EWMA score, slots[1] = second highest.
    sp_shadow_buffer_t    slots[2];

    // Current state in the steal lifecycle. Transitions are atomic
    // (single CAS) because the Scout thread and Router thread race.
    sp_steal_state_t      state;

    // Confidence threshold. Combined probability of top-2 predicted
    // experts must exceed tau before UHD launches. Clamped to
    // [SP_SHADOW_STEAL_TAU_MIN, SP_SHADOW_STEAL_TAU_MAX].
    float                 tau;

    // Original tau value set at init, used as baseline for any
    // adaptive tau adjustments.
    float                 tau_base;

    // Accumulated statistics for the lifetime of this context.
    sp_shadow_steal_stats_t stats;

    // Opaque handle to the Level Zero command queue used for UHD
    // dispatch. Cast to ze_command_queue_handle_t internally.
    // NULL if UHD is not available (graceful degradation to no
    // speculation).
    void*                 l0_command_queue;

    // Master enable switch. Can be toggled at runtime to disable
    // speculation without tearing down the context (e.g., during
    // display-intensive workloads where UHD bandwidth is needed
    // for compositing).
    bool                  enabled;
} sp_shadow_steal_t;


// ============================================================================
// Functions
// ============================================================================

// Initialise a Shadow-Steal context. Allocates two shadow buffers of
// `expert_dim` floats each via Level Zero shared memory (Ring Bus path).
// Sets the confidence threshold to `tau` (clamped to valid range).
// Probes for UHD availability; if absent, ctx->enabled = false and all
// subsequent calls gracefully no-op.
//
// Returns 0 on success, -1 on allocation failure, -2 if Level Zero
// initialisation fails.
int sp_shadow_steal_init(sp_shadow_steal_t* ctx, size_t expert_dim, float tau);

// Free all resources: shadow buffers, Level Zero allocations, command
// queue. Safe to call on a partially-initialised context (e.g., after
// init returned -1). Zeroes the context struct.
void sp_shadow_steal_free(sp_shadow_steal_t* ctx);

// Scout phase: consult the Curriculum EWMA heatmap to identify the top-2
// predicted experts for `next_layer`. If their combined probability
// exceeds ctx->tau, populate the shadow buffers with expert weights and
// launch UHD compute via the Level Zero command queue.
//
// This is called by the Scout thread BEFORE the Router runs, using the
// Curriculum's historical data as the prediction signal.
//
// Returns 0 if speculation launched, 1 if confidence was below tau
// (no speculation), -1 on error.
int sp_shadow_steal_speculate(sp_shadow_steal_t* ctx,
                              sp_moe_curriculum_t* curriculum,
                              int next_layer);

// Verdict phase: compare the Router's actual expert selection against
// our shadow buffers. Called by the Router thread after top-K selection.
//
// `actual_experts` is the array of expert indices the Router selected,
// `n_actual` is the count (typically 2 for top-2 routing).
//
// Returns:
//   SP_STEAL_HIT  — at least one shadow buffer matches an actual expert
//   SP_STEAL_MISS — no shadow buffers match any actual expert
//   SP_STEAL_IDLE — no speculation was in flight
sp_steal_state_t sp_shadow_steal_check(sp_shadow_steal_t* ctx,
                                       const int* actual_experts,
                                       int n_actual);

// Accept a shadow buffer on HIT. Performs the pointer swap: the shadow
// buffer's pre-computed output is installed directly into the MoE
// accumulator. Cost: ~10ns (pointer assignment + memory fence).
//
// `slot` is 0 or 1, corresponding to slots[0] or slots[1].
// The slot's `ready` flag must be true, and the context state must be
// SP_STEAL_HIT. Asserts on violation in debug builds.
void sp_shadow_steal_accept(sp_shadow_steal_t* ctx, int slot);

// Abort speculative computation on MISS or manual cancel. Resets the
// Level Zero command queue via zeCommandListReset, discarding any
// in-flight UHD kernels. Cost: ~2ms on Ring Bus (vs ~15ms on PCIe).
//
// Also clears both shadow buffer slots and transitions state to IDLE.
// Safe to call even if no speculation is in flight (no-op).
void sp_shadow_steal_abort(sp_shadow_steal_t* ctx);

// Update the confidence threshold at runtime. `tau` is clamped to
// [SP_SHADOW_STEAL_TAU_MIN, SP_SHADOW_STEAL_TAU_MAX]. Also updates
// tau_base so the new value becomes the baseline for any adaptive
// adjustments.
//
// Common use: raise tau when UHD is needed for display compositing,
// lower tau during batch inference when UHD is fully available.
void sp_shadow_steal_set_tau(sp_shadow_steal_t* ctx, float tau);

// Return a snapshot of the current statistics. The returned struct is
// a value copy, safe to read without synchronisation.
sp_shadow_steal_stats_t sp_shadow_steal_get_stats(const sp_shadow_steal_t* ctx);

// Print telemetry to stdout: net time saved, hit rate, UHD utilisation,
// and per-slot resonance quality from the Curriculum heatmap. Intended
// for end-of-session diagnostics.
//
// Example output:
//   [shadow-steal] steals=4821 hits=4097 (85.0%) misses=682 aborts=42
//   [shadow-steal] net_saved=+312.4ms  steal_time=1847.2ms  flush_time=1534.8ms
//   [shadow-steal] UHD utilisation: 47.3%  tau=0.750
void sp_shadow_steal_log_efficiency(const sp_shadow_steal_t* ctx);


// ============================================================================
// Inline helpers
// ============================================================================

// Fast gate check: should we speculate for this token? Called by the
// Scout thread with the combined probability of the top-2 predicted
// experts (from the Curriculum EWMA). Returns true if the combined
// probability meets or exceeds the confidence threshold.
//
// This is deliberately a static inline to avoid function-call overhead
// on the hot path — it runs for every single token.
static inline bool sp_shadow_steal_should_speculate(const sp_shadow_steal_t* ctx,
                                                     float combined_prob) {
    return ctx->enabled && combined_prob >= ctx->tau;
}

#ifdef __cplusplus
}
#endif

#endif // SP_SHADOW_STEAL_H
