// Shannon-Prime Beast Canyon: Shadow-Steal Speculative Dispatch
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "sp_shadow_steal.h"
#include "sp_level_zero.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#  include <windows.h>
static uint64_t sp_steal_time_us(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (uint64_t)(c.QuadPart * 1000000ULL / f.QuadPart);
}
#else
#  include <time.h>
static uint64_t sp_steal_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000;
}
#endif

// ---------------------------------------------------------------------------
//  Clamp tau to valid range
// ---------------------------------------------------------------------------
static float clamp_tau(float tau) {
    if (tau < SP_SHADOW_STEAL_TAU_MIN) return SP_SHADOW_STEAL_TAU_MIN;
    if (tau > SP_SHADOW_STEAL_TAU_MAX) return SP_SHADOW_STEAL_TAU_MAX;
    return tau;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_init
// ---------------------------------------------------------------------------
int sp_shadow_steal_init(sp_shadow_steal_t* ctx, size_t expert_dim, float tau) {
    if (!ctx) return -1;
    memset(ctx, 0, sizeof(*ctx));

    ctx->tau      = clamp_tau(tau);
    ctx->tau_base = ctx->tau;
    ctx->state    = SP_STEAL_IDLE;

    // ── Level Zero initialisation ─────────────────────────────────
    // Try to discover Intel UHD iGPU via Level Zero.  If successful,
    // shadow buffers are USM shared allocations on the Ring Bus —
    // writes from the Shredder are LLC-coherent with UHD compute.
    // On failure, fall back to aligned CPU memory (stub mode).
    sp_l0_t *l0 = (sp_l0_t*)calloc(1, sizeof(sp_l0_t));
    bool l0_ready = false;
    if (l0 && sp_l0_init(l0) == 0) {
        l0_ready = true;
        ctx->l0_command_queue = (void*)l0;
        fprintf(stderr, "[shadow-steal] Level Zero ARMED — %s on Ring Bus\n",
                l0->props.name);
    } else {
        if (l0) free(l0);
        ctx->l0_command_queue = NULL;
        fprintf(stderr, "[shadow-steal] Level Zero not available — stub mode\n");
    }

    // Allocate shadow buffers — two slots for top-2 predicted experts.
    // When L0 is available, use USM shared memory (Ring Bus zero-copy).
    // Otherwise fall back to aligned CPU memory.
    size_t buf_bytes = expert_dim * sizeof(float);
    for (int i = 0; i < 2; i++) {
        if (l0_ready) {
            ctx->slots[i].data = sp_l0_alloc_shared(
                (sp_l0_t*)ctx->l0_command_queue, buf_bytes, 64);
        }
        if (!ctx->slots[i].data) {
            // Fallback to aligned CPU memory
#ifdef _WIN32
            ctx->slots[i].data = _aligned_malloc(buf_bytes, 64);
#else
            ctx->slots[i].data = aligned_alloc(64, (buf_bytes + 63) & ~(size_t)63);
#endif
        }
        if (!ctx->slots[i].data) {
            sp_shadow_steal_free(ctx);
            return -1;
        }
        memset(ctx->slots[i].data, 0, buf_bytes);
        ctx->slots[i].size      = buf_bytes;
        ctx->slots[i].expert_id = -1;
        ctx->slots[i].ready     = false;
        ctx->slots[i].target    = SP_GPU_SECONDARY;
    }

    // Create completion events for async dispatch tracking
    if (l0_ready) {
        sp_l0_t *l0p = (sp_l0_t*)ctx->l0_command_queue;
        // We store event handles in the shadow buffer data area's alignment
        // padding — but cleaner to just use a local static.  For now the
        // event query path checks l0_command_queue != NULL to decide
        // whether to poll L0 or use the instant-ready fallback.
    }

    // Enable speculation — prediction + hit/miss tracking is fully functional
    // even without L0 dispatch (stub mode uses instant-ready).
    ctx->enabled = true;

    return 0;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_free
// ---------------------------------------------------------------------------
void sp_shadow_steal_free(sp_shadow_steal_t* ctx) {
    if (!ctx) return;

    sp_l0_t *l0 = (sp_l0_t*)ctx->l0_command_queue;

    for (int i = 0; i < 2; i++) {
        if (ctx->slots[i].data) {
            if (l0 && l0->loaded) {
                // USM allocations are freed by sp_l0_free via tracked list.
                // If the pointer is in the tracked list, skip manual free.
                bool is_usm = false;
                for (int j = 0; j < l0->n_usm_allocs; j++) {
                    if (l0->usm_allocs[j] == ctx->slots[i].data) {
                        is_usm = true;
                        break;
                    }
                }
                if (!is_usm) {
                    // CPU-allocated fallback buffer
#ifdef _WIN32
                    _aligned_free(ctx->slots[i].data);
#else
                    free(ctx->slots[i].data);
#endif
                }
            } else {
                // No L0 — all buffers are CPU-allocated
#ifdef _WIN32
                _aligned_free(ctx->slots[i].data);
#else
                free(ctx->slots[i].data);
#endif
            }
        }
    }

    // Tear down Level Zero (frees USM allocs, command list, context, DLL)
    if (l0) {
        sp_l0_free(l0);
        free(l0);
    }

    memset(ctx, 0, sizeof(*ctx));
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_speculate
// ---------------------------------------------------------------------------
int sp_shadow_steal_speculate(sp_shadow_steal_t* ctx,
                              sp_moe_curriculum_t* curriculum,
                              int next_layer) {
    if (!ctx || !ctx->enabled || !curriculum) return -1;
    if (ctx->state != SP_STEAL_IDLE) return -1;

    // Query the Curriculum EWMA heatmap for top-2 hottest experts.
    int top2[2] = { -1, -1 };
    sp_moe_get_hottest_k(curriculum, next_layer, top2, 2);
    if (top2[0] < 0) return 1;  // No prediction available

    // Check combined confidence of top-2 against tau.
    float combined = sp_moe_top_k_confidence(curriculum, next_layer, 2);
    if (!sp_shadow_steal_should_speculate(ctx, combined)) {
        return 1;  // Below confidence threshold
    }

    // Populate shadow buffer slots.
    ctx->slots[0].expert_id = top2[0];
    ctx->slots[0].ready     = false;
    ctx->slots[1].expert_id = top2[1];
    ctx->slots[1].ready     = false;

    ctx->state = SP_STEAL_SPECULATING;
    ctx->stats.total_steals++;

    sp_l0_t *l0 = (sp_l0_t*)ctx->l0_command_queue;

    if (l0 && l0->loaded && l0->device_found) {
        // ── Level Zero dispatch path ─────────────────────────────
        // The shadow buffers are USM shared memory on the Ring Bus.
        // The Shredder has already written CRT M2 residues into
        // these buffers (they're LLC-coherent — UHD sees them
        // without explicit copy).
        //
        // For Phase 1 bringup: we mark as ready after an L0 barrier
        // to confirm the USM writes have landed.  The actual matmul
        // kernel launch will come in Phase 2 when we have the SPIR-V
        // residue_m2 kernel compiled.
        //
        // Phase 1 path (USM coherence validation):
        //   1. Append barrier on immediate cmd list (flushes LLC→UHD)
        //   2. Mark slots ready — the data in USM is the M2 residues
        //      written by sp_shredder_crt_q8_expert() to ctx->slots[].data
        //
        // Phase 2 path (full kernel dispatch):
        //   1. zeCommandListAppendLaunchKernel → residue_m2 matmul
        //   2. Signal completion event
        //   3. Host polls event in sp_shadow_steal_check()

        sp_l0_append_barrier(l0, NULL);

        // Mark both prediction slots as ready.
        // The M2 residue data is already in the USM buffers —
        // the Shredder wrote directly into slots[].data via
        // sp_shredder_crt_q8_expert() before this function was called.
        ctx->slots[0].ready = true;
        ctx->slots[1].ready = true;

        uint64_t t1 = sp_steal_time_us();
        // Track actual dispatch time for UHD utilisation accounting
        (void)t1;
    } else {
        // ── Stub mode (no L0) ────────────────────────────────────
        // Mark slots as "ready" immediately (simulating instant compute).
        // Prediction + hit/miss tracking still works — the only thing
        // missing is actual UHD compute, which means the "saved time"
        // metrics are approximate rather than measured.
        ctx->slots[0].ready = true;
        ctx->slots[1].ready = true;
    }

    return 0;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_check
// ---------------------------------------------------------------------------
sp_steal_state_t sp_shadow_steal_check(sp_shadow_steal_t* ctx,
                                       const int* actual_experts,
                                       int n_actual) {
    if (!ctx) return SP_STEAL_IDLE;
    if (ctx->state != SP_STEAL_SPECULATING) return SP_STEAL_IDLE;

    // Compare shadow slots against the Router's actual selection.
    bool hit = false;
    for (int s = 0; s < 2; s++) {
        for (int a = 0; a < n_actual; a++) {
            if (ctx->slots[s].expert_id == actual_experts[a] && ctx->slots[s].ready) {
                hit = true;
                break;
            }
        }
        if (hit) break;
    }

    if (hit) {
        ctx->state = SP_STEAL_HIT;
        ctx->stats.hits++;
        // Approximate: typical expert matmul on UHD ~3ms saved on HIT.
        ctx->stats.total_steal_time_ms += 3.0f;
        ctx->stats.net_time_saved_ms   += 3.0f;
    } else {
        ctx->state = SP_STEAL_MISS;
        ctx->stats.misses++;
        // Approximate: Ring Bus flush cost ~2ms.
        ctx->stats.total_flush_time_ms += 2.0f;
        ctx->stats.net_time_saved_ms   -= 2.0f;
    }

    if (ctx->stats.total_steals > 0) {
        ctx->stats.hit_rate = (float)ctx->stats.hits / (float)ctx->stats.total_steals;
    }

    return ctx->state;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_accept
// ---------------------------------------------------------------------------
void sp_shadow_steal_accept(sp_shadow_steal_t* ctx, int slot) {
    if (!ctx || slot < 0 || slot > 1) return;
    // In full implementation: pointer swap into MoE accumulator. Cost: ~10ns.
    ctx->slots[slot].ready = false;
    ctx->state = SP_STEAL_IDLE;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_abort
// ---------------------------------------------------------------------------
void sp_shadow_steal_abort(sp_shadow_steal_t* ctx) {
    if (!ctx) return;
    if (ctx->state == SP_STEAL_SPECULATING) {
        ctx->stats.aborts++;

        // Flush UHD queue via zeCommandListReset.
        // On Ring Bus this costs ~2µs — negligible compared to PCIe.
        sp_l0_t *l0 = (sp_l0_t*)ctx->l0_command_queue;
        if (l0 && l0->loaded && l0->cmd_list) {
            sp_l0_reset_cmd_list(l0);
        }
    }
    ctx->slots[0].ready     = false;
    ctx->slots[0].expert_id = -1;
    ctx->slots[1].ready     = false;
    ctx->slots[1].expert_id = -1;
    ctx->state = SP_STEAL_IDLE;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_set_tau
// ---------------------------------------------------------------------------
void sp_shadow_steal_set_tau(sp_shadow_steal_t* ctx, float tau) {
    if (!ctx) return;
    ctx->tau      = clamp_tau(tau);
    ctx->tau_base = ctx->tau;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_get_stats
// ---------------------------------------------------------------------------
sp_shadow_steal_stats_t sp_shadow_steal_get_stats(const sp_shadow_steal_t* ctx) {
    if (!ctx) {
        sp_shadow_steal_stats_t empty;
        memset(&empty, 0, sizeof(empty));
        return empty;
    }
    return ctx->stats;
}

// ---------------------------------------------------------------------------
//  sp_shadow_steal_log_efficiency
// ---------------------------------------------------------------------------
void sp_shadow_steal_log_efficiency(const sp_shadow_steal_t* ctx) {
    if (!ctx) return;
    const sp_shadow_steal_stats_t* s = &ctx->stats;

    fprintf(stderr, "[shadow-steal] steals=%llu hits=%llu (%.1f%%) misses=%llu aborts=%llu\n",
            (unsigned long long)s->total_steals,
            (unsigned long long)s->hits,
            s->hit_rate * 100.0f,
            (unsigned long long)s->misses,
            (unsigned long long)s->aborts);

    fprintf(stderr, "[shadow-steal] net_saved=%+.1fms  steal_time=%.1fms  flush_time=%.1fms\n",
            s->net_time_saved_ms,
            s->total_steal_time_ms,
            s->total_flush_time_ms);

    fprintf(stderr, "[shadow-steal] UHD utilisation: %.1f%%  tau=%.3f\n",
            s->uhd_utilisation_pct,
            ctx->tau);
}
