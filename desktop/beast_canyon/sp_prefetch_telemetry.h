/*
 * sp_prefetch_telemetry.h — Shannon-Prime Beast Canyon Prefetch Telemetry
 *
 * Dashboard for tracking all speculation metrics: lookahead hit rates,
 * shadow-steal efficiency, thermal zone distribution, and net benefit
 * from prefetch speculation across MoE expert dispatch.
 *
 * Copyright (c) 2026 Ray Daniels
 * Licensed under the GNU Affero General Public License v3.0
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef SP_PREFETCH_TELEMETRY_H
#define SP_PREFETCH_TELEMETRY_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------
 * Telemetry snapshot — one per inference context.
 *
 * Primary hit   = predicted expert matches top-1 actual.
 * Secondary hit = predicted expert matches top-2..top-K actual.
 * Resonance     = combined (primary + secondary) / total_lookaheads.
 *
 * Steal time    = wall-clock spent on speculative prefetch work.
 * Flush time    = wall-clock spent discarding mispredicted prefetch data.
 * Net benefit   = steal_time - flush_time (positive = we won).
 *
 * Thermal zones track where tokens land during speculation:
 *   Zone A (cool)    = low utilisation, plenty of headroom.
 *   Zone B (nominal) = healthy operating range.
 *   Zone C (hot)     = near throttle, speculation should back off.
 * -------------------------------------------------------------------- */

typedef struct {
    /* --- Core speculation counters --- */
    uint64_t total_lookaheads;
    uint64_t primary_hits;
    uint64_t secondary_hits;
    uint64_t total_misses;

    /* --- Derived rates (updated via sp_telemetry_update_rates) --- */
    float    primary_hit_rate;      /* primary_hits / total_lookaheads * 100  */
    float    secondary_hit_rate;    /* secondary_hits / total_lookaheads * 100 */
    float    total_resonance;       /* (primary + secondary) / total * 100    */

    /* --- Timing (milliseconds, cumulative) --- */
    float    total_steal_time_ms;   /* time spent on speculative work         */
    float    total_flush_time_ms;   /* time spent flushing mispredictions     */
    float    net_benefit_ms;        /* steal - flush; positive = net win      */

    /* --- UHD utilisation --- */
    float    uhd_utilisation_pct;   /* Intel UHD SVM bandwidth utilisation    */

    /* --- Confidence distribution --- */
    uint64_t high_confidence_events;
    uint64_t low_confidence_events;

    /* --- Thermal zone token counters --- */
    uint64_t zone_a_tokens;         /* zone 0: cool                          */
    uint64_t zone_b_tokens;         /* zone 1: nominal                       */
    uint64_t zone_c_tokens;         /* zone 2: hot                           */
} sp_prefetch_telemetry_t;

/* --------------------------------------------------------------------
 * sp_telemetry_init — zero all counters and rates.
 * -------------------------------------------------------------------- */
void sp_telemetry_init(sp_prefetch_telemetry_t* t);

/* --------------------------------------------------------------------
 * sp_telemetry_record_lookahead — record the outcome of one prefetch
 * speculation event.
 *
 *   layer_id           — transformer layer that triggered the lookahead.
 *   actual_experts     — array of expert indices that were actually routed.
 *   n_actual           — length of actual_experts[].
 *   predicted_primary  — the primary (top-1) expert we speculated on.
 *   predicted_secondary — the secondary (top-2) expert we speculated on,
 *                         or -1 if no secondary prediction was made.
 *
 * Increments total_lookaheads, and one of primary_hits / secondary_hits /
 * total_misses depending on whether predicted_primary or predicted_secondary
 * appears in actual_experts[].
 * -------------------------------------------------------------------- */
void sp_telemetry_record_lookahead(sp_prefetch_telemetry_t* t,
                                   int layer_id,
                                   const int* actual_experts,
                                   int n_actual,
                                   int predicted_primary,
                                   int predicted_secondary);

/* --------------------------------------------------------------------
 * sp_telemetry_record_steal — record a shadow-steal event.
 *
 *   steal_time_ms — wall-clock time consumed by the speculative prefetch.
 *   was_hit       — true if the prefetched data was actually used.
 *
 * Accumulates into total_steal_time_ms.  If !was_hit, also accumulates
 * into total_flush_time_ms (the prefetch was wasted and had to be flushed).
 * -------------------------------------------------------------------- */
void sp_telemetry_record_steal(sp_prefetch_telemetry_t* t,
                               float steal_time_ms,
                               bool was_hit);

/* --------------------------------------------------------------------
 * sp_telemetry_record_thermal_zone — classify current token into a
 * thermal zone.
 *
 *   zone: 0 = cool (Zone A), 1 = nominal (Zone B), 2 = hot (Zone C).
 *
 * When zone == 2 (hot), the prefetch oracle should reduce speculation
 * depth to avoid pushing the system into thermal throttling.
 * -------------------------------------------------------------------- */
void sp_telemetry_record_thermal_zone(sp_prefetch_telemetry_t* t, int zone);

/* --------------------------------------------------------------------
 * sp_telemetry_log_pulse — periodic printf every 128 lookaheads.
 *
 * Emits a one-line summary to stderr:
 *   [SP-TELEM L<layer>] <n> lookaheads | pri=XX.X% sec=YY.Y% res=ZZ.Z% |
 *     steal=SS.Sms flush=FF.Fms net=NN.Nms
 *
 * Call this after every sp_telemetry_record_lookahead(); it no-ops
 * unless total_lookaheads is a multiple of 128.
 * -------------------------------------------------------------------- */
void sp_telemetry_log_pulse(sp_prefetch_telemetry_t* t, int layer_id);

/* --------------------------------------------------------------------
 * sp_telemetry_log_summary — full end-of-inference summary.
 *
 * Prints to stderr:
 *   - Total lookaheads and hit/miss breakdown.
 *   - Primary / secondary / resonance hit rates.
 *   - Net benefit (steal - flush) with verdict.
 *   - UHD utilisation percentage.
 *   - Thermal zone distribution (A/B/C token counts and percentages).
 *   - Confidence event distribution (high vs low).
 * -------------------------------------------------------------------- */
void sp_telemetry_log_summary(sp_prefetch_telemetry_t* t);

/* --------------------------------------------------------------------
 * sp_telemetry_update_rates — recompute derived rates from counters.
 *
 * Static inline so it can be called from hot paths without function-call
 * overhead.  Updates primary_hit_rate, secondary_hit_rate, total_resonance,
 * and net_benefit_ms.
 * -------------------------------------------------------------------- */
static inline void sp_telemetry_update_rates(sp_prefetch_telemetry_t* t) {
    if (t->total_lookaheads == 0) return;
    t->primary_hit_rate = (float)t->primary_hits * 100.0f / (float)t->total_lookaheads;
    t->secondary_hit_rate = (float)t->secondary_hits * 100.0f / (float)t->total_lookaheads;
    t->total_resonance = (float)(t->primary_hits + t->secondary_hits) * 100.0f / (float)t->total_lookaheads;
    t->net_benefit_ms = t->total_steal_time_ms - t->total_flush_time_ms;
}

#ifdef __cplusplus
}
#endif

#endif /* SP_PREFETCH_TELEMETRY_H */
