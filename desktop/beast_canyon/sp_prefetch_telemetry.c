// Shannon-Prime Beast Canyon: Prefetch Telemetry Dashboard
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "sp_prefetch_telemetry.h"

#include <string.h>

// ---------------------------------------------------------------------------
//  sp_telemetry_init
// ---------------------------------------------------------------------------
void sp_telemetry_init(sp_prefetch_telemetry_t* t) {
    if (!t) return;
    memset(t, 0, sizeof(*t));
}

// ---------------------------------------------------------------------------
//  sp_telemetry_record_lookahead
// ---------------------------------------------------------------------------
void sp_telemetry_record_lookahead(sp_prefetch_telemetry_t* t,
                                   int layer_id,
                                   const int* actual_experts,
                                   int n_actual,
                                   int predicted_primary,
                                   int predicted_secondary) {
    if (!t) return;
    (void)layer_id;

    t->total_lookaheads++;

    // Check primary prediction against actual selection.
    bool primary_hit = false;
    bool secondary_hit = false;

    for (int i = 0; i < n_actual; i++) {
        if (actual_experts[i] == predicted_primary) {
            primary_hit = true;
            break;
        }
    }

    if (primary_hit) {
        t->primary_hits++;
        t->high_confidence_events++;
    } else if (predicted_secondary >= 0) {
        // Check secondary prediction.
        for (int i = 0; i < n_actual; i++) {
            if (actual_experts[i] == predicted_secondary) {
                secondary_hit = true;
                break;
            }
        }
        if (secondary_hit) {
            t->secondary_hits++;
            t->high_confidence_events++;
        } else {
            t->total_misses++;
            t->low_confidence_events++;
        }
    } else {
        t->total_misses++;
        t->low_confidence_events++;
    }
}

// ---------------------------------------------------------------------------
//  sp_telemetry_record_steal
// ---------------------------------------------------------------------------
void sp_telemetry_record_steal(sp_prefetch_telemetry_t* t,
                               float steal_time_ms,
                               bool was_hit) {
    if (!t) return;

    t->total_steal_time_ms += steal_time_ms;

    if (!was_hit) {
        t->total_flush_time_ms += steal_time_ms;
    }
}

// ---------------------------------------------------------------------------
//  sp_telemetry_record_thermal_zone
// ---------------------------------------------------------------------------
void sp_telemetry_record_thermal_zone(sp_prefetch_telemetry_t* t, int zone) {
    if (!t) return;

    switch (zone) {
        case 0:  t->zone_a_tokens++; break;
        case 1:  t->zone_b_tokens++; break;
        case 2:  t->zone_c_tokens++; break;
        default: t->zone_b_tokens++; break;  // default to nominal
    }
}

// ---------------------------------------------------------------------------
//  sp_telemetry_log_pulse
// ---------------------------------------------------------------------------
void sp_telemetry_log_pulse(sp_prefetch_telemetry_t* t, int layer_id) {
    if (!t) return;
    if (t->total_lookaheads == 0 || (t->total_lookaheads % 128) != 0) return;

    sp_telemetry_update_rates(t);

    fprintf(stderr, "[SP-TELEM L%d] %llu lookaheads | pri=%.1f%% sec=%.1f%% res=%.1f%% | "
                    "steal=%.1fms flush=%.1fms net=%.1fms\n",
            layer_id,
            (unsigned long long)t->total_lookaheads,
            t->primary_hit_rate,
            t->secondary_hit_rate,
            t->total_resonance,
            t->total_steal_time_ms,
            t->total_flush_time_ms,
            t->net_benefit_ms);
}

// ---------------------------------------------------------------------------
//  sp_telemetry_log_summary
// ---------------------------------------------------------------------------
void sp_telemetry_log_summary(sp_prefetch_telemetry_t* t) {
    if (!t) return;
    sp_telemetry_update_rates(t);

    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║         Shannon-Prime Prefetch Telemetry        ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════╣\n");

    fprintf(stderr, "║  Total lookaheads  : %10llu                ║\n",
            (unsigned long long)t->total_lookaheads);
    fprintf(stderr, "║  Primary hits      : %10llu  (%5.1f%%)       ║\n",
            (unsigned long long)t->primary_hits, t->primary_hit_rate);
    fprintf(stderr, "║  Secondary hits    : %10llu  (%5.1f%%)       ║\n",
            (unsigned long long)t->secondary_hits, t->secondary_hit_rate);
    fprintf(stderr, "║  Misses            : %10llu                ║\n",
            (unsigned long long)t->total_misses);
    fprintf(stderr, "║  Total resonance   :             %5.1f%%        ║\n",
            t->total_resonance);

    fprintf(stderr, "╠══════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Steal time (ms)   : %10.1f                ║\n",
            t->total_steal_time_ms);
    fprintf(stderr, "║  Flush time (ms)   : %10.1f                ║\n",
            t->total_flush_time_ms);
    fprintf(stderr, "║  Net benefit (ms)  : %+10.1f  %s       ║\n",
            t->net_benefit_ms,
            t->net_benefit_ms >= 0.0f ? "[WIN]" : "[LOSS]");

    fprintf(stderr, "╠══════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  UHD utilisation   :             %5.1f%%        ║\n",
            t->uhd_utilisation_pct);

    // Thermal zone distribution
    uint64_t total_zone = t->zone_a_tokens + t->zone_b_tokens + t->zone_c_tokens;
    if (total_zone > 0) {
        fprintf(stderr, "╠══════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║  Thermal Zones:                                 ║\n");
        fprintf(stderr, "║    Cool    (A): %10llu  (%5.1f%%)           ║\n",
                (unsigned long long)t->zone_a_tokens,
                (float)t->zone_a_tokens * 100.0f / (float)total_zone);
        fprintf(stderr, "║    Nominal (B): %10llu  (%5.1f%%)           ║\n",
                (unsigned long long)t->zone_b_tokens,
                (float)t->zone_b_tokens * 100.0f / (float)total_zone);
        fprintf(stderr, "║    Hot     (C): %10llu  (%5.1f%%)           ║\n",
                (unsigned long long)t->zone_c_tokens,
                (float)t->zone_c_tokens * 100.0f / (float)total_zone);
    }

    // Confidence distribution
    uint64_t total_conf = t->high_confidence_events + t->low_confidence_events;
    if (total_conf > 0) {
        fprintf(stderr, "╠══════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║  Confidence:                                    ║\n");
        fprintf(stderr, "║    High: %10llu  (%5.1f%%)                  ║\n",
                (unsigned long long)t->high_confidence_events,
                (float)t->high_confidence_events * 100.0f / (float)total_conf);
        fprintf(stderr, "║    Low:  %10llu  (%5.1f%%)                  ║\n",
                (unsigned long long)t->low_confidence_events,
                (float)t->low_confidence_events * 100.0f / (float)total_conf);
    }

    fprintf(stderr, "╚══════════════════════════════════════════════════╝\n\n");
}
