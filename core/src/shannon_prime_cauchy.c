// Shannon-Prime: Cauchy Reset System
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Three-layer navigation stack for decode-chain causal stability:
//   Layer 1: Zeta Schedule  (pre-computed, static)    — tidal chart
//   Layer 2: Mertens Oracle (proactive, arithmetic)   — barometer
//   Layer 3: Ricci Sentinel (reactive, empirical)     — altimeter

#include "shannon_prime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>   // calloc / free / malloc / abs — REQUIRED, without it
                      // MSVC treats calloc as implicit-int, truncating the
                      // 64-bit pointer to 32 bits and sign-extending on use.
#include <string.h>

// ============================================================================
// Layer 3: Ricci Sentinel — p=3 band energy drift monitor
// ============================================================================

int sp_ricci_init(sp_ricci_sentinel_t *rs, const sp_band_config_t *band_cfg,
                  float params_b) {
    memset(rs, 0, sizeof(*rs));

    // The p=3 band is band index 1 (band 0 = p=2, band 1 = p=3).
    // If fewer than 2 bands, degenerate — monitor the whole vector.
    if (band_cfg->n_bands >= 2) {
        sp_band_span(band_cfg, 1, &rs->p3_band_offset, &rs->p3_band_size);
    } else {
        rs->p3_band_offset = 0;
        rs->p3_band_size   = band_cfg->head_dim;
    }

    // EMA parameters
    rs->ema_alpha = 0.05;
    rs->p3_ema    = 1.0;  // Starts at "no drift"
    rs->n_samples = 0;

    // Model-size-dependent threshold via params^1.1 scaling.
    // At 1B: threshold ~ 0.05 (fragile metric)
    // At 8B: threshold ~ 0.15 (robust metric)
    // At 70B: threshold ~ 0.25 (very robust)
    // Formula: 0.05 * params_b^0.45 (empirically derived)
    if (params_b > 0.0f) {
        rs->metric_criticality = 0.05 * pow((double)params_b, 0.45);
        if (rs->metric_criticality > 0.30)
            rs->metric_criticality = 0.30;  // cap
        if (rs->metric_criticality < 0.03)
            rs->metric_criticality = 0.03;  // floor
    } else {
        rs->metric_criticality = 0.05;  // conservative default
    }

    rs->calibrated = false;
    rs->reset_recommended = false;
    // Warm-up gate: 32 samples of EMA bedding in before any reset can
    // be recommended. Stops the runaway where ricci_reset zeros the
    // EMA and the very next sample immediately re-fires.
    rs->warmup_samples = 32;
    return 0;
}

// Compute p=3 band energy from a VHT2-domain vector.
static double compute_p3_energy(const sp_ricci_sentinel_t *rs,
                                 const float *vht2_coeffs, int hd) {
    if (rs->p3_band_offset + rs->p3_band_size > hd)
        return 0.0;

    double energy = 0.0;
    const float *p3 = vht2_coeffs + rs->p3_band_offset;
    for (int i = 0; i < rs->p3_band_size; i++) {
        energy += (double)p3[i] * (double)p3[i];
    }
    return energy;
}

void sp_ricci_calibrate_feed(sp_ricci_sentinel_t *rs,
                             const float *vht2_coeffs, int hd) {
    double e = compute_p3_energy(rs, vht2_coeffs, hd);
    rs->calib_p3_sum += e;
    rs->calib_n++;
}

void sp_ricci_calibrate_end(sp_ricci_sentinel_t *rs) {
    if (rs->calib_n > 0) {
        rs->p3_energy_calibrated = rs->calib_p3_sum / rs->calib_n;
    } else {
        rs->p3_energy_calibrated = 1.0;  // fallback
    }
    rs->calibrated = true;
    rs->p3_ema     = 1.0;
    rs->n_samples  = 0;
    rs->reset_recommended = false;

    // Reset transient accumulators
    rs->calib_p3_sum = 0.0;
    rs->calib_n      = 0;
}

bool sp_ricci_check(sp_ricci_sentinel_t *rs,
                    const float *vht2_coeffs, int hd) {
    if (!rs->calibrated || rs->p3_energy_calibrated <= 0.0)
        return false;

    double e = compute_p3_energy(rs, vht2_coeffs, hd);
    double ratio = e / rs->p3_energy_calibrated;

    // EMA update — always runs so the EMA converges during warm-up.
    if (rs->n_samples == 0) {
        rs->p3_ema = ratio;
    } else {
        rs->p3_ema = rs->ema_alpha * ratio + (1.0 - rs->ema_alpha) * rs->p3_ema;
    }
    rs->n_samples++;

    // Warm-up gate: suppress reset_recommended until the EMA has
    // bedded in. One-sample EMA is essentially the ratio itself —
    // for lossy compression that's nowhere near 1.0, so an
    // un-gated check re-fires on the very next sample after every
    // reset (runaway). With warmup_samples=32 at alpha=0.05, the
    // EMA is weighted ~80% toward the running mean by the time
    // the gate opens.
    if (rs->n_samples <= rs->warmup_samples) {
        rs->reset_recommended = false;
        return false;
    }

    // Threshold check (post-warm-up).
    double drift = fabs(1.0 - rs->p3_ema);
    rs->reset_recommended = (drift > rs->metric_criticality);
    return rs->reset_recommended;
}

void sp_ricci_reset(sp_ricci_sentinel_t *rs) {
    rs->p3_ema    = 1.0;
    rs->n_samples = 0;
    rs->reset_recommended = false;
}

double sp_ricci_drift(const sp_ricci_sentinel_t *rs) {
    return fabs(1.0 - rs->p3_ema);
}


// ============================================================================
// Layer 1 + 2: Mertens Oracle — zeta-guided proactive scheduling
// ============================================================================

// First 50 non-trivial zeros of the Riemann zeta function (imaginary parts).
// ρ_k = 1/2 + i·γ_k. These are the "harmonics" of the Mertens sea.
// Source: Odlyzko's tables, verified to 13 decimal places.
static const double ZETA_ZEROS_50[50] = {
    14.134725141734693, 21.022039638771555, 25.010857580145688,
    30.424876125859513, 32.935061587739189, 37.586178158825671,
    40.918719012147495, 43.327073280914999, 48.005150881167159,
    49.773832477672302, 52.970321477714460, 56.446247697063394,
    59.347044002602353, 60.831778524609809, 65.112544048081607,
    67.079810529494174, 69.546401711173979, 72.067157674481907,
    75.704690699083933, 77.144840068874805, 79.337375020249367,
    82.910380854086030, 84.735492980517050, 87.425274613125229,
    88.809111207634462, 92.491899270558484, 94.651344040519838,
    95.870634228245309, 98.831194218193692, 101.31785100573139,
    103.72553804532511, 105.44662305232542, 107.16861118427640,
    111.02953554316967, 111.87465917699263, 114.32022091545271,
    116.22668032085755, 118.79078286597621, 121.37012500242066,
    122.94682929355258, 124.25681855434576, 127.51668387959649,
    129.57870420047855, 131.08768853093265, 133.49773720299758,
    134.75650975337387, 138.11604205453344, 139.73620895212138,
    141.12370740402112, 143.11184580762063,
};

// Compute the Möbius function μ(n) directly.
// μ(n) = 0 if n has a squared prime factor
// μ(n) = (-1)^k if n is a product of k distinct primes
static int mobius_mu(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;

    int factors = 0;
    for (int p = 2; (long long)p * p <= n; p++) {
        if (n % p == 0) {
            n /= p;
            if (n % p == 0) return 0;  // p^2 divides original n
            factors++;
        }
    }
    if (n > 1) factors++;  // remaining prime factor
    return (factors % 2 == 0) ? 1 : -1;
}

// Compute M(n) = Σ_{k=1}^{n} μ(k) exactly (brute force, for moderate n).
static int mertens_exact(int n) {
    int sum = 0;
    for (int k = 1; k <= n; k++) {
        sum += mobius_mu(k);
    }
    return sum;
}

// Compute M(n) via truncated explicit formula (smooth approximation).
// Uses the first n_zeros zeta zeros for the oscillatory terms.
double sp_mertens_eval(const sp_mertens_oracle_t *mo, int n) {
    if (n <= 0) return 0.0;

    // The explicit formula: M(x) ≈ Σ_ρ x^ρ / (ρ · ζ'(ρ))
    // We use a simplified version: M(x) ≈ Σ_k  2·x^(1/2)·cos(γ_k·ln(x)) / |ρ_k·ζ'(ρ_k)|
    // Since we don't have ζ'(ρ_k) exactly, we use the known asymptotic
    // |ρ_k·ζ'(ρ_k)| ≈ γ_k·ln(γ_k/(2πe)) which is good enough for our
    // risk prediction (we care about zero-crossings, not exact values).

    double x = (double)n;
    double lnx = log(x);
    double sqrtx = sqrt(x);
    double sum = 0.0;

    for (int k = 0; k < mo->n_zeros; k++) {
        double gamma = mo->gamma[k];
        // Simplified amplitude: decays as 1/(gamma * ln(gamma))
        double amp = 1.0 / (gamma * log(gamma));
        // Oscillatory term
        sum += amp * cos(gamma * lnx);
    }

    // Scale by 2·sqrt(x) (from x^(1/2) in the explicit formula)
    return 2.0 * sqrtx * sum;
}

int sp_mertens_init(sp_mertens_oracle_t *mo, int max_ctx) {
    memset(mo, 0, sizeof(*mo));
    mo->max_ctx = max_ctx;

    // Load zeta zeros
    mo->n_zeros = SP_MERTENS_MAX_ZEROS;
    memcpy(mo->gamma, ZETA_ZEROS_50, sizeof(ZETA_ZEROS_50));

    // Pre-compute risk schedule by finding positions where M(n) is
    // most negative (squarefree-poor regions = high compression risk).
    //
    // Strategy: evaluate M(n) at every position, compute the local
    // derivative (slope over a window), and flag positions where
    // the slope is strongly negative.

    const int window = 16;  // Derivative window
    mo->n_schedule = 0;

    // For large contexts, subsample to keep init cost reasonable.
    // Evaluate every `step` positions. For ctx <= 8K, step=1 (exact).
    int step = 1;
    if (max_ctx > 8192)  step = 2;
    if (max_ctx > 32768) step = 4;

    // We use exact Mertens for moderate n (up to ~32K is fast),
    // and the smooth approximation beyond that.
    int *m_cache = NULL;
    int exact_limit = (max_ctx < 32768) ? max_ctx : 32768;
    m_cache = (int *)calloc((size_t)(exact_limit + 1), sizeof(int));
    if (m_cache) {
        int running = 0;
        for (int k = 1; k <= exact_limit; k++) {
            running += mobius_mu(k);
            m_cache[k] = running;
        }
    }

    for (int n = window + 1; n <= max_ctx && mo->n_schedule < SP_MERTENS_MAX_SCHEDULE;
         n += step) {
        double curr_m;
        if (m_cache && n <= exact_limit) {
            curr_m = (double)m_cache[n];
        } else {
            curr_m = sp_mertens_eval(mo, n);
        }

        // Local slope (negative slope = entering squarefree-poor region)
        double prev_val;
        int prev_n = n - window;
        if (m_cache && prev_n > 0 && prev_n <= exact_limit) {
            prev_val = (double)m_cache[prev_n];
        } else if (prev_n > 0) {
            prev_val = sp_mertens_eval(mo, prev_n);
        } else {
            prev_val = 0.0;
        }

        double slope = (curr_m - prev_val) / (double)window;

        // Flag strongly negative slopes as high-risk positions
        // Threshold: slope < -0.3 (empirically tuned)
        if (slope < -0.3) {
            int idx = mo->n_schedule;
            mo->schedule[idx] = n;
            // Risk score: normalized by sqrt(n) to account for M(n) growth
            double risk_raw = -slope / (sqrt((double)n) * 0.01 + 1.0);
            if (risk_raw > 1.0) risk_raw = 1.0;
            mo->risk[idx] = (float)risk_raw;
            mo->n_schedule++;
        }

    }

    if (m_cache) free(m_cache);

    mo->schedule_idx = 0;
    
    // Bake the explicit schedule into an O(1) bitmask/risk cache mapping
    if (mo->max_ctx > 0) {
        mo->risk_cache = (float *)calloc((size_t)(mo->max_ctx + 1), sizeof(float));
        if (mo->risk_cache) {
            for (int i = 0; i < mo->n_schedule; i++) {
                int p = mo->schedule[i];
                for (int diff = 0; diff <= 8; diff++) {
                    float decay = 1.0f - (float)diff / 8.0f;
                    float val = mo->risk[i] * decay;
                    if (p + diff <= mo->max_ctx && val > mo->risk_cache[p + diff]) 
                        mo->risk_cache[p + diff] = val;
                    if (p - diff >= 0 && val > mo->risk_cache[p - diff]) 
                        mo->risk_cache[p - diff] = val;
                }
            }
        }
    }

    fprintf(stderr, "[sp-mertens] Initialized for ctx=%d: %d high-risk positions flagged, O(1) risk cache %s (%d KB)\n",
            max_ctx, mo->n_schedule,
            mo->risk_cache ? "active" : "FALLBACK",
            mo->risk_cache ? (int)((mo->max_ctx + 1) * sizeof(float) / 1024) : 0);
    return 0;
}

void sp_mertens_free(sp_mertens_oracle_t *mo) {
    if (!mo) return;
    if (mo->risk_cache) {
        free(mo->risk_cache);
        mo->risk_cache = NULL;
    }
    // The schedule / gamma arrays are inline storage inside the struct;
    // zero the whole thing so a re-init on the same memory starts clean
    // and so any dangling pointers bomb loudly rather than silently.
    memset(mo, 0, sizeof(*mo));
}

float sp_mertens_risk(const sp_mertens_oracle_t *mo, int pos) {
    if (mo->risk_cache && pos >= 0 && pos <= mo->max_ctx) {
        return mo->risk_cache[pos];
    }
    
    // fallback to binary search if not mapped
    if (mo->n_schedule == 0) return 0.0f;

    int lo = 0, hi = mo->n_schedule - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (mo->schedule[mid] < pos)
            lo = mid + 1;
        else
            hi = mid;
    }

    // Check if pos is at or near a flagged position (within ±8)
    for (int i = (lo > 0 ? lo - 1 : 0);
         i < mo->n_schedule && i <= lo + 1; i++) {
        int diff = abs(mo->schedule[i] - pos);
        if (diff <= 8) {
            // Decay risk with distance from flagged position
            float decay = 1.0f - (float)diff / 8.0f;
            return mo->risk[i] * decay;
        }
    }
    return 0.0f;
}

int sp_mertens_next_risk(const sp_mertens_oracle_t *mo,
                         int current_pos, int lookahead) {
    for (int i = mo->schedule_idx; i < mo->n_schedule; i++) {
        int p = mo->schedule[i];
        if (p > current_pos && p <= current_pos + lookahead) {
            return p;
        }
        if (p > current_pos + lookahead) break;
    }
    return -1;
}

void sp_mertens_advance(sp_mertens_oracle_t *mo, int pos) {
    while (mo->schedule_idx < mo->n_schedule &&
           mo->schedule[mo->schedule_idx] <= pos) {
        mo->schedule_idx++;
    }
}


// ============================================================================
// Cauchy Reset Controller
// ============================================================================

void sp_cauchy_init(sp_cauchy_ctrl_t *cc, int mode, int fixed_n,
                    sp_ricci_sentinel_t *ricci,
                    sp_mertens_oracle_t *mertens) {
    memset(cc, 0, sizeof(*cc));
    cc->mode           = mode;
    cc->fixed_n        = (fixed_n > 0) ? fixed_n : 512;
    cc->partial_window = 64;  // default: re-accumulate last 64 tokens
    cc->last_reset_pos = -999999;  // allow first reset immediately
    cc->total_resets   = 0;
    cc->ricci          = ricci;
    cc->mertens        = mertens;
}

int sp_cauchy_check(sp_cauchy_ctrl_t *cc, int pos) {
    if (cc->mode == 0) return 0;  // off

    // Cooldown gate: no reset within `partial_window` positions of
    // the last one. Without this the Mertens lookahead (which fires
    // within 32 steps of every scheduled position, at average gap 16)
    // plus Ricci drift after EMA zero produce runaway resets —
    // measured 115 resets in a 510-token chunk, each re-prefill
    // taking ~5s. Cooldown guarantees resets fire at least
    // partial_window positions apart, independent of trigger source.
    if (pos - cc->last_reset_pos < cc->partial_window) {
        return 0;
    }

    // Mode 1: fixed-N
    if (cc->mode == 1) {
        if (pos - cc->last_reset_pos >= cc->fixed_n)
            return 1;  // full reset
        return 0;
    }

    // Mode 2: dynamic (Ricci + Mertens)
    if (cc->mode == 2) {
        // Layer 3: Ricci sentinel — reactive, highest priority
        if (cc->ricci && cc->ricci->reset_recommended) {
            return 1;  // full reset — metric is failing
        }

        // Layer 1+2: Mertens oracle — proactive
        if (cc->mertens) {
            float risk = sp_mertens_risk(cc->mertens, pos);
            if (risk > 0.5f) {
                // High risk from the zeta schedule. If we have a
                // hierarchical cache, a partial reset suffices.
                return 2;  // partial reset OK
            }

            // Look ahead: is a high-risk position coming within 32 steps?
            int next = sp_mertens_next_risk(cc->mertens, pos, 32);
            if (next >= 0) {
                // Pre-emptive partial reset before the risk zone
                return 2;
            }
        }

        return 0;
    }

    return 0;  // unknown mode
}

void sp_cauchy_record_reset(sp_cauchy_ctrl_t *cc, int pos) {
    cc->last_reset_pos = pos;
    cc->total_resets++;

    // Reset the Ricci sentinel's EMA
    if (cc->ricci) {
        sp_ricci_reset(cc->ricci);
    }

    // Advance the Mertens schedule past this position
    if (cc->mertens) {
        sp_mertens_advance(cc->mertens, pos);
    }
}

void sp_cauchy_print_stats(const sp_cauchy_ctrl_t *cc) {
    const char *mode_str = "off";
    if (cc->mode == 1) mode_str = "fixed-N";
    if (cc->mode == 2) mode_str = "dynamic (Ricci+Mertens)";

    fprintf(stderr,
        "[sp-cauchy] mode=%s  resets=%d  last_reset_pos=%d",
        mode_str, cc->total_resets, cc->last_reset_pos);

    if (cc->mode == 1) {
        fprintf(stderr, "  fixed_n=%d", cc->fixed_n);
    }
    if (cc->ricci && cc->ricci->calibrated) {
        fprintf(stderr, "  ricci_drift=%.4f  threshold=%.4f",
                sp_ricci_drift(cc->ricci), cc->ricci->metric_criticality);
    }
    if (cc->mertens) {
        fprintf(stderr, "  mertens_schedule=%d/%d",
                cc->mertens->schedule_idx, cc->mertens->n_schedule);
    }
    fprintf(stderr, "\n");
}
