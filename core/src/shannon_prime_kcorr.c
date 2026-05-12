// Shannon-Prime K-corr Spectral Acceptance
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Phase 2 SP-Flash: K-corr acceptance criterion. Accepts a draft token
// whose hidden state is spectrally close to the target's, even when token
// IDs differ. Enables acceptance of semantically equivalent paths (paraphrases)
// that exact-match would reject.

#include "shannon_prime.h"

#include <math.h>
#include <string.h>

// ---------------------------------------------------------------------------
// sp_kcorr
// ---------------------------------------------------------------------------

float sp_kcorr(const float *a, const float *b, int n, float *scratch)
{
    // scratch layout: [0..n-1] = VHT2(a), [n..2n-1] = VHT2(b)
    float *va = scratch;
    float *vb = scratch + n;

    memcpy(va, a, (size_t)n * sizeof(float));
    memcpy(vb, b, (size_t)n * sizeof(float));

    sp_vht2_forward_f32(va, n);
    sp_vht2_forward_f32(vb, n);

    double dot = 0.0, na2 = 0.0, nb2 = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)va[i] * (double)vb[i];
        na2 += (double)va[i] * (double)va[i];
        nb2 += (double)vb[i] * (double)vb[i];
    }

    const double denom = sqrt(na2) * sqrt(nb2);
    if (denom < 1e-12) return 0.0f;
    return (float)(dot / denom);
}

// ---------------------------------------------------------------------------
// sp_kcorr_floor
// ---------------------------------------------------------------------------

float sp_kcorr_floor(float params_b, int bits, float ppl_budget)
{
    // Scaling law: PPL degradation ≈ K * (1 − floor)^2 / (params^β × bits^γ)
    // Inverted:    floor = 1 − sqrt(log(1+budget) × params^β × bits^γ / K)
    //
    // Empirically fit to SP-Flash calibration data:
    //   K = 4700, β = 1.1, γ = 1.5
    //
    // Example: Qwen3.6-35B Q4 (params_b=35, bits=4, ppl_budget=0.03)
    //   rhs = log(1.03) × 35^1.1 × 4^1.5 / 4700 ≈ 0.0296 × 42.7 × 8 / 4700 ≈ 0.00216
    //   floor = 1 − sqrt(0.00216) ≈ 1 − 0.0465 ≈ 0.953
    static const float K    = 4700.0f;
    static const float beta = 1.1f;
    static const float gamma_exp = 1.5f;

    if (params_b <= 0.0f || bits <= 0 || ppl_budget <= 0.0f) return 1.0f;

    const float rhs = logf(1.0f + ppl_budget)
                    * powf(params_b, beta)
                    * powf((float)bits, gamma_exp)
                    / K;

    if (rhs <= 0.0f) return 1.0f;
    const float floor_val = 1.0f - sqrtf(rhs);
    // Clamp to [0, 1) — a floor ≥ 1.0 would never accept any draft token.
    if (floor_val < 0.0f) return 0.0f;
    if (floor_val >= 1.0f) return 0.999f;
    return floor_val;
}

// ---------------------------------------------------------------------------
// sp_kcorr_accept
// ---------------------------------------------------------------------------

bool sp_kcorr_accept(const float *draft_h, const float *target_h, int n,
                     float floor, float *scratch)
{
    return sp_kcorr(draft_h, target_h, n, scratch) >= floor;
}
