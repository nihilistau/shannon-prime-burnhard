// Shannon-Prime BurnHard — Residue matmul (host reference impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// See burnhard_residue_matmul.h for the algorithm + invariants.
//
// IMPLEMENTATION NOTES
//
// Why a single int64 accumulator (not modular reduction inside the K-loop):
//   The safe_q_max formula guarantees K · q_max² ≤ M_PRODUCT/2 ≤ 2^61, so
//   the int64 accumulator can hold the entire un-reduced K-sum without
//   overflow. Modular reduction inside the inner loop would buy us nothing
//   except a 30% perf hit from the extra divide. For the host reference
//   we keep it lean; the CUDA kernel can switch to mod-during-accum if
//   it ever needs to support inner_dim > 32k.
//
// Why TWO accumulators per cell (one mod M1, one mod M2):
//   Garner needs both residues of the same true sum. We could compute the
//   full int64 sum once and mod twice, but in the dispatched (Session 3)
//   case the M1 and M2 work happens on different devices — different
//   discretisation scales would re-introduce overflow. By computing both
//   with identical int32 inputs (A_q, B_q) and identical accumulators, the
//   Garner join is exact. This file mirrors that structure even on host
//   so the dispatcher can swap chambers without rewiring the math.

#include "burnhard_residue_matmul.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// ----------------------------------------------------------------------------
// Stock fp32 reference matmul. Naïve triple loop — exists so the verifier
// has a known-correct baseline without a BLAS dependency.
// ----------------------------------------------------------------------------
void sp_reference_matmul_fp32(const float *A,
                               const float *B,
                               float *C,
                               size_t M, size_t K, size_t N) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            double acc = 0.0;   // double for the reference; reduces rounding
            for (size_t k = 0; k < K; ++k) {
                acc += (double)A[m * K + k] * (double)B[k * N + n];
            }
            C[m * N + n] = (float)acc;
        }
    }
}

// ----------------------------------------------------------------------------
// Internal: discretise an fp32 buffer to int32 using the given scale.
// Round-to-nearest-even, saturating clamp to [-q_clamp, +q_clamp].
// ----------------------------------------------------------------------------
static void discretise_fp32(const float *in,
                             int32_t *out,
                             size_t n,
                             float scale,
                             int32_t q_clamp) {
    const float clamp_f = (float)q_clamp;
    for (size_t i = 0; i < n; ++i) {
        float v = in[i] * scale;
        if (v >  clamp_f) v =  clamp_f;
        if (v < -clamp_f) v = -clamp_f;
        out[i] = (int32_t)lrintf(v);
    }
}

// ----------------------------------------------------------------------------
// Public: residue matmul (host reference).
// ----------------------------------------------------------------------------
int sp_burnhard_residue_matmul_fp32(const float *A,
                                     const float *B,
                                     float *C,
                                     size_t M, size_t K, size_t N) {
    if (!A || !B || !C || M == 0 || K == 0 || N == 0) return -EINVAL;

    // Pick scale based on K so the int64 accumulator stays inside Garner's
    // window. q_max² · K ≤ M_PRODUCT/2 by construction.
    const int32_t q_max = sp_burnhard_safe_q_max(K);
    const float   scale = (float)q_max;
    const float   inv_scale_sq = 1.0f / (scale * scale);

    // Allocate scratch for the int32 representations of A and B. Both go to
    // both chambers — the chambers see identical int32 inputs, only the
    // mod-reduction differs. Memory cost: (M+N)·K · 4 bytes. For M=K=N=2048
    // that's 64 MiB scratch — comfortable on host, fits L3 on the i9.
    int32_t *A_q = (int32_t *)malloc(M * K * sizeof(int32_t));
    int32_t *B_q = (int32_t *)malloc(K * N * sizeof(int32_t));
    if (!A_q || !B_q) { free(A_q); free(B_q); return -ENOMEM; }

    discretise_fp32(A, A_q, M * K, scale, q_max);
    discretise_fp32(B, B_q, K * N, scale, q_max);

    // Triple loop. For each output cell, accumulate the full K-sum in two
    // parallel int64s — one will be mod-reduced by M1, the other by M2 —
    // then Garner-join.
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            int64_t acc = 0;
            for (size_t k = 0; k < K; ++k) {
                acc += (int64_t)A_q[m * K + k] * (int64_t)B_q[k * N + n];
            }
            // Reduce to the two residues. acc is the SAME value for both
            // chambers — the chambers only differ in the modular reduction.
            int64_t r1 = acc % SP_M1; if (r1 < 0) r1 += SP_M1;
            int64_t r2 = acc % SP_M2; if (r2 < 0) r2 += SP_M2;
            // Garner: recovers the original int64 within Garner's window.
            // safe_q_max guarantees |acc| ≤ M_PRODUCT/2, so the recovered
            // int64 equals acc exactly.
            int64_t recovered =
                sp_burnhard_garner_one((int32_t)r1, (int32_t)r2);
            // Unscale: divide by S² to undo both discretisations.
            C[m * N + n] = (float)((double)recovered * (double)inv_scale_sq);
        }
    }

    free(A_q);
    free(B_q);
    return 0;
}
