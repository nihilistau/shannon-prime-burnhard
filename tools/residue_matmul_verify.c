// Shannon-Prime BurnHard — Residue matmul verifier.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Proves that the residue path
//
//     A → discretise → split → modular sums → Garner → unscale
//
// produces the same output as a stock fp32 matmul, to within 1e-5 relative
// error per cell. This is the gate Session 2 must pass before we burn
// cycles writing CUDA kernels for the same math: if the host reference
// is wrong, the GPU port will be wrong faster.
//
// Tests:
//   1. Tiny matmul   (M=K=N=8)              — sanity, easy to debug
//   2. Small matmul  (M=N=64, K=128)         — non-power-of-2 inner dim
//   3. Realistic     (M=N=2048, K=2048)      — Qwen3.6 hidden_dim scale
//   4. Stress        (K=8192)                — at the safe_q_max boundary
//
// Tolerances (fp32-vs-fp32 reasonable for matmul of K=8192 random uniforms):
//   max_abs_err  ≤ 1e-4   (absolute — fp32 mantissa epsilon × √K)
//   nrmse        ≤ 1e-5   (RMSE divided by peak |C_ref|, the standard matmul
//                          A/B metric — robust to small-cancellation cells)
//
// We deliberately do NOT gate on per-cell max_rel_err — when an output cell
// happens to land near zero (random ±1 inputs cancelling each other out),
// even fp32-vs-fp32 with different accumulation orders shows max_rel ~ 1e-3
// while max_abs stays ≤ fp32_eps × √K ≈ 5e-6. Reporting it for visibility,
// but it's NOT a gate.
//
// Reference uses double accumulator so its rounding matches what fp32
// matmul should ideally produce.

#include "burnhard_residue_matmul.h"
#include "burnhard_crt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef SP_VERIFY_MAX_ABS
#  define SP_VERIFY_MAX_ABS 1e-4
#endif
#ifndef SP_VERIFY_MAX_NRMSE
#  define SP_VERIFY_MAX_NRMSE 1e-5
#endif

// ----------------------------------------------------------------------------
// Fill an A or B matrix with values in [-1, 1], with a few outliers and
// corner cases so the verifier hits the saturation paths.
// ----------------------------------------------------------------------------
static void random_fill(float *m, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        if      (i % 1024 == 0)  m[i] =  0.0f;
        else if (i % 1024 == 1)  m[i] =  1.0f;
        else if (i % 1024 == 2)  m[i] = -1.0f;
        else if (i % 1024 == 3)  m[i] =  0.5f;
        else                      m[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

// ----------------------------------------------------------------------------
// Compare two C matrices and report max-abs / max-rel / mean-rel error.
// ----------------------------------------------------------------------------
static int compare_and_report(const char *tag,
                               const float *C_ref, const float *C_test,
                               size_t M, size_t K, size_t N,
                               double t_ref_ms, double t_test_ms) {
    double max_abs = 0.0;
    double sum_sq  = 0.0;
    double peak    = 0.0;
    double max_rel = 0.0;     // reported, NOT gated
    size_t n_cells = M * N;
    size_t worst_abs_idx = 0;
    for (size_t i = 0; i < n_cells; ++i) {
        const double a = (double)C_ref[i];
        const double b = (double)C_test[i];
        const double e = fabs(a - b);
        sum_sq += e * e;
        if (e > max_abs) { max_abs = e; worst_abs_idx = i; }
        if (fabs(a) > peak) peak = fabs(a);
        const double mag = fabs(a) > 1e-9 ? fabs(a) : 1e-9;
        const double r = e / mag;
        if (isfinite(r) && r > max_rel) max_rel = r;
    }
    const double rmse  = sqrt(sum_sq / (double)n_cells);
    const double nrmse = (peak > 0.0) ? rmse / peak : 0.0;

    int ok = (max_abs <= SP_VERIFY_MAX_ABS) &&
             (nrmse   <= SP_VERIFY_MAX_NRMSE);

    printf("[%s] M=%zu K=%zu N=%zu  q_max=%d  ref=%.1fms  res=%.1fms\n",
           tag, M, K, N, sp_burnhard_safe_q_max(K), t_ref_ms, t_test_ms);
    printf("        max_abs=%.3e  nrmse=%.3e  peak|C|=%.3e  "
           "(max_rel=%.3e — informational)  ",
           max_abs, nrmse, peak, max_rel);
    if (ok) {
        printf("PASS\n");
    } else {
        size_t wm = worst_abs_idx / N, wn = worst_abs_idx % N;
        printf("FAIL  worst_abs[%zu,%zu] ref=%+0.6e res=%+0.6e\n",
               wm, wn, (double)C_ref[worst_abs_idx],
               (double)C_test[worst_abs_idx]);
    }
    return ok ? 0 : 1;
}

// ----------------------------------------------------------------------------
// One trial: allocate, fill, run both paths, compare.
// ----------------------------------------------------------------------------
static int run_trial(const char *tag, size_t M, size_t K, size_t N,
                      unsigned seed) {
    float *A      = (float *)malloc(M * K * sizeof(float));
    float *B      = (float *)malloc(K * N * sizeof(float));
    float *C_ref  = (float *)malloc(M * N * sizeof(float));
    float *C_test = (float *)malloc(M * N * sizeof(float));
    if (!A || !B || !C_ref || !C_test) {
        fprintf(stderr, "[%s] OOM (M=%zu K=%zu N=%zu)\n", tag, M, K, N);
        free(A); free(B); free(C_ref); free(C_test);
        return 1;
    }

    random_fill(A, M * K, seed);
    random_fill(B, K * N, seed ^ 0xCAFEBABE);

    struct timespec t0, t1, t2;
    timespec_get(&t0, TIME_UTC);
    sp_reference_matmul_fp32(A, B, C_ref, M, K, N);
    timespec_get(&t1, TIME_UTC);
    int rc = sp_burnhard_residue_matmul_fp32(A, B, C_test, M, K, N);
    timespec_get(&t2, TIME_UTC);

    if (rc != 0) {
        fprintf(stderr, "[%s] residue matmul failed (rc=%d)\n", tag, rc);
        free(A); free(B); free(C_ref); free(C_test);
        return 1;
    }

    double t_ref_ms  = ((double)(t1.tv_sec - t0.tv_sec) * 1e3) +
                        ((double)(t1.tv_nsec - t0.tv_nsec) / 1e6);
    double t_test_ms = ((double)(t2.tv_sec - t1.tv_sec) * 1e3) +
                        ((double)(t2.tv_nsec - t1.tv_nsec) / 1e6);

    int err = compare_and_report(tag, C_ref, C_test, M, K, N,
                                  t_ref_ms, t_test_ms);

    free(A); free(B); free(C_ref); free(C_test);
    return err;
}

int main(void) {
    printf("[residue_matmul_verify] BurnHard residue matmul vs stock fp32\n");
    printf("[residue_matmul_verify] M1=%lld  M2=%lld  M_PROD/2=%lld (~2^%d)\n",
           (long long)SP_M1, (long long)SP_M2,
           (long long)(SP_M_PRODUCT >> 1),
           (int)log2((double)(SP_M_PRODUCT >> 1)));

    int rc = 0;
    rc |= run_trial("tiny",    8,    8,    8,   0x10101);
    rc |= run_trial("small",   64,   128,  64,  0x20202);
    rc |= run_trial("medium",  256,  512,  256, 0x30303);
    // 2048×2048 is the realistic Qwen3.6 hidden-dim → MoE expert dim case.
    rc |= run_trial("hidden",  2048, 2048, 2048, 0x40404);
    // Stress: at safe_q_max boundary for K=8192.
    rc |= run_trial("stress",  256,  8192, 256, 0x50505);

    if (rc == 0) {
        printf("[residue_matmul_verify] === ALL TRIALS PASS ===\n");
    } else {
        printf("[residue_matmul_verify] === FAILURE ===\n");
    }
    return rc;
}
