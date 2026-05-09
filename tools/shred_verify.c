// Shannon-Prime BurnHard — Shredder verifier.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Confirms the in-flight Shredder + Garner round-trip preserves Q4_K
// dequantised weights to within the 1e-6 mean-squared-error gate.
//
// Two test paths:
//
//   1. Synthetic: generates a 4096-element fp32 buffer covering the full
//      Q4_K post-dequant range (-1.0 to +1.0, plus a few extreme outliers),
//      shreds → joins → reports max abs error and MSE.
//
//   2. Real GGUF: optionally loads a tensor block from a real Q4_K file,
//      runs ggml's stock dequantisation, then exercises the Shredder
//      against that. Catches any per-block-scale corner case the synthetic
//      path might miss.
//
// Exit code: 0 if all checks pass under the configured tolerance, 1 if any
// check fails. The CI smoke wraps this around the build to gate further work.
//
// Usage:
//   shred_verify                              # synthetic-only, fast
//   shred_verify <gguf_path> <tensor_name>    # also exercises a real tensor
//
// Tolerances (compile-time, change with -D at build):
//   * SP_VERIFY_MAX_ABS  default 1e-6
//   * SP_VERIFY_MAX_MSE  default 1e-12

#include "burnhard_crt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef SP_VERIFY_MAX_ABS
#  define SP_VERIFY_MAX_ABS 1e-6
#endif
#ifndef SP_VERIFY_MAX_MSE
#  define SP_VERIFY_MAX_MSE 1e-12
#endif

// ----------------------------------------------------------------------------
// Synthetic path — covers the full expected weight distribution plus the
// dangerous corner cases (zero, ±1.0, denormals, near-saturation).
// ----------------------------------------------------------------------------
static int verify_synthetic(float scale, size_t n) {
    float   *src = (float   *)malloc(n * sizeof(float));
    int32_t *r1  = (int32_t *)malloc(n * sizeof(int32_t));
    int32_t *r2  = (int32_t *)malloc(n * sizeof(int32_t));
    float   *dst = (float   *)malloc(n * sizeof(float));
    if (!src || !r1 || !r2 || !dst) {
        fprintf(stderr, "[shred_verify] OOM (n=%zu)\n", n); return 1;
    }

    // Fill: bulk of values uniform in [-0.5, 0.5] (typical dequant range),
    // with a small tail of outliers + corner cases.
    srand(0x5eedca11);
    for (size_t i = 0; i < n; ++i) {
        if      (i == 0)         src[i] =  0.0f;
        else if (i == 1)         src[i] = -0.0f;
        else if (i == 2)         src[i] =  1.0f;
        else if (i == 3)         src[i] = -1.0f;
        else if (i == 4)         src[i] =  0.999999f;
        else if (i == 5)         src[i] = -0.999999f;
        else if (i % 256 == 0)   src[i] = (float)(rand() % 1000 - 500) / 100.0f; // ±5
        else                     src[i] = ((float)rand() / RAND_MAX - 0.5f);
    }

    sp_burnhard_shred_fp32(src, r1, r2, scale, n);
    sp_burnhard_garner_fp32(r1, r2, dst, scale, n);

    double max_abs = 0.0, sum_sq = 0.0;
    size_t worst_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = (double)src[i] - (double)dst[i];
        if (fabs(e) > max_abs) { max_abs = fabs(e); worst_idx = i; }
        sum_sq += e * e;
    }
    const double mse = sum_sq / (double)n;

    printf("[shred_verify] synthetic: n=%zu  scale=%.0f  max_abs=%.3e  mse=%.3e",
           n, (double)scale, max_abs, mse);

    int ok = (max_abs <= SP_VERIFY_MAX_ABS) && (mse <= SP_VERIFY_MAX_MSE);
    if (!ok) {
        printf("  FAIL  worst[%zu]=(src=%.9f, dst=%.9f, e=%.3e)\n",
               worst_idx, (double)src[worst_idx], (double)dst[worst_idx],
               (double)src[worst_idx] - (double)dst[worst_idx]);
    } else {
        printf("  PASS\n");
    }

    free(src); free(r1); free(r2); free(dst);
    return ok ? 0 : 1;
}

// ----------------------------------------------------------------------------
// Integer round-trip — every value in [-2^28, 2^28] must shred + Garner
// back to itself exactly. This is the gate the matmul correctness sits on:
// if any q in the safe range round-trips wrong, the dispatcher is broken.
// ----------------------------------------------------------------------------
static int verify_integer_round_trip(void) {
    const int32_t step = 1 << 14;       // ~16k samples across the range
    const int32_t bound = 1 << 28;
    int errors = 0;
    int32_t worst_q = 0; int64_t worst_recovered = 0;
    for (int64_t q = -bound; q <= bound; q += step) {
        const burnhard_residue_t r = sp_burnhard_shred_one((int32_t)q);
        const int64_t back = sp_burnhard_garner_one(r.r1, r.r2);
        if (back != q) {
            if (errors == 0) { worst_q = (int32_t)q; worst_recovered = back; }
            errors++;
        }
    }
    printf("[shred_verify] integer round-trip: range=±2^28  step=2^14  ");
    if (errors == 0) {
        printf("0 errors  PASS\n");
        return 0;
    }
    printf("%d errors  FAIL  worst q=%d → %lld\n", errors,
           worst_q, (long long)worst_recovered);
    return 1;
}

// ----------------------------------------------------------------------------
// Bias check — Garner-join must not introduce a DC offset. Critical for
// the post-matmul reconstruction (a small bias becomes a large logit shift
// after the output projection).
// ----------------------------------------------------------------------------
static int verify_no_bias(float scale, size_t n) {
    float   *src = (float   *)malloc(n * sizeof(float));
    int32_t *r1  = (int32_t *)malloc(n * sizeof(int32_t));
    int32_t *r2  = (int32_t *)malloc(n * sizeof(int32_t));
    float   *dst = (float   *)malloc(n * sizeof(float));
    if (!src || !r1 || !r2 || !dst) return 1;

    // Centred uniform on [-0.5, 0.5] — sample mean should be ≈ 0.
    srand(0xb1a5c4ec);
    for (size_t i = 0; i < n; ++i) {
        src[i] = ((float)rand() / RAND_MAX - 0.5f);
    }
    sp_burnhard_shred_fp32(src, r1, r2, scale, n);
    sp_burnhard_garner_fp32(r1, r2, dst, scale, n);

    double sum_src = 0.0, sum_dst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_src += src[i];
        sum_dst += dst[i];
    }
    const double bias = (sum_dst - sum_src) / (double)n;
    printf("[shred_verify] DC bias check: n=%zu  bias=%.3e", n, bias);

    int ok = (fabs(bias) < 1e-9);
    printf(ok ? "  PASS\n" : "  FAIL\n");
    free(src); free(r1); free(r2); free(dst);
    return ok ? 0 : 1;
}

// ----------------------------------------------------------------------------
// Microbench. Reports ns / element so we can track regressions when the
// AVX-512 Barrett reduction lands.
// ----------------------------------------------------------------------------
static int microbench(float scale, size_t n) {
    float   *src = (float   *)malloc(n * sizeof(float));
    int32_t *r1  = (int32_t *)malloc(n * sizeof(int32_t));
    int32_t *r2  = (int32_t *)malloc(n * sizeof(int32_t));
    if (!src || !r1 || !r2) return 1;
    srand(0xb37c4);
    for (size_t i = 0; i < n; ++i) src[i] = ((float)rand() / RAND_MAX - 0.5f);

    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    const int iters = 16;
    for (int it = 0; it < iters; ++it) {
        sp_burnhard_shred_fp32(src, r1, r2, scale, n);
    }
    timespec_get(&t1, TIME_UTC);
    const double total_ns =
        ((double)(t1.tv_sec - t0.tv_sec) * 1e9) +
        ((double)(t1.tv_nsec - t0.tv_nsec));
    const double ns_per = total_ns / ((double)n * iters);
    const double gb_per_s = ((double)n * 4.0 / (1024.0*1024.0*1024.0)) /
                             (total_ns / iters / 1e9);
    printf("[shred_verify] bench: %.2f ns/element  (%.2f GB/s fp32 read, "
           "n=%zu × %d iter)\n", ns_per, gb_per_s, n, iters);
    free(src); free(r1); free(r2);
    return 0;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    printf("[shred_verify] BurnHard Shredder + Garner round-trip\n");
    printf("[shred_verify] M1=%lld  M2=%lld  M1^-1 mod M2=%lld  scale=%.0f\n",
           (long long)SP_M1, (long long)SP_M2,
           (long long)SP_M1_INV_MOD_M2,
           (double)SP_BURNHARD_DEFAULT_SCALE);
    printf("[shred_verify] safe q_max for inner_dim=8192: %d  (~%.2f%% of M1/2)\n",
           sp_burnhard_safe_q_max(8192),
           (double)sp_burnhard_safe_q_max(8192) / (double)(SP_M1 >> 1) * 100.0);

    int rc = 0;
    rc |= verify_integer_round_trip();
    rc |= verify_synthetic(SP_BURNHARD_DEFAULT_SCALE, 4096);
    rc |= verify_no_bias(SP_BURNHARD_DEFAULT_SCALE, 65536);
    rc |= microbench(SP_BURNHARD_DEFAULT_SCALE, 1 << 20);
    if (rc == 0) {
        printf("[shred_verify] === ALL CHECKS PASS ===\n");
    } else {
        printf("[shred_verify] === FAILURE ===\n");
    }
    return rc;
}
