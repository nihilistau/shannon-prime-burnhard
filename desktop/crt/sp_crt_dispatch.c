// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "sp_crt_dispatch.h"
#ifdef SP_ENGINE_WITH_BEAST
#include "../beast_canyon/sp_hetero_sync.h"
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_us(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1e6;
}
#else
#include <time.h>
static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
#endif

// ============================================================================
// Init / Free
// ============================================================================

int sp_crt_dispatch_init(sp_crt_dispatch_t* d, int max_dim) {
    memset(d, 0, sizeof(*d));

    d->max_M = max_dim;
    d->max_N = max_dim;
    d->max_K = max_dim;

    // Init the underlying CRT context with the given max dimensions.
    int rc = sp_crt_init(&d->crt, max_dim, max_dim, max_dim, NULL, NULL);
    if (rc != 0) return rc;

    // Try heterogeneous GPU acceleration (CUDA + Vulkan) first.
    // Falls back to CUDA-only, then CPU if both fail. Non-fatal.
#ifdef SP_ENGINE_WITH_BEAST
    {
        sp_hetero_barrier_t barrier;
        memset(&barrier, 0, sizeof(barrier));
        int det = sp_hetero_detect_gpus(&barrier);
        if (det == 0 && barrier.n_gpus >= 2) {
            int hrc = sp_crt_init_gpu_hetero(&d->crt, &barrier);
            if (hrc != 0) {
                fprintf(stderr, "[Shannon-Prime CRT] Hetero init failed (rc=%d), trying CUDA-only\n", hrc);
                sp_crt_init_gpu(&d->crt);
            }
        } else {
            sp_crt_init_gpu(&d->crt);
        }
    }
#else
    sp_crt_init_gpu(&d->crt);
#endif

    // Transpose scratch: max_M * max_K floats for the ggml path.
    d->transpose_scratch = (float*)malloc((size_t)max_dim * max_dim * sizeof(float));
    if (!d->transpose_scratch) {
        sp_crt_free(&d->crt);
        return -1;
    }

    return 0;
}

void sp_crt_dispatch_free(sp_crt_dispatch_t* d) {
    if (!d) return;
    sp_crt_free(&d->crt);
    free(d->transpose_scratch);
    d->transpose_scratch = NULL;
}

void sp_crt_dispatch_stats(const sp_crt_dispatch_t* d) {
    if (!d) return;
    printf("CRT dispatch stats:\n");
    printf("  dispatched:   %d matmuls\n", d->n_dispatched);
    printf("  fallthrough:  %d matmuls\n", d->n_fallthrough);
    if (d->n_dispatched > 0) {
        printf("  total time:   %.1f ms\n", d->total_us / 1e3);
        printf("  avg per call: %.1f us\n", d->total_us / d->n_dispatched);
    }
    printf("  GPU accel:    %s\n", d->crt.gpu_ready ? "YES" : "no (CPU fallback)");
}

// ============================================================================
// Internal: compute range of a float buffer
// ============================================================================

static void compute_range(const float* data, int n, float* out_min, float* out_max) {
    float mn = data[0], mx = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    *out_min = mn;
    *out_max = mx;
}

// ============================================================================
// Internal: transpose [K, M] → [M, K] (column-major → row-major)
// ============================================================================

static void transpose_f32(const float* src, float* dst, int K, int M) {
    // src[k * M + m] → dst[m * K + k]
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            dst[m * K + k] = src[k * M + m];
        }
    }
}

// ============================================================================
// Internal: fp16 → fp32 conversion (portable, no intrinsics needed)
// ============================================================================

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;

    if (exp == 0) {
        if (frac == 0) {
            // Zero
            uint32_t bits = sign;
            float f;
            memcpy(&f, &bits, 4);
            return f;
        }
        // Denormalized
        exp = 1;
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3FF;
        uint32_t bits = sign | ((exp + 112) << 23) | (frac << 13);
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
    if (exp == 31) {
        // Inf or NaN
        uint32_t bits = sign | 0x7F800000 | (frac << 13);
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
    // Normalized
    uint32_t bits = sign | ((exp + 112) << 23) | (frac << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// ============================================================================
// Hook 1: ggml_map_custom2 callback
// ============================================================================
//
// a = weight [K, M] (ggml column-major), b = input [K, N], dst = [M, N]
// This matches ggml_mul_mat(a, b) = a^T × b = [M, K] × [K, N] = [M, N]

// Forward-declare the ggml tensor fields we need without pulling in ggml.h.
// ggml_tensor has ne[4] at offset 0 and data at a known offset.
// We use the struct tag that ggml.h defines.
#include "ggml.h"

void sp_crt_ggml_matmul(struct ggml_tensor* dst,
                         const struct ggml_tensor* a,
                         const struct ggml_tensor* b,
                         int ith, int nth, void* userdata) {
    // Only thread 0 does the CRT dispatch
    if (ith != 0) return;

    sp_crt_dispatch_t* d = (sp_crt_dispatch_t*)userdata;

    // Extract dimensions from ggml tensor shapes.
    // a = [K, M]: a->ne[0] = K, a->ne[1] = M
    // b = [K, N]: b->ne[0] = K, b->ne[1] = N
    // dst = [M, N]
    int K = (int)a->ne[0];
    int M = (int)a->ne[1];
    int N = (int)b->ne[1];

    // Bounds check — fall through if exceeds init dimensions
    if (M > d->max_M || N > d->max_N || K > d->max_K) {
        // Can't do CRT — dimensions too large. Fall through by
        // computing a naive fp32 matmul into dst.
        // (ggml_map_custom2 doesn't have a "fallthrough" mechanism,
        //  so we must compute the result ourselves.)
        d->n_fallthrough++;
        const float* A = (const float*)a->data;
        const float* B = (const float*)b->data;
        float* C = (float*)dst->data;
        // Naive matmul: C[m,n] = sum_k A[k*M + m] * B[k*N + n]
        memset(C, 0, (size_t)M * N * sizeof(float));
        for (int k = 0; k < K; k++) {
            for (int m = 0; m < M; m++) {
                float a_val = A[k * M + m];
                for (int n2 = 0; n2 < N; n2++) {
                    C[m * N + n2] += a_val * B[k * N + n2];
                }
            }
        }
        return;
    }

    double t0 = get_time_us();

    // Transpose A from [K, M] (ggml) to [M, K] (CRT convention)
    const float* A_colmaj = (const float*)a->data;
    float* A_rowmaj = d->transpose_scratch;
    transpose_f32(A_colmaj, A_rowmaj, K, M);

    const float* B_data = (const float*)b->data;
    float* C_data = (float*)dst->data;

    // Calibrate quantization from data ranges.
    // Use K-aware calibration to prevent overflow in the accumulator.
    float a_min, a_max, b_min, b_max;
    compute_range(A_rowmaj, M * K, &a_min, &a_max);
    compute_range(B_data,   K * N, &b_min, &b_max);

    sp_crt_quant_calibrate_k(&d->crt.act_quant,    b_min, b_max, K);
    sp_crt_quant_calibrate_k(&d->crt.weight_quant,  a_min, a_max, K);

    // Dispatch — GPU if available, CPU fallback
    int rc;
    if (d->crt.gpu_ready) {
        rc = sp_crt_matmul_gpu(&d->crt, A_rowmaj, B_data, C_data, M, N, K);
    } else {
        d->crt.M = M;
        d->crt.N = N;
        d->crt.K = K;
        rc = sp_crt_matmul(&d->crt, A_rowmaj, B_data, C_data, M, N, K);
    }

    if (rc != 0) {
        // CRT failed — fallback to naive
        d->n_fallthrough++;
        memset(C_data, 0, (size_t)M * N * sizeof(float));
        for (int k = 0; k < K; k++) {
            for (int m = 0; m < M; m++) {
                float a_val = A_rowmaj[m * K + k];
                for (int n2 = 0; n2 < N; n2++) {
                    C_data[m * N + n2] += a_val * B_data[k * N + n2];
                }
            }
        }
    } else {
        d->n_dispatched++;
    }

    double t1 = get_time_us();
    d->total_us += (t1 - t0);
}

// ============================================================================
// Hook 2: native forward mm_dispatch callback
// ============================================================================
//
// lhs = [M, K] fp32 (activations), W_fp16 = [K, N] fp16 (weights)
// out = [M, N] fp32
//
// The native path already provides the weight in [K, N] row-major and
// the input in [M, K] row-major. For CRT we need A=[M,K] and B=[K,N].
// A = lhs, B = W (after fp16 → fp32 conversion). Direct — no transpose.

int sp_crt_native_mm_dispatch(void* userdata,
                               const float*    lhs,
                               const uint16_t* W_fp16,
                               int M, int K, int N,
                               float*          out) {
    sp_crt_dispatch_t* d = (sp_crt_dispatch_t*)userdata;

    // Bounds check
    if (M > d->max_M || N > d->max_N || K > d->max_K) {
        d->n_fallthrough++;
        return -1;   // non-zero → caller falls back to CPU
    }

    double t0 = get_time_us();

    // Convert fp16 weights to fp32 into transpose_scratch
    // (reuse buffer — native path doesn't need transpose)
    int wn = K * N;
    float* W_fp32 = d->transpose_scratch;
    for (int i = 0; i < wn; i++) {
        W_fp32[i] = fp16_to_fp32(W_fp16[i]);
    }

    // Calibrate
    float a_min, a_max, w_min, w_max;
    compute_range(lhs,    M * K, &a_min, &a_max);
    compute_range(W_fp32, wn,    &w_min, &w_max);

    sp_crt_quant_calibrate_k(&d->crt.act_quant,    a_min, a_max, K);
    sp_crt_quant_calibrate_k(&d->crt.weight_quant,  w_min, w_max, K);

    // Dispatch
    int rc;
    if (d->crt.gpu_ready) {
        rc = sp_crt_matmul_gpu(&d->crt, lhs, W_fp32, out, M, N, K);
    } else {
        d->crt.M = M;
        d->crt.N = N;
        d->crt.K = K;
        rc = sp_crt_matmul(&d->crt, lhs, W_fp32, out, M, N, K);
    }

    if (rc != 0) {
        d->n_fallthrough++;
        return -1;
    }

    d->n_dispatched++;
    double t1 = get_time_us();
    d->total_us += (t1 - t0);

    return 0;  // success — caller uses our result
}
