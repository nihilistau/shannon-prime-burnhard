// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// CRT Multi-GPU Tensor Splitting — Host-Side Implementation
//
// This file contains the host-side Garner reconstruction, quantization
// calibration, and the dispatch coordinator. GPU-specific kernels are
// in sp_crt_cuda.cu and sp_crt_vulkan.c.
//
// QUANTIZATION DESIGN (v2 — zero-centered symmetric):
//
// We use zero-centered symmetric quantization: 0.0 maps to 0 in the ring,
// positive values map to small positive integers, negative values map to
// M - |q| (the modular representation of negative numbers).
//
// This eliminates the zero_point entirely, which means:
//   - No cross-term correction in the matmul dequantization
//   - No catastrophic cancellation from subtracting large similar values
//   - Signed Garner reconstruction: values > M1*M2/2 are negative
//
// The K-aware ceiling ensures accumulation fits the CRT combined range:
//   K × Q_max² < M1×M2/2   (factor 2 for signed range)
//   Q_max < sqrt(M1×M2 / (2K))
//
// For K=64:  Q_max ≈ 1.90e8 (28-bit Int28 — well above fp16 mantissa)
// For K=4096: Q_max ≈ 2.37e7 (25-bit — still excellent precision)

#include "sp_crt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sp_crt_vulkan.h"

// Forward-declare the hetero barrier type for sp_crt_init_gpu_hetero
#include "../beast_canyon/sp_hetero_sync.h"

// ============================================================================
// Quantization calibration — zero-centered symmetric
// ============================================================================

void sp_crt_quant_calibrate(sp_crt_quant_t *q, float min_val, float max_val) {
    sp_crt_quant_calibrate_k(q, min_val, max_val, 1);
}

void sp_crt_quant_calibrate_k(sp_crt_quant_t *q, float min_val, float max_val, int K) {
    // Zero-centered symmetric: scale maps max_abs to Q_max.
    // Negative floats become M - |q| in the ring (modular negative).
    //
    // K-aware ceiling: K × Q_max² < M1×M2 / 2 (signed range)
    //   Q_max < sqrt(M1×M2 / (2K))

    double max_abs = fabs(min_val);
    if (fabs(max_val) > max_abs) max_abs = fabs(max_val);
    if (max_abs < 1e-8) max_abs = 1e-8;
    if (K < 1) K = 1;

    // CRT signed range: half the combined modulus (positive half)
    double crt_range = (double)SP_CRT_M1 * (double)SP_CRT_M2;
    double q_max = sqrt(crt_range / (2.0 * (double)K)) * 0.9;  // 90% safety margin

    // Cap at (M1-2)/2 — no point exceeding half a single ring
    double ring_cap = (double)(SP_CRT_M1 - 2) / 2.0;
    if (q_max > ring_cap) q_max = ring_cap;

    q->scale = q_max / max_abs;
    q->inv_scale = max_abs / q_max;
    q->zero_point = 0;  // zero-centered: no offset
}

// ============================================================================
// Zero-centered quantization: float → ring element
// ============================================================================
//
// Positive x → round(x * scale)              [small positive integer]
// Negative x → M - round(|x| * scale)        [modular negative]
// Zero     → 0

static uint32_t sp_crt_quantize_symmetric(float val, double scale, uint64_t modulus) {
    double qd = val * scale;
    // Round to nearest integer (away from zero for .5)
    int64_t q = (qd >= 0) ? (int64_t)(qd + 0.5) : -(int64_t)(-qd + 0.5);

    if (q >= 0) {
        return (uint32_t)((uint64_t)q % modulus);
    } else {
        // Modular negative: M - |q|
        uint64_t pos = (uint64_t)(-q);
        pos %= modulus;
        return (pos == 0) ? 0 : (uint32_t)(modulus - pos);
    }
}

static void sp_crt_cpu_quantize_symmetric(const float *input, uint32_t *output,
                                           int n, double scale, uint64_t modulus) {
    for (int i = 0; i < n; i++) {
        output[i] = sp_crt_quantize_symmetric(input[i], scale, modulus);
    }
}

// ============================================================================
// Signed Garner reconstruction
// ============================================================================
//
// Standard Garner gives X in [0, M1*M2). For signed interpretation:
//   if X > M1*M2/2, the true value is X - M1*M2 (negative).
//
// We return a double to avoid int64 overflow concerns.

static double sp_crt_garner_signed(uint32_t a1, uint32_t a2) {
    const uint64_t m1 = SP_CRT_M1;
    const uint64_t m2 = SP_CRT_M2;
    const uint64_t gc = SP_CRT_GARNER_C;

    uint64_t a1_m2 = (uint64_t)a1 % m2;
    uint64_t diff = (a2 >= (uint32_t)a1_m2)
                  ? ((uint64_t)a2 - a1_m2)
                  : (m2 - a1_m2 + (uint64_t)a2);
    uint64_t h = (diff * gc) % m2;
    // X = a1 + h * M1.  h < M2 < 2^31, M1 < 2^31, so h*M1 < 2^62 — fits uint64.
    uint64_t raw = (uint64_t)a1 + h * m1;

    // Signed interpretation: if raw > M1*M2/2, it represents a negative value.
    double half_range = (double)m1 * (double)m2 / 2.0;
    double full_range = (double)m1 * (double)m2;
    double val = (double)raw;
    if (val > half_range) {
        val -= full_range;
    }
    return val;
}

// ============================================================================
// Host-side Garner batch reconstruction (signed, zero-centered)
// ============================================================================

void sp_crt_garner_batch(const uint32_t *residue_0,
                         const uint32_t *residue_1,
                         float *output,
                         size_t n,
                         const sp_crt_quant_t *quant) {
    const double inv_scale = quant->inv_scale;
    // For zero-centered quant, output = signed_garner(r0, r1) * inv_scale
    // (no zero_point subtraction needed)
    for (size_t i = 0; i < n; i++) {
        double signed_val = sp_crt_garner_signed(residue_0[i], residue_1[i]);
        output[i] = (float)(signed_val * inv_scale);
    }
}

// ============================================================================
// CRT context lifecycle
// ============================================================================

int sp_crt_init(sp_crt_context_t *ctx, int max_M, int max_N, int max_K,
                void *stream_0, void *stream_1) {
    memset(ctx, 0, sizeof(*ctx));

    size_t max_output = (size_t)max_M * max_N;

    ctx->h_residue_0 = (uint32_t *)calloc(max_output, sizeof(uint32_t));
    ctx->h_residue_1 = (uint32_t *)calloc(max_output, sizeof(uint32_t));
    ctx->h_output    = (float *)calloc(max_output, sizeof(float));

    if (!ctx->h_residue_0 || !ctx->h_residue_1 || !ctx->h_output) {
        free(ctx->h_residue_0);
        free(ctx->h_residue_1);
        free(ctx->h_output);
        memset(ctx, 0, sizeof(*ctx));
        return -1;
    }

    ctx->M = max_M;
    ctx->N = max_N;
    ctx->K = max_K;
    ctx->stream_0 = stream_0;
    ctx->stream_1 = stream_1;

    // Default calibration (caller should re-calibrate with observed ranges)
    sp_crt_quant_calibrate(&ctx->weight_quant, -1.0f, 1.0f);
    sp_crt_quant_calibrate(&ctx->act_quant, -4.0f, 4.0f);

    ctx->initialized = 1;

    if (getenv("SHANNON_PRIME_VERBOSE")) {
        fprintf(stderr, "[Shannon-Prime CRT] initialized: max %d×%d×%d, "
                        "M1=%llu M2=%llu range=%.2e\n",
                max_M, max_K, max_N,
                (unsigned long long)SP_CRT_M1,
                (unsigned long long)SP_CRT_M2,
                (double)SP_CRT_RANGE);
    }

    return 0;
}

void sp_crt_free(sp_crt_context_t *ctx) {
    free(ctx->h_residue_0);
    free(ctx->h_residue_1);
    free(ctx->h_output);
    if (ctx->gpu_ready) {
        // Ring M1 is always CUDA
        sp_crt_cuda_free(ctx->d_a_m1);
        sp_crt_cuda_free(ctx->d_b_m1);
        sp_crt_cuda_free(ctx->d_c_m1);

        // Ring M2: Vulkan or CUDA
        if (ctx->vk_ring2 && ctx->vk_ctx) {
            sp_crt_vulkan_ctx_t *vk = (sp_crt_vulkan_ctx_t *)ctx->vk_ctx;
            // Vulkan buffers are freed inside sp_crt_vulkan_free
            sp_crt_vulkan_free(vk);
            ctx->vk_ctx = NULL;
        } else {
            sp_crt_cuda_free(ctx->d_a_m2);
            sp_crt_cuda_free(ctx->d_b_m2);
            sp_crt_cuda_free(ctx->d_c_m2);
        }
    }
    memset(ctx, 0, sizeof(*ctx));
}

// ============================================================================
// CPU reference: modular matmul
// ============================================================================

static void sp_crt_cpu_matmul_mod(const uint32_t *A, const uint32_t *B,
                                   uint32_t *C,
                                   int M, int N, int K,
                                   uint64_t modulus) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            uint64_t acc = 0;
            for (int k = 0; k < K; k++) {
                uint64_t a = (uint64_t)A[i * K + k];
                uint64_t b = (uint64_t)B[k * N + j];
                // With zero-centered quantization, ring values can be near M
                // (modular negatives), so a*b can reach ~M^2 ≈ 2^62.
                // Reduce every iteration to keep acc < 2^62 + M < 2^63.
                acc += a * b;
                acc %= modulus;
            }
            C[i * N + j] = (uint32_t)(acc % modulus);
        }
    }
}

static void sp_crt_cpu_matmul_mersenne(const uint32_t *A, const uint32_t *B,
                                        uint32_t *C,
                                        int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            uint64_t acc = 0;
            for (int k = 0; k < K; k++) {
                uint64_t a = (uint64_t)A[i * K + k];
                uint64_t b = (uint64_t)B[k * N + j];
                // Reduce every iteration: after reduce acc < M1 < 2^31,
                // then acc + a*b < 2^31 + M1^2 < 2^31 + 2^62 < 2^63.
                // The improved sp_crt_mersenne_reduce handles up to 2^63.
                acc += a * b;
                acc = sp_crt_mersenne_reduce(acc);
            }
            C[i * N + j] = (uint32_t)acc;  // already reduced
        }
    }
}

// ============================================================================
// Main CRT matmul dispatch — zero-centered symmetric path
// ============================================================================
//
// With zero-centered quantization, there are no cross-terms to correct.
// The matmul in the ring computes ∑ q_a × q_b where q_a = round(a × scale).
// Negative values wrap around M, so the modular product of two negatives
// gives a positive (correct), and a positive × negative gives M - |result|
// (modular negative, also correct).
//
// After signed Garner reconstruction, we simply divide by scale_a × scale_b.

int sp_crt_matmul(sp_crt_context_t *ctx,
                  const float *d_A, const float *d_B, float *d_C,
                  int M, int N, int K) {
    if (!ctx->initialized) return -1;
    if (M * N > ctx->M * ctx->N) return -2;

    size_t a_size = (size_t)M * K;
    size_t b_size = (size_t)K * N;
    size_t c_size = (size_t)M * N;

    // --- Quantize A and B to both residue rings (zero-centered) ---
    uint32_t *a_m1 = (uint32_t *)malloc(a_size * sizeof(uint32_t));
    uint32_t *b_m1 = (uint32_t *)malloc(b_size * sizeof(uint32_t));
    uint32_t *a_m2 = (uint32_t *)malloc(a_size * sizeof(uint32_t));
    uint32_t *b_m2 = (uint32_t *)malloc(b_size * sizeof(uint32_t));
    if (!a_m1 || !b_m1 || !a_m2 || !b_m2) {
        free(a_m1); free(b_m1); free(a_m2); free(b_m2);
        return -3;
    }

    sp_crt_cpu_quantize_symmetric(d_A, a_m1, (int)a_size, ctx->act_quant.scale, SP_CRT_M1);
    sp_crt_cpu_quantize_symmetric(d_B, b_m1, (int)b_size, ctx->weight_quant.scale, SP_CRT_M1);
    sp_crt_cpu_quantize_symmetric(d_A, a_m2, (int)a_size, ctx->act_quant.scale, SP_CRT_M2);
    sp_crt_cpu_quantize_symmetric(d_B, b_m2, (int)b_size, ctx->weight_quant.scale, SP_CRT_M2);

    // --- Matmul in both residue rings ---
    sp_crt_cpu_matmul_mersenne(a_m1, b_m1, ctx->h_residue_0, M, N, K);
    sp_crt_cpu_matmul_mod(a_m2, b_m2, ctx->h_residue_1, M, N, K, SP_CRT_M2);

    free(a_m1); free(b_m1); free(a_m2); free(b_m2);

    // --- Signed Garner reconstruction + dequantize ---
    const double inv_scale = 1.0 / (ctx->act_quant.scale * ctx->weight_quant.scale);

    for (size_t i = 0; i < c_size; i++) {
        double signed_val = sp_crt_garner_signed(ctx->h_residue_0[i],
                                                  ctx->h_residue_1[i]);
        d_C[i] = (float)(signed_val * inv_scale);
    }

    return 0;
}

// ============================================================================
// GPU-accelerated CRT init + matmul
// ============================================================================
//
// sp_crt_init_gpu: allocate device buffers for both residue rings.
// sp_crt_matmul_gpu: quantize on host, H2D, dual-ring matmul on two
//   CUDA streams, D2H, signed Garner on host. When streams are on
//   different GPUs (Beast Canyon: RTX 2060 + Intel UHD), the two ring
//   matmuls run in true parallel with zero inter-GPU communication.

int sp_crt_init_gpu(sp_crt_context_t *ctx) {
    if (!ctx->initialized) return -1;

    size_t max_a = (size_t)ctx->M * ctx->K * sizeof(uint32_t);
    size_t max_b = (size_t)ctx->K * ctx->N * sizeof(uint32_t);
    size_t max_c = (size_t)ctx->M * ctx->N * sizeof(uint32_t);

    // Allocate device memory for both rings
    ctx->d_a_m1 = (uint32_t *)sp_crt_cuda_alloc(max_a);
    ctx->d_b_m1 = (uint32_t *)sp_crt_cuda_alloc(max_b);
    ctx->d_c_m1 = (uint32_t *)sp_crt_cuda_alloc(max_c);
    ctx->d_a_m2 = (uint32_t *)sp_crt_cuda_alloc(max_a);
    ctx->d_b_m2 = (uint32_t *)sp_crt_cuda_alloc(max_b);
    ctx->d_c_m2 = (uint32_t *)sp_crt_cuda_alloc(max_c);

    if (!ctx->d_a_m1 || !ctx->d_b_m1 || !ctx->d_c_m1 ||
        !ctx->d_a_m2 || !ctx->d_b_m2 || !ctx->d_c_m2) {
        sp_crt_cuda_free(ctx->d_a_m1); sp_crt_cuda_free(ctx->d_b_m1);
        sp_crt_cuda_free(ctx->d_c_m1); sp_crt_cuda_free(ctx->d_a_m2);
        sp_crt_cuda_free(ctx->d_b_m2); sp_crt_cuda_free(ctx->d_c_m2);
        ctx->d_a_m1 = ctx->d_b_m1 = ctx->d_c_m1 = NULL;
        ctx->d_a_m2 = ctx->d_b_m2 = ctx->d_c_m2 = NULL;
        return -2;
    }

    // Create streams if caller didn't provide them
    if (!ctx->stream_0) ctx->stream_0 = sp_crt_cuda_stream_create();
    if (!ctx->stream_1) ctx->stream_1 = sp_crt_cuda_stream_create();
    if (!ctx->stream_0 || !ctx->stream_1) return -3;

    ctx->gpu_ready = 1;

    if (getenv("SHANNON_PRIME_VERBOSE")) {
        fprintf(stderr, "[Shannon-Prime CRT] GPU dispatch ready: "
                        "6 device buffers, 2 streams\n");
    }
    return 0;
}

// ============================================================================
// Heterogeneous GPU init: CUDA (Ring M1) + Vulkan (Ring M2)
// ============================================================================

int sp_crt_init_gpu_hetero(sp_crt_context_t *ctx, void *barrier_ptr) {
    sp_hetero_barrier_t *barrier = (sp_hetero_barrier_t *)barrier_ptr;
    if (!ctx->initialized) return -1;
    if (!barrier || barrier->n_gpus < 2) {
        // Fall back to CUDA-only if we don't have 2 GPUs
        fprintf(stderr, "[Shannon-Prime CRT] Hetero: need 2 GPUs, have %d — falling back to CUDA-only\n",
                barrier ? barrier->n_gpus : 0);
        return sp_crt_init_gpu(ctx);
    }

    // Verify GPU[0]=CUDA, GPU[1]=Vulkan
    if (barrier->gpu[0].type != SP_GPU_CUDA ||
        barrier->gpu[1].type != SP_GPU_VULKAN) {
        fprintf(stderr, "[Shannon-Prime CRT] Hetero: expected GPU[0]=CUDA, GPU[1]=Vulkan — falling back\n");
        return sp_crt_init_gpu(ctx);
    }

    size_t max_a = (size_t)ctx->M * ctx->K * sizeof(uint32_t);
    size_t max_b = (size_t)ctx->K * ctx->N * sizeof(uint32_t);
    size_t max_c = (size_t)ctx->M * ctx->N * sizeof(uint32_t);
    int max_elems = ctx->M * ctx->N;  // largest buffer in elements
    if (ctx->M * ctx->K > max_elems) max_elems = ctx->M * ctx->K;
    if (ctx->K * ctx->N > max_elems) max_elems = ctx->K * ctx->N;

    // --- Ring M1: CUDA on RTX 2060 (GPU[0]) ---
    ctx->d_a_m1 = (uint32_t *)sp_crt_cuda_alloc(max_a);
    ctx->d_b_m1 = (uint32_t *)sp_crt_cuda_alloc(max_b);
    ctx->d_c_m1 = (uint32_t *)sp_crt_cuda_alloc(max_c);

    if (!ctx->d_a_m1 || !ctx->d_b_m1 || !ctx->d_c_m1) {
        fprintf(stderr, "[Shannon-Prime CRT] Hetero: CUDA alloc failed — falling back\n");
        sp_crt_cuda_free(ctx->d_a_m1);
        sp_crt_cuda_free(ctx->d_b_m1);
        sp_crt_cuda_free(ctx->d_c_m1);
        ctx->d_a_m1 = ctx->d_b_m1 = ctx->d_c_m1 = NULL;
        return sp_crt_init_gpu(ctx);
    }

    // CUDA stream from barrier
    ctx->stream_0 = barrier->gpu[0].stream;
    if (!ctx->stream_0) ctx->stream_0 = sp_crt_cuda_stream_create();

    // --- Ring M2: Vulkan on Intel UHD (GPU[1]) ---
    sp_crt_vulkan_ctx_t *vk = sp_crt_vulkan_init(
        barrier->gpu[1].context,   // VkInstance from hetero detect
        barrier->gpu[1].device_id, // Physical device index
        max_elems);

    if (!vk) {
        fprintf(stderr, "[Shannon-Prime CRT] Hetero: Vulkan init failed — Ring M2 on CUDA\n");
        // Allocate Ring M2 on CUDA too (same GPU, two streams)
        ctx->d_a_m2 = (uint32_t *)sp_crt_cuda_alloc(max_a);
        ctx->d_b_m2 = (uint32_t *)sp_crt_cuda_alloc(max_b);
        ctx->d_c_m2 = (uint32_t *)sp_crt_cuda_alloc(max_c);
        if (!ctx->d_a_m2 || !ctx->d_b_m2 || !ctx->d_c_m2) {
            sp_crt_cuda_free(ctx->d_a_m2);
            sp_crt_cuda_free(ctx->d_b_m2);
            sp_crt_cuda_free(ctx->d_c_m2);
            ctx->d_a_m2 = ctx->d_b_m2 = ctx->d_c_m2 = NULL;
            return -2;
        }
        ctx->stream_1 = sp_crt_cuda_stream_create();
        ctx->vk_ring2 = 0;
    } else {
        // Vulkan success — allocate Ring M2 buffers on Vulkan device
        ctx->d_a_m2 = sp_crt_vulkan_alloc(vk, max_a);
        ctx->d_b_m2 = sp_crt_vulkan_alloc(vk, max_b);
        ctx->d_c_m2 = sp_crt_vulkan_alloc(vk, max_c);
        if (!ctx->d_a_m2 || !ctx->d_b_m2 || !ctx->d_c_m2) {
            fprintf(stderr, "[Shannon-Prime CRT] Hetero: Vulkan buffer alloc failed\n");
            sp_crt_vulkan_free(vk);
            return -2;
        }
        ctx->stream_1 = sp_crt_vulkan_get_queue(vk);
        ctx->vk_ctx = vk;
        ctx->vk_ring2 = 1;

        // Update barrier GPU[1] with our Vulkan handles for fence polling
        barrier->gpu[1].stream = sp_crt_vulkan_get_queue(vk);
        barrier->gpu[1].event  = sp_crt_vulkan_get_fence(vk);
    }

    ctx->gpu_ready = 1;

    if (getenv("SHANNON_PRIME_VERBOSE")) {
        fprintf(stderr, "[Shannon-Prime CRT] HETEROGENEOUS dispatch ready:\n"
                        "  Ring M1 (Mersenne): CUDA on %s\n"
                        "  Ring M2 (generic):  %s on %s\n",
                barrier->gpu[0].name,
                ctx->vk_ring2 ? "Vulkan" : "CUDA",
                ctx->vk_ring2 ? barrier->gpu[1].name : barrier->gpu[0].name);
    }
    return 0;
}

int sp_crt_matmul_gpu(sp_crt_context_t *ctx,
                      const float *A, const float *B, float *C,
                      int M, int N, int K) {
    // Fall back to CPU if GPU not initialized
    if (!ctx->gpu_ready) return sp_crt_matmul(ctx, A, B, C, M, N, K);
    if (!ctx->initialized) return -1;
    if (M * N > ctx->M * ctx->N) return -2;

    size_t a_size = (size_t)M * K;
    size_t b_size = (size_t)K * N;
    size_t c_size = (size_t)M * N;

    // --- Step 1: Quantize on CPU (zero-centered symmetric) ---
    uint32_t *h_a_m1 = (uint32_t *)malloc(a_size * sizeof(uint32_t));
    uint32_t *h_b_m1 = (uint32_t *)malloc(b_size * sizeof(uint32_t));
    uint32_t *h_a_m2 = (uint32_t *)malloc(a_size * sizeof(uint32_t));
    uint32_t *h_b_m2 = (uint32_t *)malloc(b_size * sizeof(uint32_t));
    if (!h_a_m1 || !h_b_m1 || !h_a_m2 || !h_b_m2) {
        free(h_a_m1); free(h_b_m1); free(h_a_m2); free(h_b_m2);
        return -3;
    }

    sp_crt_cpu_quantize_symmetric(A, h_a_m1, (int)a_size, ctx->act_quant.scale, SP_CRT_M1);
    sp_crt_cpu_quantize_symmetric(B, h_b_m1, (int)b_size, ctx->weight_quant.scale, SP_CRT_M1);
    sp_crt_cpu_quantize_symmetric(A, h_a_m2, (int)a_size, ctx->act_quant.scale, SP_CRT_M2);
    sp_crt_cpu_quantize_symmetric(B, h_b_m2, (int)b_size, ctx->weight_quant.scale, SP_CRT_M2);

    // --- Step 2: H2D transfers ---
    // Ring M1 always goes via CUDA (RTX 2060)
    sp_crt_cuda_memcpy_h2d(ctx->d_a_m1, h_a_m1, a_size * sizeof(uint32_t), ctx->stream_0);
    sp_crt_cuda_memcpy_h2d(ctx->d_b_m1, h_b_m1, b_size * sizeof(uint32_t), ctx->stream_0);

    // Ring M2: Vulkan (Intel UHD) or CUDA (same GPU, second stream)
    if (ctx->vk_ring2) {
        sp_crt_vulkan_ctx_t *vk = (sp_crt_vulkan_ctx_t *)ctx->vk_ctx;
        sp_crt_vulkan_memcpy_h2d(vk, ctx->d_a_m2, h_a_m2, a_size * sizeof(uint32_t));
        sp_crt_vulkan_memcpy_h2d(vk, ctx->d_b_m2, h_b_m2, b_size * sizeof(uint32_t));
    } else {
        sp_crt_cuda_memcpy_h2d(ctx->d_a_m2, h_a_m2, a_size * sizeof(uint32_t), ctx->stream_1);
        sp_crt_cuda_memcpy_h2d(ctx->d_b_m2, h_b_m2, b_size * sizeof(uint32_t), ctx->stream_1);
    }

    free(h_a_m1); free(h_b_m1); free(h_a_m2); free(h_b_m2);

    // --- Step 3: Dual-ring matmul (concurrent on both GPUs) ---
    // Ring M1 (Mersenne) on CUDA stream_0
    sp_crt_cuda_matmul_mersenne(ctx->d_a_m1, ctx->d_b_m1, ctx->d_c_m1,
                                 M, N, K, ctx->stream_0);

    // Ring M2 (generic mod) — Vulkan on Intel UHD or CUDA stream_1
    if (ctx->vk_ring2) {
        sp_crt_vulkan_ctx_t *vk = (sp_crt_vulkan_ctx_t *)ctx->vk_ctx;
        sp_crt_vulkan_matmul_mod_dispatch(vk, ctx->d_a_m2, ctx->d_b_m2, ctx->d_c_m2,
                                          M, N, K, (uint32_t)SP_CRT_M2);
    } else {
        sp_crt_cuda_matmul_mod(ctx->d_a_m2, ctx->d_b_m2, ctx->d_c_m2,
                                M, N, K, SP_CRT_M2, ctx->stream_1);
    }

    // --- Step 4: D2H transfers ---
    sp_crt_cuda_memcpy_d2h(ctx->h_residue_0, ctx->d_c_m1,
                            c_size * sizeof(uint32_t), ctx->stream_0);

    if (ctx->vk_ring2) {
        sp_crt_vulkan_ctx_t *vk = (sp_crt_vulkan_ctx_t *)ctx->vk_ctx;
        sp_crt_vulkan_memcpy_d2h(vk, ctx->h_residue_1, ctx->d_c_m2,
                                  c_size * sizeof(uint32_t));
    } else {
        sp_crt_cuda_memcpy_d2h(ctx->h_residue_1, ctx->d_c_m2,
                                c_size * sizeof(uint32_t), ctx->stream_1);
    }

    // --- Step 5: Synchronize both GPUs ---
    sp_crt_cuda_stream_sync(ctx->stream_0);
    if (ctx->vk_ring2) {
        sp_crt_vulkan_ctx_t *vk = (sp_crt_vulkan_ctx_t *)ctx->vk_ctx;
        sp_crt_vulkan_sync(vk);
    } else {
        sp_crt_cuda_stream_sync(ctx->stream_1);
    }

    // --- Step 6: Signed Garner reconstruction on host (i9 AVX-512) ---
    const double inv_scale = 1.0 / (ctx->act_quant.scale * ctx->weight_quant.scale);

    for (size_t i = 0; i < c_size; i++) {
        double signed_val = sp_crt_garner_signed(ctx->h_residue_0[i],
                                                  ctx->h_residue_1[i]);
        C[i] = (float)(signed_val * inv_scale);
    }

    return 0;
}

// ============================================================================
// Verification: single-element CRT round-trip test
// ============================================================================
//
// Tests that quantize → split → modular multiply → Garner → dequantize
// produces the correct result for a × b (scalar).

int sp_crt_verify_roundtrip(float a, float b, float *error_out) {
    sp_crt_quant_t q;
    sp_crt_quant_calibrate(&q, -4.0f, 4.0f);

    // Quantize to both rings (zero-centered symmetric)
    uint32_t qa_m1 = sp_crt_quantize_symmetric(a, q.scale, SP_CRT_M1);
    uint32_t qb_m1 = sp_crt_quantize_symmetric(b, q.scale, SP_CRT_M1);
    uint32_t qa_m2 = sp_crt_quantize_symmetric(a, q.scale, SP_CRT_M2);
    uint32_t qb_m2 = sp_crt_quantize_symmetric(b, q.scale, SP_CRT_M2);

    // Multiply in each ring
    uint32_t r1 = sp_crt_mersenne_reduce((uint64_t)qa_m1 * qb_m1);
    uint32_t r2 = sp_crt_m2_reduce((uint64_t)qa_m2 * qb_m2);

    // Signed Garner reconstruction
    double signed_product = sp_crt_garner_signed(r1, r2);

    // Dequantize: product_int = round(a*s) * round(b*s) ≈ a*b*s²
    // So a*b ≈ product_int / s²
    double inv_s2 = 1.0 / (q.scale * q.scale);
    float result = (float)(signed_product * inv_s2);

    float expected = a * b;
    float err = fabsf(result - expected);

    if (error_out) *error_out = err;
    return (err < 0.01f) ? 0 : -1;
}
