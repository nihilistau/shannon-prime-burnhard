// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

// CUDA backend validation.
// Exercises sp_cuda_cache on a real GPU. Backend-agnostic math (VHT2
// round-trip, Möbius function, banded quantization, Vilenkin) is covered
// by test_core; this suite verifies the GPU path produces equivalent
// correlations through the CUDA API surface.

#include "../backends/cuda/shannon_prime_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int tests_run = 0, tests_passed = 0;
#define CHECK(cond, msg) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  [%s] %s\n", PASS, msg); } \
    else { printf("  [%s] %s\n", FAIL, msg); } \
} while(0)

static void fill_random(float *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

int main(void) {
    srand(42);
    printf("Shannon-Prime CUDA Backend Validation\n");
    printf("======================================\n");

    // Verify a CUDA device is present before we touch the cache.
    int device_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&device_count);
    if (cerr != cudaSuccess || device_count < 1) {
        printf("  [%s] No CUDA device available (%s)\n",
               FAIL, cudaGetErrorString(cerr));
        return 1;
    }
    printf("  CUDA devices detected: %d\n\n", device_count);

    const int hd       = 128;
    const int n_layers = 4;
    const int n_heads  = 2;
    const int max_seq  = 1024;

    // ── Test 1: cache init ──────────────────────────────────────
    printf("== CUDA Init ==\n");
    sp_config_t cfg;
    sp_config_init(&cfg, hd, n_layers, n_heads);

    sp_cuda_cache_t cc;
    int rc = sp_cuda_cache_init(&cc, &cfg, max_seq, NULL);
    CHECK(rc == 0, "sp_cuda_cache_init succeeds (default stream)");
    if (rc != 0) {
        printf("\nResults: %d/%d passed\n", tests_passed, tests_run);
        return 1;
    }

    // Staging buffers: API takes device pointers, so we stage via cudaMemcpy.
    float *h_vec  = (float *)malloc(hd * sizeof(float));
    float *h_recon = (float *)malloc(hd * sizeof(float));
    float *d_vec, *d_recon;
    cudaMalloc((void **)&d_vec,   hd * sizeof(float));
    cudaMalloc((void **)&d_recon, hd * sizeof(float));

    char msg[160];

    // ── Test 2: K single-vector pipeline ────────────────────────
    printf("\n== CUDA K Pipeline ==\n");
    fill_random(h_vec, hd);
    cudaMemcpy(d_vec, h_vec, hd * sizeof(float), cudaMemcpyHostToDevice);
    sp_cuda_write_k(&cc, 0, 0, 0, d_vec);
    sp_cuda_read_k(&cc, 0, 0, 0, d_recon);
    cudaMemcpy(h_recon, d_recon, hd * sizeof(float), cudaMemcpyDeviceToHost);
    float k_corr = sp_correlation_f32(h_vec, h_recon, hd);
    snprintf(msg, sizeof(msg),
             "K pipeline: correlation=%.4f (need >0.990)", k_corr);
    CHECK(k_corr > 0.990f, msg);

    // ── Test 3: V single-vector pipeline ────────────────────────
    printf("\n== CUDA V Pipeline ==\n");
    fill_random(h_vec, hd);
    cudaMemcpy(d_vec, h_vec, hd * sizeof(float), cudaMemcpyHostToDevice);
    sp_cuda_write_v(&cc, 0, 0, 0, d_vec);
    sp_cuda_read_v(&cc, 0, 0, 0, d_recon);
    cudaMemcpy(h_recon, d_recon, hd * sizeof(float), cudaMemcpyDeviceToHost);
    float v_corr = sp_correlation_f32(h_vec, h_recon, hd);
    snprintf(msg, sizeof(msg),
             "V pipeline: correlation=%.4f (need >0.950)", v_corr);
    CHECK(v_corr > 0.950f, msg);

    // ── Test 4: K batch ops ─────────────────────────────────────
    printf("\n== CUDA K Batch (8 positions) ==\n");
    {
        int n_pos = 8;
        float *h_batch  = (float *)malloc(n_pos * hd * sizeof(float));
        float *h_brecon = (float *)malloc(n_pos * hd * sizeof(float));
        float *d_batch, *d_brecon;
        cudaMalloc((void **)&d_batch,  n_pos * hd * sizeof(float));
        cudaMalloc((void **)&d_brecon, n_pos * hd * sizeof(float));

        fill_random(h_batch, n_pos * hd);
        cudaMemcpy(d_batch, h_batch, n_pos * hd * sizeof(float),
                   cudaMemcpyHostToDevice);

        sp_cuda_write_k_batch(&cc, 1, 0, 0, n_pos, d_batch);
        sp_cuda_read_k_batch(&cc, 1, 0, 0, n_pos, d_brecon);
        cudaMemcpy(h_brecon, d_brecon, n_pos * hd * sizeof(float),
                   cudaMemcpyDeviceToHost);

        float total = 0.0f;
        for (int p = 0; p < n_pos; p++) {
            total += sp_correlation_f32(h_batch + p * hd,
                                        h_brecon + p * hd, hd);
        }
        float avg = total / n_pos;
        snprintf(msg, sizeof(msg),
                 "Batch K (%d pos): avg_corr=%.4f (need >0.990)", n_pos, avg);
        CHECK(avg > 0.990f, msg);

        free(h_batch); free(h_brecon);
        cudaFree(d_batch); cudaFree(d_brecon);
    }

    // ── Test 5: V batch ops ─────────────────────────────────────
    printf("\n== CUDA V Batch (8 positions) ==\n");
    {
        int n_pos = 8;
        float *h_batch  = (float *)malloc(n_pos * hd * sizeof(float));
        float *h_brecon = (float *)malloc(n_pos * hd * sizeof(float));
        float *d_batch, *d_brecon;
        cudaMalloc((void **)&d_batch,  n_pos * hd * sizeof(float));
        cudaMalloc((void **)&d_brecon, n_pos * hd * sizeof(float));

        fill_random(h_batch, n_pos * hd);
        cudaMemcpy(d_batch, h_batch, n_pos * hd * sizeof(float),
                   cudaMemcpyHostToDevice);

        sp_cuda_write_v_batch(&cc, 1, 0, 0, n_pos, d_batch);
        sp_cuda_read_v_batch(&cc, 1, 0, 0, n_pos, d_brecon);
        cudaMemcpy(h_brecon, d_brecon, n_pos * hd * sizeof(float),
                   cudaMemcpyDeviceToHost);

        float total = 0.0f;
        for (int p = 0; p < n_pos; p++) {
            total += sp_correlation_f32(h_batch + p * hd,
                                        h_brecon + p * hd, hd);
        }
        float avg = total / n_pos;
        snprintf(msg, sizeof(msg),
                 "Batch V (%d pos): avg_corr=%.4f (need >0.950)", n_pos, avg);
        CHECK(avg > 0.950f, msg);

        free(h_batch); free(h_brecon);
        cudaFree(d_batch); cudaFree(d_brecon);
    }

    // ── Test 6: multi-layer, multi-head — catches per-slot indexing bugs ─
    printf("\n== CUDA Multi-Layer/Head (4L × 2H × 4 pos K) ==\n");
    {
        float min_corr = 1.0f;
        int   n_pos    = 4;
        for (int L = 0; L < n_layers; L++) {
            for (int H = 0; H < n_heads; H++) {
                for (int p = 0; p < n_pos; p++) {
                    fill_random(h_vec, hd);
                    cudaMemcpy(d_vec, h_vec, hd * sizeof(float),
                               cudaMemcpyHostToDevice);
                    sp_cuda_write_k(&cc, L, H, p, d_vec);
                    sp_cuda_read_k(&cc, L, H, p, d_recon);
                    cudaMemcpy(h_recon, d_recon, hd * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    float c = sp_correlation_f32(h_vec, h_recon, hd);
                    if (c < min_corr) min_corr = c;
                }
            }
        }
        int total = n_layers * n_heads * n_pos;
        snprintf(msg, sizeof(msg),
                 "Multi-layer/head (%d vectors): min_corr=%.4f (need >0.980)",
                 total, min_corr);
        CHECK(min_corr > 0.980f, msg);
    }

    // ── Test 7: memory diagnostic ───────────────────────────────
    printf("\n== CUDA Memory ==\n");
    sp_cuda_print_memory(&cc);
    CHECK(1, "sp_cuda_print_memory completes without crash");

    // Cleanup
    free(h_vec); free(h_recon);
    cudaFree(d_vec); cudaFree(d_recon);
    sp_cuda_cache_free(&cc);

    printf("\n======================================\n");
    printf("Results: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
