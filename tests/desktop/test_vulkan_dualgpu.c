// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

// Dual-GPU Vulkan validation.
// Requires a system with 2+ Vulkan-capable GPUs (e.g. discrete + iGPU).
// Tests:
//   1. Independent init on each GPU
//   2. K/V pipeline on each GPU independently
//   3. Cross-device transfer: compress on GPU0, decompress, re-compress on GPU1
//   4. Batch operations on iGPU (large system memory pool)

#include "../backends/vulkan/shannon_prime_vulkan.h"
#include "../core/shannon_prime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int tests_run = 0, tests_passed = 0;
#define CHECK(cond, msg) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  [%s] %s\n", PASS, msg); } \
    else { printf("  [%s] %s\n", FAIL, msg); } \
} while(0)

static float rand_float(void) {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

// ── Test 1: Independent init on two GPUs ────────────────────────────
static void test_dual_init(sp_vulkan_cache_t **cc0, sp_vulkan_cache_t **cc1,
                           const sp_config_t *cfg, int max_seq) {
    printf("\n== Dual-GPU Init ==\n");

    int rc0 = sp_vulkan_cache_init(cc0, cfg, max_seq, NULL, NULL, 0);
    CHECK(rc0 == 0 && *cc0 != NULL, "GPU 0 init succeeds");

    int rc1 = sp_vulkan_cache_init(cc1, cfg, max_seq, NULL, NULL, 1);
    CHECK(rc1 == 0 && *cc1 != NULL, "GPU 1 init succeeds");
}

// ── Test 2: K/V pipeline on each GPU ────────────────────────────────
static void test_pipeline_each_gpu(sp_vulkan_cache_t *cc0, sp_vulkan_cache_t *cc1,
                                    int hd) {
    printf("\n== Pipeline on each GPU ==\n");
    char msg[256];

    float *k_orig  = (float *)malloc(hd * sizeof(float));
    float *k_out0  = (float *)malloc(hd * sizeof(float));
    float *k_out1  = (float *)malloc(hd * sizeof(float));
    float *v_orig  = (float *)malloc(hd * sizeof(float));
    float *v_out0  = (float *)malloc(hd * sizeof(float));
    float *v_out1  = (float *)malloc(hd * sizeof(float));

    for (int i = 0; i < hd; i++) {
        k_orig[i] = rand_float();
        v_orig[i] = rand_float();
    }

    // GPU 0
    sp_vulkan_write_k(cc0, 0, 0, 0, k_orig);
    sp_vulkan_read_k(cc0, 0, 0, 0, k_out0);
    float k_corr0 = sp_correlation_f32(k_orig, k_out0, hd);
    snprintf(msg, sizeof(msg), "GPU 0 K pipeline: corr=%.4f (need >0.990)", k_corr0);
    CHECK(k_corr0 > 0.990f, msg);

    sp_vulkan_write_v(cc0, 0, 0, 0, v_orig);
    sp_vulkan_read_v(cc0, 0, 0, 0, v_out0);
    float v_corr0 = sp_correlation_f32(v_orig, v_out0, hd);
    snprintf(msg, sizeof(msg), "GPU 0 V pipeline: corr=%.4f (need >0.950)", v_corr0);
    CHECK(v_corr0 > 0.950f, msg);

    // GPU 1
    sp_vulkan_write_k(cc1, 0, 0, 0, k_orig);
    sp_vulkan_read_k(cc1, 0, 0, 0, k_out1);
    float k_corr1 = sp_correlation_f32(k_orig, k_out1, hd);
    snprintf(msg, sizeof(msg), "GPU 1 K pipeline: corr=%.4f (need >0.990)", k_corr1);
    CHECK(k_corr1 > 0.990f, msg);

    sp_vulkan_write_v(cc1, 0, 0, 0, v_orig);
    sp_vulkan_read_v(cc1, 0, 0, 0, v_out1);
    float v_corr1 = sp_correlation_f32(v_orig, v_out1, hd);
    snprintf(msg, sizeof(msg), "GPU 1 V pipeline: corr=%.4f (need >0.950)", v_corr1);
    CHECK(v_corr1 > 0.950f, msg);

    free(k_orig); free(k_out0); free(k_out1);
    free(v_orig); free(v_out0); free(v_out1);
}

// ── Test 3: Cross-device transfer ───────────────────────────────────
// Write original on GPU0, read back, write that reconstruction to GPU1,
// read back from GPU1. The GPU1 output should correlate with GPU0's output
// at >0.999 (both saw the same input through the same quantize path).
static void test_cross_device(sp_vulkan_cache_t *cc0, sp_vulkan_cache_t *cc1,
                               int hd) {
    printf("\n== Cross-Device Transfer ==\n");
    char msg[256];
    int n_pos = 4;

    float *k_orig  = (float *)malloc(n_pos * hd * sizeof(float));
    float *k_from0 = (float *)malloc(n_pos * hd * sizeof(float));
    float *k_from1 = (float *)malloc(n_pos * hd * sizeof(float));

    for (int i = 0; i < n_pos * hd; i++)
        k_orig[i] = rand_float();

    // Write to GPU0, read back
    for (int p = 0; p < n_pos; p++)
        sp_vulkan_write_k(cc0, 0, 0, p, k_orig + p * hd);
    for (int p = 0; p < n_pos; p++)
        sp_vulkan_read_k(cc0, 0, 0, p, k_from0 + p * hd);

    // Write GPU0's reconstruction to GPU1, read back
    for (int p = 0; p < n_pos; p++)
        sp_vulkan_write_k(cc1, 0, 0, p, k_from0 + p * hd);
    for (int p = 0; p < n_pos; p++)
        sp_vulkan_read_k(cc1, 0, 0, p, k_from1 + p * hd);

    // GPU0 output vs original
    float total0 = 0;
    for (int p = 0; p < n_pos; p++)
        total0 += sp_correlation_f32(k_orig + p * hd, k_from0 + p * hd, hd);
    float avg0 = total0 / n_pos;
    snprintf(msg, sizeof(msg),
             "GPU0→host round-trip (%d pos): avg_corr=%.4f (need >0.990)", n_pos, avg0);
    CHECK(avg0 > 0.990f, msg);

    // GPU0 output vs GPU1 output (double-quantized)
    float total_cross = 0;
    for (int p = 0; p < n_pos; p++)
        total_cross += sp_correlation_f32(k_from0 + p * hd, k_from1 + p * hd, hd);
    float avg_cross = total_cross / n_pos;
    snprintf(msg, sizeof(msg),
             "GPU0→GPU1 cross-device (%d pos): avg_corr=%.4f (need >0.980)", n_pos, avg_cross);
    CHECK(avg_cross > 0.980f, msg);

    // Original vs GPU1 (double compress/decompress)
    float total_e2e = 0;
    for (int p = 0; p < n_pos; p++)
        total_e2e += sp_correlation_f32(k_orig + p * hd, k_from1 + p * hd, hd);
    float avg_e2e = total_e2e / n_pos;
    snprintf(msg, sizeof(msg),
             "Original→GPU0→GPU1 end-to-end (%d pos): avg_corr=%.4f (need >0.970)", n_pos, avg_e2e);
    CHECK(avg_e2e > 0.970f, msg);

    free(k_orig); free(k_from0); free(k_from1);
}

// ── Test 4: Batch ops on GPU1 (iGPU / large memory pool) ───────────
static void test_igpu_batch(sp_vulkan_cache_t *cc1, int hd) {
    printf("\n== iGPU Batch Stress ==\n");
    char msg[256];
    int n_pos = 64;  // larger batch — iGPU has system memory

    float *k_batch = (float *)malloc(n_pos * hd * sizeof(float));
    float *k_recon = (float *)malloc(n_pos * hd * sizeof(float));

    for (int i = 0; i < n_pos * hd; i++)
        k_batch[i] = rand_float();

    sp_vulkan_write_k_batch(cc1, 1, 0, 0, n_pos, k_batch);
    sp_vulkan_read_k_batch(cc1, 1, 0, 0, n_pos, k_recon);

    float total = 0;
    for (int p = 0; p < n_pos; p++)
        total += sp_correlation_f32(k_batch + p * hd, k_recon + p * hd, hd);
    float avg = total / n_pos;
    snprintf(msg, sizeof(msg),
             "iGPU batch K (%d pos): avg_corr=%.4f (need >0.990)", n_pos, avg);
    CHECK(avg > 0.990f, msg);

    // V batch
    float *v_batch = (float *)malloc(n_pos * hd * sizeof(float));
    float *v_recon = (float *)malloc(n_pos * hd * sizeof(float));
    for (int i = 0; i < n_pos * hd; i++)
        v_batch[i] = rand_float();

    for (int p = 0; p < n_pos; p++)
        sp_vulkan_write_v(cc1, 1, 0, p, v_batch + p * hd);
    for (int p = 0; p < n_pos; p++)
        sp_vulkan_read_v(cc1, 1, 0, p, v_recon + p * hd);

    total = 0;
    for (int p = 0; p < n_pos; p++)
        total += sp_correlation_f32(v_batch + p * hd, v_recon + p * hd, hd);
    avg = total / n_pos;
    snprintf(msg, sizeof(msg),
             "iGPU batch V (%d pos): avg_corr=%.4f (need >0.950)", n_pos, avg);
    CHECK(avg > 0.950f, msg);

    free(k_batch); free(k_recon);
    free(v_batch); free(v_recon);
}

// ── Test 5: Out-of-range gpu_index ──────────────────────────────────
static void test_bad_index(const sp_config_t *cfg) {
    printf("\n== Invalid gpu_index ==\n");
    sp_vulkan_cache_t *bad = NULL;
    int rc = sp_vulkan_cache_init(&bad, cfg, 256, NULL, NULL, 99);
    // Should fail gracefully or fall back to CPU
    CHECK(rc == -1 || bad != NULL, "gpu_index=99 handled gracefully");
    if (bad) sp_vulkan_cache_free(bad);
}

int main(void) {
    srand(42);
    printf("Shannon-Prime Dual-GPU Vulkan Validation\n");
    printf("=========================================\n");

    sp_config_t cfg;
    sp_config_init(&cfg, 128, 4, 2);
    int max_seq = 1024;
    int hd = 128;

    sp_vulkan_cache_t *cc0 = NULL, *cc1 = NULL;

    // Init both GPUs
    test_dual_init(&cc0, &cc1, &cfg, max_seq);
    if (!cc0 || !cc1) {
        printf("\nCannot proceed — need 2 GPUs. Skipping remaining tests.\n");
        if (cc0) sp_vulkan_cache_free(cc0);
        if (cc1) sp_vulkan_cache_free(cc1);
        printf("\nResults: %d/%d passed\n", tests_passed, tests_run);
        return (tests_passed == tests_run) ? 0 : 1;
    }

    test_pipeline_each_gpu(cc0, cc1, hd);
    test_cross_device(cc0, cc1, hd);
    test_igpu_batch(cc1, hd);
    test_bad_index(&cfg);

    sp_vulkan_cache_free(cc0);
    sp_vulkan_cache_free(cc1);

    printf("\n=========================================\n");
    printf("Results: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
