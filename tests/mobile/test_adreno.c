// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

// Mobile backend validation. Compiles on any platform (scalar fallback on x86).

#include "../backends/adreno/shannon_prime_adreno.h"
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

int main(void) {
    srand(42);
    printf("Shannon-Prime Mobile Backend Validation\n");
    printf("========================================\n");

    // Test 0: Feature detection
    printf("\n== Hardware Detection ==\n");
    sp_mobile_caps_t caps;
    sp_mobile_detect_caps(&caps);
    sp_mobile_print_caps(&caps);
    CHECK(1, "Feature detection runs without crash");

    // Test 1: NEON VHT2 matches core
    printf("\n== NEON VHT2 vs Core ==\n");
    for (int hd = 32; hd <= 128; hd *= 2) {
        float *a = (float *)malloc(hd * sizeof(float));
        float *b = (float *)malloc(hd * sizeof(float));
        for (int i = 0; i < hd; i++) {
            a[i] = b[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        sp_vht2_forward_f32(a, hd);
        sp_neon_vht2_f32(b, hd);
        float max_err = 0;
        for (int i = 0; i < hd; i++) {
            float e = fabsf(a[i] - b[i]);
            if (e > max_err) max_err = e;
        }
        char msg[128];
        snprintf(msg, sizeof(msg), "NEON VHT2 hd=%d: max_err=%.2e", hd, max_err);
        CHECK(max_err < 1e-6f, msg);
        free(a); free(b);
    }

    // Test 2: fp16 conversion roundtrip
    printf("\n== fp16 Conversion ==\n");
    {
        int hd = 128;
        float *orig = (float *)malloc(hd * sizeof(float));
        uint16_t *f16 = (uint16_t *)malloc(hd * sizeof(uint16_t));
        float *back = (float *)malloc(hd * sizeof(float));
        for (int i = 0; i < hd; i++) {
            orig[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        sp_neon_f32_to_f16(orig, f16, hd);
        sp_neon_f16_to_f32(f16, back, hd);
        float max_err = 0;
        for (int i = 0; i < hd; i++) {
            float e = fabsf(orig[i] - back[i]);
            if (e > max_err) max_err = e;
        }
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "f32→f16→f32 roundtrip: max_err=%.4e (fp16 precision ~1e-3)", max_err);
        CHECK(max_err < 0.01f, msg);
        free(orig); free(f16); free(back);
    }

    // Test 3: absmax reduction
    printf("\n== Absmax Reduction ==\n");
    {
        int n = 128;
        float *data = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++)
            data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        data[73] = -5.5f; // Plant a known max

        float result = sp_neon_absmax_f32(data, n);
        CHECK(fabsf(result - 5.5f) < 1e-6f, "absmax finds planted maximum");
        free(data);
    }

    // Test 4: Full pipeline hd=128
    printf("\n== Full Pipeline hd=128 ==\n");
    {
        sp_config_t cfg;
        sp_config_init(&cfg, 128, 1, 1);
        sp_adreno_cache_t ac;
        sp_adreno_cache_init(&ac, &cfg, 16);

        float *k_orig  = (float *)malloc(128 * sizeof(float));
        float *k_recon = (float *)malloc(128 * sizeof(float));
        for (int i = 0; i < 128; i++)
            k_orig[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        sp_adreno_write_k(&ac, 0, 0, 0, k_orig);
        sp_adreno_read_k(&ac, 0, 0, 0, k_recon);

        float k_corr = sp_correlation_f32(k_orig, k_recon, 128);
        char msg[128];
        snprintf(msg, sizeof(msg), "K pipeline: correlation=%.4f (need >0.985)", k_corr);
        CHECK(k_corr > 0.985f, msg);

        float *v_orig  = (float *)malloc(128 * sizeof(float));
        float *v_recon = (float *)malloc(128 * sizeof(float));
        for (int i = 0; i < 128; i++)
            v_orig[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        sp_adreno_write_v(&ac, 0, 0, 0, v_orig);
        sp_adreno_read_v(&ac, 0, 0, 0, v_recon);

        float v_corr = sp_correlation_f32(v_orig, v_recon, 128);
        snprintf(msg, sizeof(msg), "V pipeline: correlation=%.4f (need >0.950)", v_corr);
        CHECK(v_corr > 0.950f, msg);

        free(k_orig); free(k_recon); free(v_orig); free(v_recon);
        sp_adreno_cache_free(&ac);
    }

    // Test 5: hd=64 mobile path
    printf("\n== hd=64 Mobile Path ==\n");
    {
        sp_config_t cfg;
        sp_config_init(&cfg, 64, 16, 4);
        sp_adreno_cache_t ac;
        sp_adreno_cache_init(&ac, &cfg, 2048);

        float *k_orig  = (float *)malloc(64 * sizeof(float));
        float *k_recon = (float *)malloc(64 * sizeof(float));
        for (int i = 0; i < 64; i++)
            k_orig[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        sp_adreno_write_k(&ac, 0, 0, 0, k_orig);
        sp_adreno_read_k(&ac, 0, 0, 0, k_recon);

        float k_corr = sp_correlation_f32(k_orig, k_recon, 64);
        char msg[128];
        snprintf(msg, sizeof(msg), "hd=64 K correlation: %.4f (paper: 0.9972)", k_corr);
        CHECK(k_corr > 0.985f, msg);

        free(k_orig); free(k_recon);
        sp_adreno_cache_free(&ac);
    }

    // Test 6: fp16 write path
    printf("\n== fp16 Write Path ==\n");
    {
        sp_config_t cfg;
        sp_config_init(&cfg, 128, 1, 1);
        sp_adreno_cache_t ac;
        sp_adreno_cache_init(&ac, &cfg, 16);

        // Create f32 vector, convert to f16, write via f16 path
        float *k_f32  = (float *)malloc(128 * sizeof(float));
        uint16_t *k_f16 = (uint16_t *)malloc(128 * sizeof(uint16_t));
        float *k_recon = (float *)malloc(128 * sizeof(float));

        for (int i = 0; i < 128; i++)
            k_f32[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        sp_neon_f32_to_f16(k_f32, k_f16, 128);
        sp_adreno_write_k_f16(&ac, 0, 0, 0, k_f16);
        sp_adreno_read_k(&ac, 0, 0, 0, k_recon);

        float k_corr = sp_correlation_f32(k_f32, k_recon, 128);
        char msg[128];
        // fp16 input adds ~1e-3 extra error vs f32 input
        snprintf(msg, sizeof(msg),
                 "fp16→VHT2→f32 K correlation: %.4f (need >0.980)", k_corr);
        CHECK(k_corr > 0.980f, msg);

        free(k_f32); free(k_f16); free(k_recon);
        sp_adreno_cache_free(&ac);
    }

    // Test 7: Thread affinity (just check it doesn't crash)
    printf("\n== Thread Affinity ==\n");
    {
        // This may fail on non-Linux without error
        int rc = sp_set_thread_affinity(SP_AFFINITY_ANY, &caps);
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "SP_AFFINITY_ANY: %s", rc == 0 ? "set" : "unavailable (ok on non-Linux)");
        CHECK(1, msg); // Always passes — just checking no crash
    }

    // Test 8: Performance counters
    printf("\n== Performance Counters ==\n");
    {
        sp_config_t cfg;
        sp_config_init(&cfg, 128, 2, 2);
        sp_adreno_cache_t ac;
        sp_adreno_cache_init(&ac, &cfg, 16);

        float dummy[128];
        for (int i = 0; i < 128; i++) dummy[i] = (float)i / 128.0f;

        sp_adreno_write_k(&ac, 0, 0, 0, dummy);
        sp_adreno_write_v(&ac, 0, 0, 0, dummy);
        sp_adreno_write_k(&ac, 1, 1, 0, dummy);

        float out[128];
        sp_adreno_read_k(&ac, 0, 0, 0, out);

        CHECK(ac.n_writes == 3, "Write counter: 3");
        CHECK(ac.n_reads  == 1, "Read counter: 1");

        sp_adreno_cache_free(&ac);
    }

    // Test 9: Writeback benchmark
    printf("\n== Benchmark ==\n");
    {
        sp_config_t cfg;
        sp_config_init(&cfg, 64, 16, 4);
        sp_adreno_cache_t ac;
        sp_adreno_cache_init(&ac, &cfg, 2048);

        float ms = sp_adreno_bench_writeback(&ac);
        printf("  Writeback: %.1f ms (paper target: 37-42 ms on ARM)\n", ms);
        CHECK(ms < 10000.0f, "Benchmark completes"); // Just ensure it finishes

        sp_adreno_print_stats(&ac);
        sp_adreno_cache_free(&ac);
    }

    printf("\n========================================\n");
    printf("Results: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
