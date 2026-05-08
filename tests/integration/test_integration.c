// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

// Integration test for the llama.cpp layer.

#include "../tools/shannon_prime_llama.h"
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
    printf("Shannon-Prime Integration Test\n");
    printf("==============================\n");

    // Test 1: Programmatic init (no env var needed)
    printf("\n== Programmatic Init ==\n");
    {
        sp_llama_params_t params;
        memset(&params, 0, sizeof(params));
        params.head_dim    = 128;
        params.n_layers    = 32;
        params.n_heads_kv  = 8;
        params.max_seq_len = 4096;
        params.backend     = SP_BACKEND_CPU;

        sp_config_t cfg;
        sp_config_init(&cfg, 128, 32, 8);

        sp_llama_ctx_t *ctx = sp_llama_init_config(&params, &cfg);
        CHECK(ctx != NULL, "Programmatic init succeeds");

        // Test 2: Write + read K
        printf("\n== KV Write/Read ==\n");
        int hd = 128;
        float *k = (float *)malloc(hd * sizeof(float));
        float *v = (float *)malloc(hd * sizeof(float));
        float *k_out = (float *)malloc(hd * sizeof(float));
        float *v_out = (float *)malloc(hd * sizeof(float));

        for (int i = 0; i < hd; i++) {
            k[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            v[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }

        sp_llama_write_kv(ctx, 0, 0, 0, k, v);
        sp_llama_read_k(ctx, 0, 0, 0, k_out);
        sp_llama_read_v(ctx, 0, 0, 0, v_out);

        float k_corr = sp_correlation_f32(k, k_out, hd);
        float v_corr = sp_correlation_f32(v, v_out, hd);
        char msg[256];
        snprintf(msg, sizeof(msg), "K correlation: %.4f (need >0.990)", k_corr);
        CHECK(k_corr > 0.990f, msg);
        snprintf(msg, sizeof(msg), "V correlation: %.4f (need >0.950)", v_corr);
        CHECK(v_corr > 0.950f, msg);

        // Test 3: Batch write + batch read
        printf("\n== Batch Operations ==\n");
        int n_pos = 16;
        float *k_batch = (float *)malloc(n_pos * hd * sizeof(float));
        float *k_batch_out = (float *)malloc(n_pos * hd * sizeof(float));
        for (int i = 0; i < n_pos * hd; i++) {
            k_batch[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }

        sp_llama_write_k_batch(ctx, 1, 0, 0, n_pos, k_batch);
        sp_llama_read_k_batch(ctx, 1, 0, 0, n_pos, k_batch_out);

        float total_corr = 0;
        for (int i = 0; i < n_pos; i++) {
            total_corr += sp_correlation_f32(
                k_batch + i * hd, k_batch_out + i * hd, hd);
        }
        snprintf(msg, sizeof(msg), "Batch K (%d pos): avg_corr=%.4f",
                 n_pos, total_corr / n_pos);
        CHECK(total_corr / n_pos > 0.990f, msg);

        // Test 4: Multiple layers and heads
        printf("\n== Multi-layer, Multi-head ==\n");
        float min_corr = 1.0f;
        for (int l = 0; l < 4; l++) {
            for (int h = 0; h < 8; h++) {
                for (int i = 0; i < hd; i++) {
                    k[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                }
                sp_llama_write_k(ctx, l, h, 0, k);
                sp_llama_read_k(ctx, l, h, 0, k_out);
                float c = sp_correlation_f32(k, k_out, hd);
                if (c < min_corr) min_corr = c;
            }
        }
        snprintf(msg, sizeof(msg),
                 "32 K vectors (4L×8H): min_corr=%.4f", min_corr);
        // Random data minimum is ~0.98; real RoPE-structured K vectors achieve 0.997+
        CHECK(min_corr > 0.980f, msg);

        // Test 5: Memory reporting
        printf("\n== Memory ==\n");
        sp_llama_memory_t mem = sp_llama_memory(ctx);
        printf("  Positions: %d\n", mem.n_positions);
        printf("  Compressed: %.2f KB\n", mem.compressed_bytes / 1024.0);
        printf("  Baseline:   %.2f KB\n", mem.baseline_bytes / 1024.0);
        printf("  Ratio:      %.1f×\n", mem.compression_ratio);
        CHECK(mem.compression_ratio > 3.0f, "Compression ratio >3×");

        // Test 6: Validate K helper
        printf("\n== Validation Helper ==\n");
        for (int i = 0; i < hd; i++) {
            k[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        float val_corr = sp_llama_validate_k(ctx, k, hd);
        snprintf(msg, sizeof(msg), "Validate K: correlation=%.4f", val_corr);
        CHECK(val_corr > 0.990f, msg);

        free(k); free(v); free(k_out); free(v_out);
        free(k_batch); free(k_batch_out);
        sp_llama_free(ctx);
    }

    printf("\n==============================\n");
    printf("Results: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
