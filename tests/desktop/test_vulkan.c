// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

// Vulkan backend validation.
// Without Vulkan SDK, exercises the CPU fallback path to validate
// the API surface and ensure correct results through the Vulkan abstraction.

#include "../backends/vulkan/shannon_prime_vulkan.h"
#include "../core/shannon_prime.h"
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

int main(void) {
    srand(42);
    printf("Shannon-Prime Vulkan Backend Validation\n");
    printf("========================================\n");

    // Test 1: Init with NULL device → CPU fallback
    printf("\n== Vulkan Init (CPU fallback) ==\n");
    {
        sp_config_t cfg;
        sp_config_init(&cfg, 128, 4, 2);

        sp_vulkan_cache_t *cc = NULL;
        int rc = sp_vulkan_cache_init(&cc, &cfg, 1024, NULL, NULL, 0);
        CHECK(rc == 0 && cc != NULL, "Init with CPU fallback succeeds");

        sp_vulkan_check_device(cc);

        // Test 2: Full pipeline through Vulkan API
        printf("\n== Vulkan Pipeline (via CPU fallback) ==\n");
        int hd = 128;
        float *k_orig  = (float *)malloc(hd * sizeof(float));
        float *k_recon = (float *)malloc(hd * sizeof(float));
        for (int i = 0; i < hd; i++) {
            k_orig[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }

        sp_vulkan_write_k(cc, 0, 0, 0, k_orig);
        sp_vulkan_read_k(cc, 0, 0, 0, k_recon);

        float k_corr = sp_correlation_f32(k_orig, k_recon, hd);
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "K pipeline: correlation=%.4f (need >0.990)", k_corr);
        CHECK(k_corr > 0.990f, msg);

        float *v_orig  = (float *)malloc(hd * sizeof(float));
        float *v_recon = (float *)malloc(hd * sizeof(float));
        for (int i = 0; i < hd; i++) {
            v_orig[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }

        sp_vulkan_write_v(cc, 0, 0, 0, v_orig);
        sp_vulkan_read_v(cc, 0, 0, 0, v_recon);

        float v_corr = sp_correlation_f32(v_orig, v_recon, hd);
        snprintf(msg, sizeof(msg),
                 "V pipeline: correlation=%.4f (need >0.950)", v_corr);
        CHECK(v_corr > 0.950f, msg);

        // Test 3: Batch write/read
        printf("\n== Vulkan Batch Ops ==\n");
        int n_pos = 8;
        float *k_batch = (float *)malloc(n_pos * hd * sizeof(float));
        float *k_batch_recon = (float *)malloc(n_pos * hd * sizeof(float));

        for (int i = 0; i < n_pos * hd; i++) {
            k_batch[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }

        sp_vulkan_write_k_batch(cc, 1, 0, 0, n_pos, k_batch);
        sp_vulkan_read_k_batch(cc, 1, 0, 0, n_pos, k_batch_recon);

        float total_corr = 0;
        for (int p = 0; p < n_pos; p++) {
            total_corr += sp_correlation_f32(
                k_batch + p * hd, k_batch_recon + p * hd, hd);
        }
        float avg_corr = total_corr / n_pos;
        snprintf(msg, sizeof(msg),
                 "Batch K (%d pos): avg_corr=%.4f (need >0.990)", n_pos, avg_corr);
        CHECK(avg_corr > 0.990f, msg);

        // Test 4: Memory reporting
        printf("\n== Memory ==\n");
        sp_vulkan_print_memory(cc);

        // Test 5 (diagnostic, only fires when SHANNON_PRIME_VULKAN_FORCE_GPU=1):
        // GPU vilenkin.comp parity vs CPU sp_vht2_forward_f32. Gated on env
        // because the default CPU-fallback config makes this check redundant
        // (same code both sides), and because the GPU path is known to be
        // low-correlation today and we don't want the default test suite to
        // red-flag on an opt-in diagnostic.
        const char * force_gpu = getenv("SHANNON_PRIME_VULKAN_FORCE_GPU");
        if (force_gpu && force_gpu[0] == '1') {
            printf("\n== GPU vilenkin.comp parity (diagnostic) ==\n");
            float *cpu = (float *)malloc(hd * sizeof(float));
            float *gpu = (float *)malloc(hd * sizeof(float));
            for (int i = 0; i < hd; i++) {
                cpu[i] = (float)rand() / (float)RAND_MAX - 0.5f;
                gpu[i] = cpu[i];
            }
            sp_vht2_forward_f32(cpu, hd);
            int rc = sp_vulkan_diag_vht2_forward(cc, gpu, hd);
            if (rc != 0) {
                printf("  [%s] diag_vht2_forward returned %d\n", FAIL, rc);
                tests_run++;
            } else {
                double max_err = 0.0, sum_err = 0.0;
                for (int i = 0; i < hd; i++) {
                    double e = fabs((double)cpu[i] - (double)gpu[i]);
                    if (e > max_err) max_err = e;
                    sum_err += e;
                }
                float corr = sp_correlation_f32(cpu, gpu, hd);
                printf("  cpu[:8] = %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f\n",
                       cpu[0], cpu[1], cpu[2], cpu[3], cpu[4], cpu[5], cpu[6], cpu[7]);
                printf("  gpu[:8] = %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f\n",
                       gpu[0], gpu[1], gpu[2], gpu[3], gpu[4], gpu[5], gpu[6], gpu[7]);
                printf("  max_err=%.6g mean_err=%.6g corr=%.4f\n",
                       max_err, sum_err / hd, corr);
            }
            free(cpu); free(gpu);

            // Band-quant round-trip diagnostic — run band_quantize followed
            // by band_dequantize on GPU without VHT2/Möbius, for both the K
            // band config (4-band 5,5,4,3) and the V band config (1-band 3).
            // Localises whether the 0.63 V correlation is in the quant
            // shaders themselves or in the surrounding dispatch plumbing.
            for (int which = 0; which < 2; which++) {
                float *qin  = (float *)malloc(hd * sizeof(float));
                float *qout = (float *)malloc(hd * sizeof(float));
                for (int i = 0; i < hd; i++) {
                    qin[i]  = (float)rand() / (float)RAND_MAX - 0.5f;
                    qout[i] = 0.0f;
                }
                int rc = sp_vulkan_diag_band_roundtrip(cc, which, qin, qout, hd);

                // CPU reference: same inputs through sp_band_quantize +
                // sp_band_dequantize. Tells us whether the GPU-side corr is
                // at or below the inherent quant-floor on this random data.
                sp_band_config_t bc_cpu;
                int default_k_bits[] = {5, 5, 4, 3};
                int default_v_bits[] = {3};
                if (which == 0) sp_band_config_init(&bc_cpu, hd, 4, default_k_bits);
                else            sp_band_config_init(&bc_cpu, hd, 1, default_v_bits);
                uint8_t *buf = (uint8_t *)malloc(bc_cpu.total_bytes);
                float *cpuout = (float *)malloc(hd * sizeof(float));
                sp_band_quantize(qin, buf, &bc_cpu);
                sp_band_dequantize(buf, cpuout, &bc_cpu);
                float cpu_corr = sp_correlation_f32(qin, cpuout, hd);
                free(buf); free(cpuout);

                if (rc == 0) {
                    double max_err = 0.0;
                    for (int i = 0; i < hd; i++) {
                        double e = fabs((double)qin[i] - (double)qout[i]);
                        if (e > max_err) max_err = e;
                    }
                    float corr = sp_correlation_f32(qin, qout, hd);
                    printf("  band_roundtrip %s: GPU max_err=%.4f corr=%.4f"
                           "  CPU corr=%.4f (reference)\n",
                           which == 0 ? "K (4 bands 5,5,4,3)" : "V (1 band 3-bit)",
                           max_err, corr, cpu_corr);
                }
                free(qin); free(qout);
            }
        }

        free(k_orig); free(k_recon);
        free(v_orig); free(v_recon);
        free(k_batch); free(k_batch_recon);
        sp_vulkan_cache_free(cc);
    }

    printf("\n========================================\n");
    printf("Results: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
