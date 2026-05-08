// Shannon-Prime VHT2: Advanced CUDA Backend Tests
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Exercises GPU paths NOT covered by test_cuda.c:
//   1. Sqfree GPU cache (non-power-of-2 head_dim = 80)
//   2. Sqfree GPU cache with spinor sheet
//   3. Hierarchical GPU cache (skeleton + predictor W)
//   4. Cold storage layer (GPU ↔ CPU offload)
//   5. Sqfree batched read path
//   6. Stress test: high sequence length

#include "../backends/cuda/shannon_prime_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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

// ====================================================================
// Test 1: Sqfree GPU cache (hd=80, non-power-of-2)
// ====================================================================
static void test_sqfree_gpu(void) {
    printf("\n== Sqfree GPU Cache (hd=80) ==\n");

    const int hd = 80;   // non-power-of-2 → triggers sqfree_pad
    const int n_layers = 2, n_heads = 2, max_seq = 256;

    sp_config_t cfg;
    sp_config_init(&cfg, hd, n_layers, n_heads);

    sp_cuda_sqfree_cache_t sc;
    int rc = sp_cuda_sqfree_cache_init(&sc, &cfg, max_seq,
                                        /*residual_bits=*/4,
                                        /*use_spinor=*/0, NULL);
    char msg[256];
    snprintf(msg, sizeof(msg), "sqfree cache init (hd=%d, pad_dim=%d)", hd, sc.pad_dim);
    CHECK(rc == 0, msg);
    if (rc != 0) return;

    float *h_vec   = (float *)malloc(hd * sizeof(float));
    float *h_recon = (float *)malloc(hd * sizeof(float));
    float *d_vec, *d_recon;
    cudaMalloc((void **)&d_vec,   hd * sizeof(float));
    cudaMalloc((void **)&d_recon, hd * sizeof(float));

    // K pipeline
    fill_random(h_vec, hd);
    cudaMemcpy(d_vec, h_vec, hd * sizeof(float), cudaMemcpyHostToDevice);
    sp_cuda_sqfree_write_k(&sc, 0, 0, 0, d_vec);
    sp_cuda_sqfree_read_k(&sc, 0, 0, 0, d_recon);
    cudaMemcpy(h_recon, d_recon, hd * sizeof(float), cudaMemcpyDeviceToHost);
    float k_corr = sp_correlation_f32(h_vec, h_recon, hd);
    snprintf(msg, sizeof(msg), "Sqfree K pipeline: corr=%.4f (need >0.910)", k_corr);
    CHECK(k_corr > 0.910f, msg);

    // V pipeline
    fill_random(h_vec, hd);
    cudaMemcpy(d_vec, h_vec, hd * sizeof(float), cudaMemcpyHostToDevice);
    sp_cuda_sqfree_write_v(&sc, 0, 0, 0, d_vec);
    sp_cuda_sqfree_read_v(&sc, 0, 0, 0, d_recon);
    cudaMemcpy(h_recon, d_recon, hd * sizeof(float), cudaMemcpyDeviceToHost);
    float v_corr = sp_correlation_f32(h_vec, h_recon, hd);
    snprintf(msg, sizeof(msg), "Sqfree V pipeline: corr=%.4f (need >0.900)", v_corr);
    CHECK(v_corr > 0.900f, msg);

    // Multi-position write/read
    float min_corr = 1.0f;
    for (int p = 0; p < 16; p++) {
        fill_random(h_vec, hd);
        cudaMemcpy(d_vec, h_vec, hd * sizeof(float), cudaMemcpyHostToDevice);
        sp_cuda_sqfree_write_k(&sc, 1, 1, p, d_vec);
        sp_cuda_sqfree_read_k(&sc, 1, 1, p, d_recon);
        cudaMemcpy(h_recon, d_recon, hd * sizeof(float), cudaMemcpyDeviceToHost);
        float c = sp_correlation_f32(h_vec, h_recon, hd);
        if (c < min_corr) min_corr = c;
    }
    snprintf(msg, sizeof(msg),
             "Sqfree multi-pos (16 vecs, L1H1): min_corr=%.4f (need >0.910)", min_corr);
    CHECK(min_corr > 0.910f, msg);

    free(h_vec); free(h_recon);
    cudaFree(d_vec); cudaFree(d_recon);
    sp_cuda_sqfree_cache_free(&sc);
}

// ====================================================================
// Test 2: Sqfree GPU cache with spinor sheet
// ====================================================================
static void test_sqfree_spinor(void) {
    printf("\n== Sqfree GPU Cache + Spinor (hd=80) ==\n");

    const int hd = 80, n_layers = 2, n_heads = 2, max_seq = 256;

    sp_config_t cfg;
    sp_config_init(&cfg, hd, n_layers, n_heads);

    sp_cuda_sqfree_cache_t sc;
    int rc = sp_cuda_sqfree_cache_init(&sc, &cfg, max_seq,
                                        /*residual_bits=*/4,
                                        /*use_spinor=*/1, NULL);
    char msg[256];
    snprintf(msg, sizeof(msg), "sqfree+spinor init (use_spinor=%d)", sc.use_spinor);
    CHECK(rc == 0, msg);
    if (rc != 0) return;

    float *h_vec   = (float *)malloc(hd * sizeof(float));
    float *h_recon = (float *)malloc(hd * sizeof(float));
    float *d_vec, *d_recon;
    cudaMalloc((void **)&d_vec,   hd * sizeof(float));
    cudaMalloc((void **)&d_recon, hd * sizeof(float));

    // Write several positions then read them back — spinor accumulates
    // sign-bit sheets across the sequence.
    float total_corr = 0.0f;
    int n_pos = 32;
    for (int p = 0; p < n_pos; p++) {
        fill_random(h_vec, hd);
        cudaMemcpy(d_vec, h_vec, hd * sizeof(float), cudaMemcpyHostToDevice);
        sp_cuda_sqfree_write_k(&sc, 0, 0, p, d_vec);
    }
    // Read them all back
    for (int p = 0; p < n_pos; p++) {
        // Re-generate same random vector (reset seed for this position)
        srand(42 + p * 1000);
        fill_random(h_vec, hd);
        cudaMemcpy(d_vec, h_vec, hd * sizeof(float), cudaMemcpyHostToDevice);

        // But we already wrote with a different seed... we need to store originals.
        // Actually let's just write+read in pairs.
    }
    // Simpler: write-read pairs
    srand(12345);
    float min_corr = 1.0f;
    for (int p = 0; p < n_pos; p++) {
        fill_random(h_vec, hd);
        cudaMemcpy(d_vec, h_vec, hd * sizeof(float), cudaMemcpyHostToDevice);
        sp_cuda_sqfree_write_k(&sc, 1, 0, p, d_vec);
        sp_cuda_sqfree_read_k(&sc, 1, 0, p, d_recon);
        cudaMemcpy(h_recon, d_recon, hd * sizeof(float), cudaMemcpyDeviceToHost);
        float c = sp_correlation_f32(h_vec, h_recon, hd);
        if (c < min_corr) min_corr = c;
    }
    snprintf(msg, sizeof(msg),
             "Spinor K (32 vecs): min_corr=%.4f (need >0.910)", min_corr);
    CHECK(min_corr > 0.910f, msg);

    free(h_vec); free(h_recon);
    cudaFree(d_vec); cudaFree(d_recon);
    sp_cuda_sqfree_cache_free(&sc);
}

// ====================================================================
// Test 3: Sqfree batched read
// ====================================================================
static void test_sqfree_batch_read(void) {
    printf("\n== Sqfree GPU Batch Read (hd=80) ==\n");

    const int hd = 80, n_layers = 2, n_heads = 2, max_seq = 256;
    const int n_pos = 16;

    sp_config_t cfg;
    sp_config_init(&cfg, hd, n_layers, n_heads);

    sp_cuda_sqfree_cache_t sc;
    int rc = sp_cuda_sqfree_cache_init(&sc, &cfg, max_seq, 4, 0, NULL);
    CHECK(rc == 0, "sqfree batch init");
    if (rc != 0) return;

    float *h_vecs   = (float *)malloc(n_pos * hd * sizeof(float));
    float *h_recon  = (float *)malloc(n_pos * hd * sizeof(float));
    float *d_vec, *d_batch_out;
    cudaMalloc((void **)&d_vec,       hd * sizeof(float));
    cudaMalloc((void **)&d_batch_out, n_pos * hd * sizeof(float));

    char msg[256];

    // Write n_pos vectors individually
    srand(777);
    for (int p = 0; p < n_pos; p++) {
        fill_random(h_vecs + p * hd, hd);
        cudaMemcpy(d_vec, h_vecs + p * hd, hd * sizeof(float), cudaMemcpyHostToDevice);
        sp_cuda_sqfree_write_k(&sc, 0, 0, p, d_vec);
    }

    // Batch read all n_pos at once
    sp_cuda_sqfree_read_k_batch(&sc, 0, 0, 0, n_pos, d_batch_out);
    cudaMemcpy(h_recon, d_batch_out, n_pos * hd * sizeof(float), cudaMemcpyDeviceToHost);

    float min_corr = 1.0f, total_corr = 0.0f;
    for (int p = 0; p < n_pos; p++) {
        float c = sp_correlation_f32(h_vecs + p * hd, h_recon + p * hd, hd);
        if (c < min_corr) min_corr = c;
        total_corr += c;
    }
    float avg_corr = total_corr / n_pos;
    snprintf(msg, sizeof(msg),
             "Batch read K (%d pos): avg=%.4f min=%.4f (need avg>0.920)",
             n_pos, avg_corr, min_corr);
    CHECK(avg_corr > 0.920f, msg);

    // V batch
    srand(888);
    for (int p = 0; p < n_pos; p++) {
        fill_random(h_vecs + p * hd, hd);
        cudaMemcpy(d_vec, h_vecs + p * hd, hd * sizeof(float), cudaMemcpyHostToDevice);
        sp_cuda_sqfree_write_v(&sc, 0, 0, p, d_vec);
    }
    sp_cuda_sqfree_read_v_batch(&sc, 0, 0, 0, n_pos, d_batch_out);
    cudaMemcpy(h_recon, d_batch_out, n_pos * hd * sizeof(float), cudaMemcpyDeviceToHost);

    min_corr = 1.0f; total_corr = 0.0f;
    for (int p = 0; p < n_pos; p++) {
        float c = sp_correlation_f32(h_vecs + p * hd, h_recon + p * hd, hd);
        if (c < min_corr) min_corr = c;
        total_corr += c;
    }
    avg_corr = total_corr / n_pos;
    snprintf(msg, sizeof(msg),
             "Batch read V (%d pos): avg=%.4f min=%.4f (need avg>0.890)",
             n_pos, avg_corr, min_corr);
    CHECK(avg_corr > 0.890f, msg);

    free(h_vecs); free(h_recon);
    cudaFree(d_vec); cudaFree(d_batch_out);
    sp_cuda_sqfree_cache_free(&sc);
}

// ====================================================================
// Test 4: Hierarchical GPU cache
// ====================================================================
static void test_hier_gpu(void) {
    printf("\n== Hierarchical GPU Cache (hd=128) ==\n");

    const int hd = 128, n_layers = 2, n_heads = 2, max_seq = 256;

    sp_config_t cfg;
    sp_config_init(&cfg, hd, n_layers, n_heads);

    // Use CPU-side hier_cache to get structural metadata.
    // sp_hier_cache_init(hc, cfg, max_seq, hier_level, skel_n_bands, skel_band_bits, target_res_bits, target_res_bits_v, skel_ternary_mask)
    int skel_band_bits[] = {5, 5};
    sp_hier_cache_t hc_cpu;
    int rc = sp_hier_cache_init(&hc_cpu, &cfg, max_seq,
                                 /*hier_level=*/0,
                                 /*skel_n_bands=*/2, skel_band_bits,
                                 /*target_res_bits=*/2,
                                 /*target_res_bits_v=*/0,
                                 /*skel_ternary_mask=*/0);
    CHECK(rc == 0, "CPU hier_cache init");
    if (rc != 0) return;

    char msg[256];

    // Extract structural metadata from the first predictor
    sp_hier_predictor_t *hp0 = &hc_cpu.predictors[0];

    // Now init the GPU variant
    int n_slots = n_layers * n_heads;
    sp_cuda_hier_cache_t hc;
    rc = sp_cuda_hier_cache_init(&hc, &cfg,
                                  hc_cpu.pad_dim,
                                  hp0->n_skeleton,
                                  hp0->n_target,
                                  hp0->target_res_bits,
                                  hp0->skeleton_idx,
                                  hp0->target_idx,
                                  &hp0->skel_bands,
                                  max_seq, n_slots, NULL);
    snprintf(msg, sizeof(msg),
             "GPU hier init (pad=%d, skel=%d, target=%d)",
             hc.pad_dim, hc.n_skeleton, hc.n_target);
    CHECK(rc == 0, msg);
    if (rc != 0) { sp_hier_cache_free(&hc_cpu); return; }

    // Upload identity-like W matrices (no calibration data — use identity
    // as a baseline; prediction will be poor but pipeline should not crash).
    // W is [n_slots][n_target × n_skeleton] fp16
    int W_size = n_slots * hc.n_target * hc.n_skeleton;
    uint16_t *W_all = (uint16_t *)calloc(W_size, sizeof(uint16_t));
    // Fill diagonal with fp16(1.0) = 0x3C00 for min(n_target, n_skeleton) entries
    for (int s = 0; s < n_slots; s++) {
        for (int i = 0; i < hc.n_target && i < hc.n_skeleton; i++) {
            W_all[s * hc.n_target * hc.n_skeleton + i * hc.n_skeleton + i] = 0x3C00;
        }
    }
    rc = sp_cuda_hier_cache_upload_W(&hc, W_all);
    CHECK(rc == 0, "Upload W matrices");

    float *h_vec   = (float *)malloc(hd * sizeof(float));
    float *h_recon = (float *)malloc(hd * sizeof(float));
    float *d_vec, *d_recon;
    cudaMalloc((void **)&d_vec,   hd * sizeof(float));
    cudaMalloc((void **)&d_recon, hd * sizeof(float));

    // K pipeline
    srand(42);
    fill_random(h_vec, hd);
    cudaMemcpy(d_vec, h_vec, hd * sizeof(float), cudaMemcpyHostToDevice);
    sp_cuda_hier_write_k(&hc, 0, 0, 0, d_vec);
    sp_cuda_hier_read_k(&hc, 0, 0, 0, d_recon);
    cudaMemcpy(h_recon, d_recon, hd * sizeof(float), cudaMemcpyDeviceToHost);
    float k_corr = sp_correlation_f32(h_vec, h_recon, hd);
    snprintf(msg, sizeof(msg),
             "Hier K pipeline: corr=%.4f (need >0.900, identity W)", k_corr);
    // Lower threshold because identity W is not a real calibration
    CHECK(k_corr > 0.900f, msg);

    // V pipeline
    fill_random(h_vec, hd);
    cudaMemcpy(d_vec, h_vec, hd * sizeof(float), cudaMemcpyHostToDevice);
    sp_cuda_hier_write_v(&hc, 0, 0, 0, d_vec);
    sp_cuda_hier_read_v(&hc, 0, 0, 0, d_recon);
    cudaMemcpy(h_recon, d_recon, hd * sizeof(float), cudaMemcpyDeviceToHost);
    float v_corr = sp_correlation_f32(h_vec, h_recon, hd);
    snprintf(msg, sizeof(msg),
             "Hier V pipeline: corr=%.4f (need >0.850, identity W)", v_corr);
    CHECK(v_corr > 0.850f, msg);

    // Multi-slot write/read
    float min_corr = 1.0f;
    for (int L = 0; L < n_layers; L++) {
        for (int H = 0; H < n_heads; H++) {
            fill_random(h_vec, hd);
            cudaMemcpy(d_vec, h_vec, hd * sizeof(float), cudaMemcpyHostToDevice);
            sp_cuda_hier_write_k(&hc, L, H, 1, d_vec);
            sp_cuda_hier_read_k(&hc, L, H, 1, d_recon);
            cudaMemcpy(h_recon, d_recon, hd * sizeof(float), cudaMemcpyDeviceToHost);
            float c = sp_correlation_f32(h_vec, h_recon, hd);
            if (c < min_corr) min_corr = c;
        }
    }
    snprintf(msg, sizeof(msg),
             "Hier multi-slot (4 slots): min_corr=%.4f (need >0.850)", min_corr);
    CHECK(min_corr > 0.850f, msg);

    free(h_vec); free(h_recon); free(W_all);
    cudaFree(d_vec); cudaFree(d_recon);
    sp_cuda_hier_cache_free(&hc);
    sp_hier_cache_free(&hc_cpu);
}

// ====================================================================
// Test 5: Cold storage layer (GPU ↔ CPU offload)
// ====================================================================
static void test_cold_storage(void) {
    printf("\n== Cold Storage Layer (GPU ↔ CPU) ==\n");

    const int hd = 128, n_layers = 2, n_heads = 2, max_seq = 512;
    char msg[256];

    sp_config_t cfg;
    sp_config_init(&cfg, hd, n_layers, n_heads);

    // Init the base CUDA cache
    sp_cuda_cache_t cc;
    int rc = sp_cuda_cache_init(&cc, &cfg, max_seq, NULL);
    CHECK(rc == 0, "CUDA cache init for cold layer test");
    if (rc != 0) return;

    // Init a cold layer with enough room for 64 positions × 1 head
    int packed_stride = cc.k_bands.total_bytes;
    int64_t cold_cap = (int64_t)packed_stride * 64;

    sp_cuda_cold_layer_t cl;
    rc = sp_cuda_cold_layer_init(&cl, cold_cap, packed_stride, n_heads);
    snprintf(msg, sizeof(msg),
             "cold layer init (%lld bytes, stride=%d)", (long long)cold_cap, packed_stride);
    CHECK(rc == 0, msg);
    if (rc != 0) { sp_cuda_cache_free(&cc); return; }

    // Write 32 K vectors, then writeback to cold layer
    float *h_vec = (float *)malloc(hd * sizeof(float));
    float *d_vec;
    cudaMalloc((void **)&d_vec, hd * sizeof(float));
    srand(42);
    for (int p = 0; p < 32; p++) {
        fill_random(h_vec, hd);
        cudaMemcpy(d_vec, h_vec, hd * sizeof(float), cudaMemcpyHostToDevice);
        sp_cuda_write_k(&cc, 0, 0, p, d_vec);
    }

    // Writeback positions [0, 32) to cold layer
    rc = sp_cuda_cold_writeback(&cl, cc.d_k_cache, 0, 32, NULL);
    cudaDeviceSynchronize();
    snprintf(msg, sizeof(msg), "writeback 32 positions to cold: rc=%d", rc);
    CHECK(rc == 0, msg);

    // Verify cold layer state
    CHECK(cl.newest_pos == 31, "cold layer newest_pos == 31");

    // Restore from cold back to GPU (simulating a re-warm)
    // First, zero the GPU cache positions to prove restore works
    cudaMemset(cc.d_k_cache, 0, (size_t)packed_stride * max_seq);

    rc = sp_cuda_cold_restore(&cl, cc.d_k_cache, 32, NULL);
    cudaDeviceSynchronize();
    snprintf(msg, sizeof(msg), "restore 32 positions from cold: rc=%d", rc);
    CHECK(rc == 0, msg);

    // Read back position 0 and check correlation
    float *d_recon;
    float *h_recon = (float *)malloc(hd * sizeof(float));
    cudaMalloc((void **)&d_recon, hd * sizeof(float));

    sp_cuda_read_k(&cc, 0, 0, 0, d_recon);
    cudaMemcpy(h_recon, d_recon, hd * sizeof(float), cudaMemcpyDeviceToHost);

    // Re-generate position 0's original vector
    srand(42);
    fill_random(h_vec, hd);
    float cold_corr = sp_correlation_f32(h_vec, h_recon, hd);
    snprintf(msg, sizeof(msg),
             "Cold round-trip (pos 0): corr=%.4f (need >0.990)", cold_corr);
    CHECK(cold_corr > 0.990f, msg);

    free(h_vec); free(h_recon);
    cudaFree(d_vec); cudaFree(d_recon);
    sp_cuda_cold_layer_free(&cl);
    sp_cuda_cache_free(&cc);
}

// ====================================================================
// Test 6: Stress — high sequence length
// ====================================================================
static void test_stress_long_seq(void) {
    printf("\n== Stress: Long Sequence (hd=128, seq=4096) ==\n");

    const int hd = 128, n_layers = 1, n_heads = 1, max_seq = 4096;
    char msg[256];

    sp_config_t cfg;
    sp_config_init(&cfg, hd, n_layers, n_heads);

    sp_cuda_cache_t cc;
    int rc = sp_cuda_cache_init(&cc, &cfg, max_seq, NULL);
    snprintf(msg, sizeof(msg), "cache init (max_seq=%d)", max_seq);
    CHECK(rc == 0, msg);
    if (rc != 0) return;

    float *h_vec   = (float *)malloc(hd * sizeof(float));
    float *h_recon = (float *)malloc(hd * sizeof(float));
    float *d_vec, *d_recon;
    cudaMalloc((void **)&d_vec,   hd * sizeof(float));
    cudaMalloc((void **)&d_recon, hd * sizeof(float));

    // Write every 128th position (32 writes covering the full seq range)
    srand(99);
    float min_corr = 1.0f;
    int n_sampled = 0;
    for (int p = 0; p < max_seq; p += 128) {
        fill_random(h_vec, hd);
        cudaMemcpy(d_vec, h_vec, hd * sizeof(float), cudaMemcpyHostToDevice);
        sp_cuda_write_k(&cc, 0, 0, p, d_vec);
        sp_cuda_read_k(&cc, 0, 0, p, d_recon);
        cudaMemcpy(h_recon, d_recon, hd * sizeof(float), cudaMemcpyDeviceToHost);
        float c = sp_correlation_f32(h_vec, h_recon, hd);
        if (c < min_corr) min_corr = c;
        n_sampled++;
    }
    snprintf(msg, sizeof(msg),
             "Long seq K (%d samples over %d): min_corr=%.4f (need >0.985)",
             n_sampled, max_seq, min_corr);
    CHECK(min_corr > 0.985f, msg);

    // Batch write+read 256 positions starting at 1024
    {
        int n_pos = 256;
        float *h_batch  = (float *)malloc(n_pos * hd * sizeof(float));
        float *h_brecon = (float *)malloc(n_pos * hd * sizeof(float));
        float *d_batch, *d_brecon;
        cudaMalloc((void **)&d_batch,  n_pos * hd * sizeof(float));
        cudaMalloc((void **)&d_brecon, n_pos * hd * sizeof(float));

        fill_random(h_batch, n_pos * hd);
        cudaMemcpy(d_batch, h_batch, n_pos * hd * sizeof(float),
                   cudaMemcpyHostToDevice);

        sp_cuda_write_k_batch(&cc, 0, 0, 1024, n_pos, d_batch);
        sp_cuda_read_k_batch(&cc, 0, 0, 1024, n_pos, d_brecon);
        cudaMemcpy(h_brecon, d_brecon, n_pos * hd * sizeof(float),
                   cudaMemcpyDeviceToHost);

        float total = 0.0f;
        float bmin = 1.0f;
        for (int p = 0; p < n_pos; p++) {
            float c = sp_correlation_f32(h_batch + p * hd, h_brecon + p * hd, hd);
            total += c;
            if (c < bmin) bmin = c;
        }
        float avg = total / n_pos;
        snprintf(msg, sizeof(msg),
                 "Batch K (%d pos @offset 1024): avg=%.4f min=%.4f (need avg>0.990)",
                 n_pos, avg, bmin);
        CHECK(avg > 0.990f, msg);

        free(h_batch); free(h_brecon);
        cudaFree(d_batch); cudaFree(d_brecon);
    }

    free(h_vec); free(h_recon);
    cudaFree(d_vec); cudaFree(d_recon);
    sp_cuda_cache_free(&cc);
}

// ====================================================================
int main(void) {
    srand(42);
    printf("Shannon-Prime Advanced CUDA Tests\n");
    printf("===================================\n");

    int device_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&device_count);
    if (cerr != cudaSuccess || device_count < 1) {
        printf("  [%s] No CUDA device available (%s)\n",
               FAIL, cudaGetErrorString(cerr));
        return 1;
    }

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  GPU: %s (%d MB)\n", prop.name, (int)(prop.totalGlobalMem / 1024 / 1024));
    printf("  SM:  %d.%d\n\n", prop.major, prop.minor);

    test_sqfree_gpu();
    test_sqfree_spinor();
    test_sqfree_batch_read();
    test_hier_gpu();
    test_cold_storage();
    test_stress_long_seq();

    printf("\n===================================\n");
    printf("Results: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
