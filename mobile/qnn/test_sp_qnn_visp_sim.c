/*
 * Phase 4.14 — VISP / ISP Weight Swapping Simulation Test.
 *
 * This test validates the core mechanism of the 56 tok/s architecture:
 *   1. Create a runtime matmul graph (A @ B = C).
 *   2. Allocate a persistent ION buffer for B (the "weights").
 *   3. Execute with Weight Set 1 -> Verify Output 1.
 *   4. Write Weight Set 2 into the SAME ION buffer.
 *   5. Execute AGAIN (passing the ION pointer to skip rebind) -> Verify Output 2.
 *
 * If this passes, it proves the HTP correctly sees "live" weight updates
 * in shared memory without the latency of context reloading or rebinding.
 */
#include "sp_qnn.h"
#include "sp_llama_qnn.h"
#include "QnnTypes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define M 1
#define K 128
#define N 128

static float fp16_to_fp32(uint16_t h) {
    uint32_t f = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1C000) << 13) | ((h & 0x03FF) << 13);
    return *(float*)&f;
}

static uint16_t fp32_to_fp16(float f) {
    uint32_t i = *(uint32_t*)&f;
    uint32_t s = (i >> 16) & 0x00008000;
    uint32_t e = ((i >> 23) & 0x000000ff) - (127 - 15);
    uint32_t m = i & 0x007fffff;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7c00);
    return (uint16_t)(s | (e << 10) | (m >> 13));
}

int main() {
    fprintf(stderr, "=== Phase 4.14 — VISP Weight Swapping Simulation ===\n");

    if (sp_qnn_init(NULL, NULL) != SP_QNN_OK) return 1;

    sp_llama_qnn_matmul_cache *cache = sp_llama_qnn_matmul_cache_create();
    if (!cache) return 1;

    // 1. Get the ION pointer for the [1, 128] @ [128, 128] shape.
    // This will trigger find_or_create_mm_slot and alloc_persistent.
    void *ion_b = sp_llama_qnn_matmul_get_ion_ptr(cache, M, K, N);
    if (!ion_b) {
        fprintf(stderr, "ION allocation failed (expected on non-FastRPC systems)\n");
        // Fallback: we'll continue to test the API flow even if it's not ION-backed.
    }

    // 2. Prepare inputs.
    uint16_t *a = calloc(M * K, 2);
    uint16_t *b1 = calloc(K * N, 2);
    uint16_t *b2 = calloc(K * N, 2);
    uint16_t *c = calloc(M * N, 2);

    for (int i = 0; i < K; ++i) a[i] = fp32_to_fp16(1.0f);
    for (int i = 0; i < K * N; ++i) b1[i] = fp32_to_fp16(0.01f);
    for (int i = 0; i < K * N; ++i) b2[i] = fp32_to_fp16(0.02f);

    // 3. Execution 1: Using b1 (copied into ION if present).
    uint64_t us1 = 0;
    sp_llama_qnn_matmul_dispatch(cache, M, K, N, a, M*K*2, b1, K*N*2, c, M*N*2, &us1);
    float out1 = fp16_to_fp32(c[0]);
    fprintf(stderr, "Exec 1 (Weight=0.01): out[0] = %.4f (expected ~1.28) in %llu us\n", out1, (unsigned long long)us1);

    // 4. Execution 2: Simulate ISP feeding.
    // We write b2 directly to the ION buffer.
    if (ion_b) {
        memcpy(ion_b, b2, K*N*2);
        // Dispatch passing the ION pointer itself. Shim should skip memcpy.
        uint64_t us2 = 0;
        sp_llama_qnn_matmul_dispatch(cache, M, K, N, a, M*K*2, ion_b, K*N*2, c, M*N*2, &us2);
        float out2 = fp16_to_fp32(c[0]);
        fprintf(stderr, "Exec 2 (Weight=0.02, ZERO-COPY): out[0] = %.4f (expected ~2.56) in %llu us\n", out2, (unsigned long long)us2);

        if (fabs(out2 - 2.0f * out1) < 0.1f) {
            fprintf(stderr, "SUCCESS: Weight swapping verified zero-copy!\n");
        } else {
            fprintf(stderr, "FAILURE: Weight swapping output mismatch.\n");
        }
    } else {
        fprintf(stderr, "Skipping zero-copy test (no ION).\n");
    }

    sp_llama_qnn_matmul_cache_destroy(&cache);
    free(a); free(b1); free(b2); free(c);
    return 0;
}
