/*
 * Smoke test for sp_llama_qnn — exercises the cache + dispatch API
 * exactly as shannon-prime-llama would call it from llama_sp_fused_kq.cpp.
 *
 * Simulates a typical Qwen3-4B attention pattern across multiple "tokens":
 *   - n_head_kv heads, each with head_dim=128
 *   - n_kv = 2048 context window
 *   - M_q = 64 (n_seq * n_head_q after layout)
 *   - 10 simulated decode steps (each calls dispatch n_head_kv times)
 *
 * The cache should produce ONE graph (same shape for every head), all
 * subsequent dispatches reuse it with K rebound per token.
 */
#include "sp_llama_qnn.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static uint64_t now_us(void) {
    struct timeval t; gettimeofday(&t, NULL);
    return (uint64_t)t.tv_sec * 1000000ULL + (uint64_t)t.tv_usec;
}

int main(int argc, char **argv) {
    int n_tokens = (argc > 1) ? atoi(argv[1]) : 10;
    int n_heads  = (argc > 2) ? atoi(argv[2]) : 8;
    uint32_t M_q   = 64;
    uint32_t K_dim = 128;
    uint32_t N_kv  = 2048;

    fprintf(stderr, "=== sp_llama_qnn smoke test ===\n");
    fprintf(stderr, "  Simulating %d tokens x %d heads at Q[%u,%u] K[%u,%u] -> attn[%u,%u] fp16\n",
            n_tokens, n_heads, M_q, K_dim, N_kv, K_dim, M_q, N_kv);

    sp_llama_qnn_kq_cache *cache = sp_llama_qnn_kq_cache_create();
    if (!cache) { fprintf(stderr, "cache create failed\n"); return 1; }

    /* Allocate per-head Q (rebound every call) and per-token K (rebound on
     * token boundary — same K used by all heads of one token). */
    size_t q_bytes  = (size_t)M_q  * K_dim * 2;
    size_t k_bytes  = (size_t)N_kv * K_dim * 2;
    size_t out_bytes = (size_t)M_q * N_kv * 2;
    void *q   = calloc(1, q_bytes);
    void *out = calloc(1, out_bytes);

    /* Per-token K buffers — fresh allocation each token to simulate the
     * "K changed" path that triggers rebind. */
    uint64_t total_us = 0;
    int total_calls = 0;
    for (int tok = 0; tok < n_tokens; ++tok) {
        void *k_token = calloc(1, k_bytes);   /* fresh K each token */

        for (int head = 0; head < n_heads; ++head) {
            uint64_t us = 0;
            int rc = sp_llama_qnn_kq_dispatch(cache, M_q, K_dim, N_kv,
                                              q, q_bytes,
                                              k_token, k_bytes,
                                              out, out_bytes,
                                              &us);
            if (rc != 0) {
                fprintf(stderr, "  tok=%d head=%d FAILED rc=%d\n", tok, head, rc);
                goto cleanup;
            }
            total_us += us;
            total_calls++;
            if (tok < 2 || head == 0) {
                fprintf(stderr, "  tok=%d head=%d: %" PRIu64 " us%s\n",
                        tok, head, us,
                        (tok==0 && head==0) ? "  (first call: graphFinalize cost included)" : "");
            }
        }
        free(k_token);
    }

    fprintf(stderr, "\n=== summary ===\n");
    fprintf(stderr, "  total calls:    %d\n", total_calls);
    fprintf(stderr, "  total exec us:  %" PRIu64 "\n", total_us);
    fprintf(stderr, "  avg per call:   %" PRIu64 " us\n", total_us / total_calls);
    fprintf(stderr, "  per-token cost: %" PRIu64 " us (%d heads sequential)\n",
            (total_us / n_tokens), n_heads);

cleanup:
    free(q); free(out);
    sp_llama_qnn_kq_cache_destroy(&cache);
    return 0;
}
