/*
 * Phase 2.5+ — multi-op runtime graph (KQ + Softmax) with persistent K weight.
 *
 * Demonstrates three things at once:
 *   (1) Multi-op runtime graph: MatMul -> Softmax in a single graphFinalize,
 *       single graphExecute. No .bin, no AOT compile.
 *   (2) Persistent input via QnnMem_register: K bound once via memhandle,
 *       Q rebound per call. Measures the rebind savings.
 *   (3) sp_llama_qnn_kq_dispatch() — the entry point shape that
 *       shannon-prime-llama's FUSED_KQ slot would call.
 *
 * Compares two regimes:
 *   - K rebound per call (existing APP_WRITE clientBuf path)
 *   - K registered persistent via memhandle (Q-only rebind per call)
 */
#include "sp_qnn.h"
#include "QnnTypes.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static uint64_t now_us(void) {
    struct timeval t; gettimeofday(&t, NULL);
    return (uint64_t)t.tv_sec * 1000000ULL + (uint64_t)t.tv_usec;
}

/* sp_llama-shape dispatch wrapper. This is the C entry point that
 * shannon-prime-llama's existing FUSED_KQ slot would call into. The
 * llama.cpp side has the existing sp_llama_kq_matmul_fused signature;
 * this is the QNN-runtime variant.
 *
 * Lazy-creates the graph on first call for this (M_q, K_dim, N_kv) shape.
 * Registers K once (so subsequent Q's reuse the same K binding), then
 * executes per-Q. Returns the attention weights into kq_out. */
typedef struct {
    sp_qnn_handle *h;
    uint32_t M_q, K_dim, N_kv;
    int      k_bound;
} sp_llama_qnn_kq_state;

static sp_qnn_status sp_llama_qnn_kq_dispatch(sp_llama_qnn_kq_state *state,
                                              const void *q_data,  size_t q_bytes,
                                              void *k_data,        size_t k_bytes,
                                              void *out_kq,        size_t out_bytes,
                                              uint64_t *out_us) {
    if (!state || !state->h) return SP_QNN_ERR_INVALID;

    /* On the first call, register K as persistent. Subsequent calls
     * leave K bound (the model's KV cache can grow but we'd typically
     * re-register at the per-token boundary anyway). */
    if (!state->k_bound) {
        sp_qnn_register_persistent_input(state->h, /*tensor_idx=*/1,
                                          k_data, k_bytes);
        state->k_bound = 1;
    }

    /* Per-call: bind Q (input 0) only; K is already bound via memhandle. */
    const void *ins[]  = { q_data, NULL };           /* NULL = "use prior binding" */
    const size_t in_sz[] = { q_bytes, k_bytes };
    void *outs[] = { out_kq };
    const size_t out_sz[] = { out_bytes };
    return sp_qnn_execute(state->h, ins, in_sz, outs, out_sz, out_us);
}

int main(int argc, char **argv) {
    /* Defaults match the Phase 2.2 attention shape we measured: M_q=64
     * (sequence × head bound), K_dim=128 (head_dim for Qwen3-4B), N_kv=2048
     * (full 4K context window's key tokens). */
    uint32_t M_q   = (argc > 1) ? (uint32_t)atoi(argv[1]) : 64;
    uint32_t K_dim = (argc > 2) ? (uint32_t)atoi(argv[2]) : 128;
    uint32_t N_kv  = (argc > 3) ? (uint32_t)atoi(argv[3]) : 2048;
    int n_iter     = (argc > 4) ? atoi(argv[4]) : 50;

    fprintf(stderr, "=== Phase 2.5+ multi-op KQ+Softmax with persistent K ===\n");
    fprintf(stderr, "  Q[%u,%u] @ K^T[%u,%u] -> Softmax -> attn[%u,%u]\n",
            M_q, K_dim, N_kv, K_dim, M_q, N_kv);
    fprintf(stderr, "  fp16, %d iters\n\n", n_iter);

    if (sp_qnn_init(NULL, NULL) != SP_QNN_OK) return 1;

    /* === REGIME A: Q + K both APP_WRITE (rebound per call) === */
    sp_qnn_handle *h_a = NULL;
    if (sp_qnn_runtime_kq_softmax_create(M_q, K_dim, N_kv,
                                          QNN_DATATYPE_FLOAT_16, &h_a) != SP_QNN_OK) {
        fprintf(stderr, "create regime A failed\n"); return 2;
    }

    size_t q_bytes = (size_t)M_q  * K_dim * 2;
    size_t k_bytes = (size_t)N_kv * K_dim * 2;
    size_t out_bytes = (size_t)M_q * N_kv * 2;
    void *q   = calloc(1, q_bytes);
    void *k   = calloc(1, k_bytes);
    void *out = calloc(1, out_bytes);

    fprintf(stderr, "=== REGIME A: rebind Q+K every call (current default path) ===\n");
    {
        sp_qnn_bench_result br = {0};
        const void *ins[] = { q, k };
        const size_t in_sz[] = { q_bytes, k_bytes };
        void *outs[] = { out };
        const size_t out_sz[] = { out_bytes };
        if (sp_qnn_bench(h_a, ins, in_sz, outs, out_sz,
                          (uint32_t)n_iter, &br) != SP_QNN_OK) {
            fprintf(stderr, "regime A bench failed\n");
        } else {
            fprintf(stderr, "  min %" PRIu64 " us  avg %" PRIu64 " us  max %" PRIu64 " us\n",
                    br.min_us, br.avg_us, br.max_us);
        }
    }
    sp_qnn_destroy(&h_a);

    /* === REGIME B: K persistent via memhandle, only Q rebound per call === */
    sp_qnn_handle *h_b = NULL;
    if (sp_qnn_runtime_kq_softmax_create(M_q, K_dim, N_kv,
                                          QNN_DATATYPE_FLOAT_16, &h_b) != SP_QNN_OK) {
        fprintf(stderr, "create regime B failed\n"); return 3;
    }

    fprintf(stderr, "\n=== REGIME B: K registered persistent (Q-only per-call rebind) ===\n");
    {
        sp_llama_qnn_kq_state state = { .h = h_b, .M_q = M_q,
                                         .K_dim = K_dim, .N_kv = N_kv };
        uint64_t mn = UINT64_MAX, mx = 0, sum = 0;
        for (int i = 0; i < n_iter; ++i) {
            uint64_t us = 0;
            sp_qnn_status rc = sp_llama_qnn_kq_dispatch(&state, q, q_bytes, k, k_bytes,
                                                        out, out_bytes, &us);
            if (rc != SP_QNN_OK) { fprintf(stderr, "iter %d FAILED %d\n", i, rc); break; }
            if (us < mn) mn = us;
            if (us > mx) mx = us;
            sum += us;
        }
        fprintf(stderr, "  min %" PRIu64 " us  avg %" PRIu64 " us  max %" PRIu64 " us\n",
                mn, sum / (uint64_t)n_iter, mx);
    }
    sp_qnn_destroy(&h_b);

    free(q); free(k); free(out);
    sp_qnn_shutdown();
    return 0;
}
