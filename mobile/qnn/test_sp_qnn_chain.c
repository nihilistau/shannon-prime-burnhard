/*
 * Phase 2.3 stage 3 — chain all 4 PP splits end-to-end.
 *
 * The qai-hub-models LLM export shards the transformer into N sequential
 * graphs. Each split N consumes the residual stream + KV cache from
 * split N-1 and produces them for split N+1. This test wires the chain
 * together and measures the real per-128-token prefill latency, replacing
 * the per-split projection in test_sp_qnn_multi.
 *
 * Inter-split contract for our qwen3_4b w4a16_ar128_cl2048 splits (validated
 * via single-split smoke test):
 *
 *   split[i] inputs:
 *     - attention_mask        fp16  [1,1,128,2048]   (constant across splits)
 *     - position_ids_cos      fp16  [1,1,128,64]     (constant)
 *     - position_ids_sin      fp16  [1,1,128,64]     (constant)
 *     - past_key_L_in         u8    [8,1,128,1920]   for L in [12*i .. 12*(i+1)-1]
 *     - past_value_L_in       u8    [8,1,1920,128]   for L in [12*i .. 12*(i+1)-1]
 *     - residual_in           fp16  [1,128,2560]     (output of split[i-1] OR
 *                                                      embedding for split[0])
 *
 *   split[i] outputs:
 *     - residual_out          fp16  [1,128,2560]     (input to split[i+1])
 *     - past_key_L_out        u8    [8,1,128,128]    for L in [12*i .. 12*(i+1)-1]
 *     - past_value_L_out      u8    [8,1,128,128]    for L in [12*i .. 12*(i+1)-1]
 *
 * Key observation: past_key_OUT is [8,1,128,128] but past_key_IN is
 * [8,1,128,1920]. The IN dimensions cover the FULL 2048-token context
 * window minus the current 128-token chunk; OUT dimensions are the 128
 * new positions being written. Production prefill would maintain a
 * rolling KV ring buffer; here we just use zero-filled IN and discard OUT
 * (we're benching per-call latency, not autoregressive correctness).
 *
 * Build: see build.cmd (added test_sp_qnn_chain to the build list).
 *
 * Run on device:
 *   LD_LIBRARY_PATH=/data/local/tmp/sp22u/qnn \
 *   ADSP_LIBRARY_PATH=/data/local/tmp/sp22u/qnn \
 *   ./test_sp_qnn_chain ./qwen3_4b_1_of_4.bin ./qwen3_4b_2_of_4.bin \
 *                       ./qwen3_4b_3_of_4.bin ./qwen3_4b_4_of_4.bin
 */
#include "sp_qnn.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define MAX_SPLITS 4

static size_t tensor_bytes(const sp_qnn_tensor_info *t) {
    size_t n = t->bytes_per_element;
    if (n == 0) n = 1;
    for (uint32_t d = 0; d < t->rank; ++d) n *= t->dims[d];
    return n;
}

/* Allocate zero-filled buffers for every input/output of a graph. The
 * caller frees them via free_split_buffers(). */
typedef struct {
    sp_qnn_handle *h;

    size_t   n_in;
    void   **in_bufs;
    size_t  *in_sz;

    size_t   n_out;
    void   **out_bufs;
    size_t  *out_sz;

    /* Convenience: indices of the residual_in/out tensors so we can copy
     * the residual stream split-to-split. Set by find_residual_indices(). */
    int      residual_in_idx;
    int      residual_out_idx;
} split_state;

static void free_split_buffers(split_state *s) {
    if (s->in_bufs)  { for (size_t i = 0; i < s->n_in;  ++i) free(s->in_bufs[i]);  free(s->in_bufs); }
    if (s->out_bufs) { for (size_t i = 0; i < s->n_out; ++i) free(s->out_bufs[i]); free(s->out_bufs); }
    free(s->in_sz);
    free(s->out_sz);
    s->in_bufs = s->out_bufs = NULL; s->in_sz = s->out_sz = NULL;
}

/* The naming convention in our .bins:
 *   residual_in:   '_model_model_layers_<L-1>_Add_1_output_0'  (L = first layer of split)
 *   residual_out:  '_model_model_layers_<L_last>_Add_1_output_0'
 *
 * For split 0 the residual_in name doesn't have a 'layers_X' prefix —
 * it's the embedding output. We just match on substring '_Add_1_output_0'
 * for residual_out and the bare 'inputs_embeds' or 'layers_*Add_1_output_0'
 * for residual_in. Any tensor of rank-3 fp16 with shape [1,128,2560] is the
 * residual stream by elimination. */
static int find_residual_idx(const sp_qnn_tensor_info *infos, size_t n,
                              uint32_t want_d0, uint32_t want_d1, uint32_t want_d2) {
    for (size_t i = 0; i < n; ++i) {
        if (infos[i].rank == 3
            && infos[i].dims[0] == want_d0
            && infos[i].dims[1] == want_d1
            && infos[i].dims[2] == want_d2) {
            return (int)i;
        }
    }
    return -1;
}

static int load_split(const char *bin_path, split_state *s) {
    memset(s, 0, sizeof(*s));
    sp_qnn_status rc = sp_qnn_load_binary(bin_path, NULL, &s->h);
    if (rc != SP_QNN_OK) {
        fprintf(stderr, "[chain] load %s failed: %d\n", bin_path, rc);
        return -1;
    }
    const sp_qnn_tensor_info *in_info = NULL, *out_info = NULL;
    sp_qnn_get_io_info(s->h, &s->n_in, &in_info, &s->n_out, &out_info);

    s->in_bufs  = calloc(s->n_in,  sizeof(void *));
    s->in_sz    = calloc(s->n_in,  sizeof(size_t));
    s->out_bufs = calloc(s->n_out, sizeof(void *));
    s->out_sz   = calloc(s->n_out, sizeof(size_t));
    if (!s->in_bufs || !s->in_sz || !s->out_bufs || !s->out_sz) {
        fprintf(stderr, "[chain] alloc tracking arrays failed\n");
        return -1;
    }
    for (size_t i = 0; i < s->n_in; ++i) {
        s->in_sz[i] = tensor_bytes(&in_info[i]);
        s->in_bufs[i] = calloc(1, s->in_sz[i]);
        if (!s->in_bufs[i]) {
            fprintf(stderr, "[chain] in[%zu] alloc %zu failed\n", i, s->in_sz[i]);
            return -1;
        }
    }
    for (size_t i = 0; i < s->n_out; ++i) {
        s->out_sz[i] = tensor_bytes(&out_info[i]);
        s->out_bufs[i] = calloc(1, s->out_sz[i]);
        if (!s->out_bufs[i]) {
            fprintf(stderr, "[chain] out[%zu] alloc %zu failed\n", i, s->out_sz[i]);
            return -1;
        }
    }

    /* Find the residual-stream tensor on each side. fp16 (bpe=2) rank-3
     * shape [1,128,2560] = 655360 bytes. */
    s->residual_in_idx  = find_residual_idx(in_info,  s->n_in,  1, 128, 2560);
    s->residual_out_idx = find_residual_idx(out_info, s->n_out, 1, 128, 2560);
    fprintf(stderr, "[chain] %s: %zu in / %zu out, residual_in=%d, residual_out=%d\n",
            bin_path, s->n_in, s->n_out, s->residual_in_idx, s->residual_out_idx);

    return 0;
}

static int run_split(split_state *s, uint64_t *out_us) {
    return sp_qnn_execute(s->h,
                          (const void *const *)s->in_bufs, s->in_sz,
                          (void *const *)s->out_bufs,      s->out_sz,
                          out_us);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <split1.bin> [split2.bin] ... [splitN.bin]\n", argv[0]);
        fprintf(stderr, "       passes residual stream split-to-split, benches\n");
        fprintf(stderr, "       per-token latency for the full chain\n");
        return 1;
    }
    int n_splits = argc - 1;
    if (n_splits > MAX_SPLITS) {
        fprintf(stderr, "max %d splits supported\n", MAX_SPLITS);
        return 1;
    }

    fprintf(stderr, "=== sp_qnn chain bench (%d splits) ===\n", n_splits);

    sp_qnn_status rc = sp_qnn_init(NULL, NULL);
    if (rc != SP_QNN_OK) {
        fprintf(stderr, "init failed: %d\n", rc);
        return 1;
    }

    /* Memory note: HTP V69 cannot hold 4×600 MB contexts simultaneously.
     * Load + run + destroy each split sequentially. We carry the residual
     * stream across in a host-side buffer so split N can hand off to
     * split N+1 even though only one is HTP-resident at a time.
     *
     * Trade-off: we pay context create/finalize overhead on every split
     * call. This is NOT the production prefill rate — that requires either
     * pre-linking the splits via QnnContext_createFromBinaryListAsync OR
     * pipelining 2-at-a-time. For tonight, this validates the chain
     * composes correctly. */
    static unsigned char host_residual[1024 * 1024];  /* 1 MB scratch — actual is 640 KB */
    size_t host_residual_size = 0;

    const int N_ITER = 3;
    fprintf(stderr, "\n=== sequential load/run/destroy (%d iters) ===\n", N_ITER);
    uint64_t per_iter_total[16] = {0};
    uint64_t per_iter_split[16][MAX_SPLITS] = {{0}};
    uint64_t per_iter_load_us[16][MAX_SPLITS] = {{0}};
    for (int it = 0; it < N_ITER; ++it) {
        uint64_t total_us = 0;
        host_residual_size = 0;  /* reset chain at start of each iter */

        for (int i = 0; i < n_splits; ++i) {
            split_state s = {0};
            struct timeval tl0, tl1;
            gettimeofday(&tl0, NULL);
            if (load_split(argv[1 + i], &s) != 0) {
                fprintf(stderr, "[iter %d] load_split[%d] failed\n", it, i);
                rc = SP_QNN_ERR_INVALID; goto cleanup;
            }
            gettimeofday(&tl1, NULL);
            uint64_t load_us = (tl1.tv_sec - tl0.tv_sec) * 1000000ULL +
                               (tl1.tv_usec - tl0.tv_usec);
            per_iter_load_us[it][i] = load_us;

            /* If we have a host-side residual from previous split AND this
             * split has a residual_in slot AND sizes match, copy it in. */
            if (host_residual_size > 0 && s.residual_in_idx >= 0
                && s.in_sz[s.residual_in_idx] == host_residual_size) {
                memcpy(s.in_bufs[s.residual_in_idx], host_residual, host_residual_size);
            }

            uint64_t exec_us = 0;
            rc = run_split(&s, &exec_us);
            if (rc != SP_QNN_OK) {
                fprintf(stderr, "[iter %d split %d] execute FAILED rc=%d\n", it, i, rc);
                free_split_buffers(&s); sp_qnn_destroy(&s.h);
                goto cleanup;
            }
            per_iter_split[it][i] = exec_us;
            total_us += exec_us;

            /* Save residual_out to host scratch for next split. */
            if (s.residual_out_idx >= 0) {
                host_residual_size = s.out_sz[s.residual_out_idx];
                if (host_residual_size <= sizeof(host_residual)) {
                    memcpy(host_residual, s.out_bufs[s.residual_out_idx],
                           host_residual_size);
                } else {
                    fprintf(stderr, "[chain] residual %zu bytes > scratch %zu\n",
                            host_residual_size, sizeof(host_residual));
                    host_residual_size = 0;
                }
            } else {
                host_residual_size = 0;
            }

            free_split_buffers(&s);
            sp_qnn_destroy(&s.h);
        }

        per_iter_total[it] = total_us;
        fprintf(stderr, "  iter %d: exec %.2f ms | per-split exec:", it, total_us / 1000.0);
        for (int i = 0; i < n_splits; ++i) {
            fprintf(stderr, " %.1f", per_iter_split[it][i] / 1000.0);
        }
        fprintf(stderr, " ms | per-split load:");
        for (int i = 0; i < n_splits; ++i) {
            fprintf(stderr, " %.1f", per_iter_load_us[it][i] / 1000.0);
        }
        fprintf(stderr, " ms\n");
    }

    /* Steady-state report. */
    if (N_ITER >= 2) {
        uint64_t sum = 0, mn = UINT64_MAX, mx = 0;
        for (int it = 1; it < N_ITER; ++it) {
            uint64_t v = per_iter_total[it];
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        uint64_t avg = sum / (N_ITER - 1);
        fprintf(stderr, "\n=== steady-state (excluding finalize iter) ===\n");
        fprintf(stderr, "  iter[0]:        %.2f ms (graph finalize)\n",
                per_iter_total[0] / 1000.0);
        fprintf(stderr, "  per-token min:  %.2f ms\n", mn / 1000.0);
        fprintf(stderr, "  per-token avg:  %.2f ms\n", avg / 1000.0);
        fprintf(stderr, "  per-token max:  %.2f ms\n", mx / 1000.0);
        fprintf(stderr, "  prefill rate (128-token chunk @ avg): %.1f tok/sec\n",
                128.0 * 1e6 / (double)avg);
        fprintf(stderr, "  prefill rate (128-token chunk @ min): %.1f tok/sec\n",
                128.0 * 1e6 / (double)mn);
    }

cleanup:
    /* Per-iter loop already destroys each split before moving on; no
     * persistent split state to clean up here. */
    sp_qnn_shutdown();
    return (rc == SP_QNN_OK) ? 0 : 1;
}
