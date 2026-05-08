/*
 * Multi-tensor smoke test for libsp_qnn — handles graphs with arbitrary
 * input/output counts (transformer splits typically have 28+ tensors).
 * Allocates zero-filled buffers for every declared input/output, runs N
 * iterations of graphExecute, prints min/avg/max latency.
 *
 * The first execute on HTP performs graph finalization (slow); we report
 * iter[0] separately and average iter[1..N-1] for steady-state numbers.
 */
#include "sp_qnn.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t tensor_bytes(const sp_qnn_tensor_info *t) {
    size_t n = t->bytes_per_element;
    if (n == 0) n = 1; /* unknown dtype — treat as 1 byte to avoid zero-alloc */
    for (uint32_t d = 0; d < t->rank; ++d) n *= t->dims[d];
    return n;
}

int main(int argc, char **argv) {
    const char *bin   = (argc > 1) ? argv[1] : "qwen3_4b_3_of_4.bin";
    const int   iters = (argc > 2) ? atoi(argv[2]) : 5;

    fprintf(stderr, "=== sp_qnn multi-tensor bench ===\n");
    fprintf(stderr, "  binary: %s\n", bin);
    fprintf(stderr, "  iters:  %d\n", iters);

    sp_qnn_status rc = sp_qnn_init(NULL, NULL);
    if (rc != SP_QNN_OK) { fprintf(stderr, "init failed: %d\n", rc); return 1; }

    sp_qnn_handle *h = NULL;
    rc = sp_qnn_load_binary(bin, NULL, &h);
    if (rc != SP_QNN_OK) { fprintf(stderr, "load_binary failed: %d\n", rc); return 2; }

    size_t n_in = 0, n_out = 0;
    const sp_qnn_tensor_info *in_info = NULL, *out_info = NULL;
    sp_qnn_get_io_info(h, &n_in, &in_info, &n_out, &out_info);
    fprintf(stderr, "  graph: %zu inputs, %zu outputs\n", n_in, n_out);

    /* Allocate flat arrays of pointers + sizes. */
    void  **in_bufs  = calloc(n_in,  sizeof(void *));
    size_t *in_sz    = calloc(n_in,  sizeof(size_t));
    void  **out_bufs = calloc(n_out, sizeof(void *));
    size_t *out_sz   = calloc(n_out, sizeof(size_t));
    if (!in_bufs || !in_sz || !out_bufs || !out_sz) {
        fprintf(stderr, "ptr-array alloc failed\n"); return 3;
    }

    size_t total_in = 0, total_out = 0;
    for (size_t i = 0; i < n_in; ++i) {
        in_sz[i] = tensor_bytes(&in_info[i]);
        in_bufs[i] = calloc(1, in_sz[i]);
        if (!in_bufs[i]) { fprintf(stderr, "in[%zu] alloc %zu failed\n", i, in_sz[i]); return 4; }
        total_in += in_sz[i];
    }
    for (size_t i = 0; i < n_out; ++i) {
        out_sz[i] = tensor_bytes(&out_info[i]);
        out_bufs[i] = calloc(1, out_sz[i]);
        if (!out_bufs[i]) { fprintf(stderr, "out[%zu] alloc %zu failed\n", i, out_sz[i]); return 5; }
        total_out += out_sz[i];
    }
    fprintf(stderr, "  total in:  %zu bytes (%.1f MB)\n", total_in, total_in/1024.0/1024.0);
    fprintf(stderr, "  total out: %zu bytes (%.1f MB)\n", total_out, total_out/1024.0/1024.0);

    /* iter[0] is special: graph finalize happens lazily on first execute.
     * Report it separately and average the rest. */
    fprintf(stderr, "\n=== iterations ===\n");
    uint64_t first_us = 0, sum_us = 0, min_us = UINT64_MAX, max_us = 0;
    int steady_n = 0;
    for (int i = 0; i < iters; ++i) {
        uint64_t us = 0;
        rc = sp_qnn_execute(h,
                            (const void *const *)in_bufs,  in_sz,
                            (void *const *)out_bufs,       out_sz,
                            &us);
        if (rc != SP_QNN_OK) {
            fprintf(stderr, "  iter %d: FAILED rc=%d\n", i, rc);
            break;
        }
        fprintf(stderr, "  iter %d: %" PRIu64 " us (%.2f ms)\n",
                i, us, us / 1000.0);
        if (i == 0) {
            first_us = us;
        } else {
            if (us < min_us) min_us = us;
            if (us > max_us) max_us = us;
            sum_us += us;
            steady_n++;
        }
    }

    if (steady_n > 0) {
        uint64_t avg_us = sum_us / steady_n;
        fprintf(stderr, "\n=== steady-state (excluding first/finalize iter) ===\n");
        fprintf(stderr, "  iter[0]   : %" PRIu64 " us (%.2f ms) [graph finalize]\n",
                first_us, first_us / 1000.0);
        fprintf(stderr, "  steady min: %" PRIu64 " us (%.2f ms)\n", min_us, min_us / 1000.0);
        fprintf(stderr, "  steady avg: %" PRIu64 " us (%.2f ms)\n", avg_us, avg_us / 1000.0);
        fprintf(stderr, "  steady max: %" PRIu64 " us (%.2f ms)\n", max_us, max_us / 1000.0);

        /* Project full-token rate: this split is 12 layers, model has 4
         * splits, so 4 graphExecute calls per token in pipeline. */
        double avg_s   = avg_us / 1e6;
        double per_tok = 4 * avg_s;       /* 4 splits sequential */
        double tok_s   = 1.0 / per_tok;
        fprintf(stderr, "\n=== full-model projection ===\n");
        fprintf(stderr, "  per split  (steady): %.2f ms\n", avg_us / 1000.0);
        fprintf(stderr, "  per token  (4 splits sequential): %.2f ms\n", per_tok * 1000.0);
        fprintf(stderr, "  prefill rate (128-token chunk): %.1f tok/sec\n",
                128.0 / per_tok);
        fprintf(stderr, "  if pipelined (4 splits parallel): %.1f tok/sec\n", tok_s);
    }

    for (size_t i = 0; i < n_in;  ++i) free(in_bufs[i]);
    for (size_t i = 0; i < n_out; ++i) free(out_bufs[i]);
    free(in_bufs); free(in_sz); free(out_bufs); free(out_sz);
    sp_qnn_destroy(&h);
    sp_qnn_shutdown();
    return (rc == SP_QNN_OK) ? 0 : 1;
}
