/*
 * Test driver for libsp_qnn — load our existing AI-Hub-validated
 * v69_attn_qwen3_4b.bin and bench it via our own runner. Compare
 * against qnn-net-run's measurement (1.5 ms steady-state) to confirm
 * the shim doesn't introduce overhead.
 */
#include "sp_qnn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

int main(int argc, char **argv) {
    const char *bin   = (argc > 1) ? argv[1] : "v69_attn_qwen3_4b.bin";
    const int   iters = (argc > 2) ? atoi(argv[2]) : 50;
    fprintf(stderr, "=== sp_qnn bench (binary=%s, iters=%d) ===\n", bin, iters);

    sp_qnn_status rc = sp_qnn_init(NULL, NULL);
    if (rc != SP_QNN_OK) { fprintf(stderr, "init failed: %d\n", rc); return 1; }

    sp_qnn_handle *h = NULL;
    rc = sp_qnn_load_binary(bin, NULL, &h);
    if (rc != SP_QNN_OK) { fprintf(stderr, "load_binary failed: %d\n", rc); return 2; }

    /* Inspect I/O. */
    size_t n_in = 0, n_out = 0;
    const sp_qnn_tensor_info *in_info = NULL, *out_info = NULL;
    sp_qnn_get_io_info(h, &n_in, &in_info, &n_out, &out_info);
    fprintf(stderr, "I/O metadata:\n");
    for (size_t i = 0; i < n_in; ++i) {
        fprintf(stderr, "  in[%zu]  '%s' rank=%u dtype=%u bpe=%u dims=[",
                i, in_info[i].name, in_info[i].rank, in_info[i].dtype,
                in_info[i].bytes_per_element);
        size_t total = in_info[i].bytes_per_element;
        for (uint32_t d = 0; d < in_info[i].rank; ++d) {
            fprintf(stderr, "%s%u", d?",":"", in_info[i].dims[d]);
            total *= in_info[i].dims[d];
        }
        fprintf(stderr, "]  total=%zu bytes\n", total);
    }
    for (size_t i = 0; i < n_out; ++i) {
        fprintf(stderr, "  out[%zu] '%s' rank=%u dtype=%u bpe=%u dims=[",
                i, out_info[i].name, out_info[i].rank, out_info[i].dtype,
                out_info[i].bytes_per_element);
        size_t total = out_info[i].bytes_per_element;
        for (uint32_t d = 0; d < out_info[i].rank; ++d) {
            fprintf(stderr, "%s%u", d?",":"", out_info[i].dims[d]);
            total *= out_info[i].dims[d];
        }
        fprintf(stderr, "]  total=%zu bytes\n", total);
    }

    /* Allocate I/O buffers sized to first input/output. */
    if (n_in != 1 || n_out != 1) {
        fprintf(stderr, "expected 1 input + 1 output for this test (got %zu/%zu)\n",
                n_in, n_out);
        sp_qnn_destroy(&h); sp_qnn_shutdown(); return 3;
    }

    size_t in_bytes = in_info[0].bytes_per_element;
    for (uint32_t d = 0; d < in_info[0].rank; ++d) in_bytes *= in_info[0].dims[d];
    size_t out_bytes = out_info[0].bytes_per_element;
    for (uint32_t d = 0; d < out_info[0].rank; ++d) out_bytes *= out_info[0].dims[d];

    void *in_buf  = calloc(1, in_bytes);
    void *out_buf = calloc(1, out_bytes);
    if (!in_buf || !out_buf) { fprintf(stderr, "alloc failed\n"); return 4; }

    /* Bench. */
    fprintf(stderr, "running %d iterations...\n", iters);
    const void * const ins[]  = { in_buf };
    const size_t   in_sz[] = { in_bytes };
    void * const   outs[] = { out_buf };
    const size_t   out_sz[] = { out_bytes };

    sp_qnn_bench_result br = {0};
    rc = sp_qnn_bench(h, ins, in_sz, outs, out_sz, (uint32_t)iters, &br);
    if (rc != SP_QNN_OK) {
        fprintf(stderr, "bench failed: %d\n", rc);
    } else {
        fprintf(stderr, "=== bench result ===\n");
        fprintf(stderr, "  iterations: %u\n", br.n_iterations);
        fprintf(stderr, "  min:        %" PRIu64 " us\n", br.min_us);
        fprintf(stderr, "  avg:        %" PRIu64 " us\n", br.avg_us);
        fprintf(stderr, "  max:        %" PRIu64 " us\n", br.max_us);
        fprintf(stderr, "  reference (qnn-net-run Phase 2.2 min): 1916 us\n");

        /* Phase 2.4 budget check: how many calls/sec does this shape
         * sustain? At 1.5 ms steady state, 28 layers per token,
         * we need >=28 calls / desired-tok-period. */
        double avg_s   = br.avg_us / 1e6;
        double rate    = 1.0 / avg_s;
        double tok_28l = rate / 28.0;   /* if dispatching all 28 layers
                                           sequentially through the shim */
        double tok_1   = rate;          /* if the .bin is the full layer */
        fprintf(stderr, "\n=== Phase 2.4 budget projection ===\n");
        fprintf(stderr, "  sustained calls/sec:        %.0f\n", rate);
        fprintf(stderr, "  tok/sec @ 28 layers/call:   %.1f  (current .bin is 1 attn block)\n", tok_28l);
        fprintf(stderr, "  tok/sec @ 1 layer/call:     %.1f  (if full graph compiled)\n", tok_1);
        fprintf(stderr, "  fp32 -> w4a16 prediction:   %.1f -> %.1f tok/sec @ 28 layers\n",
                tok_28l, tok_28l * (1500.0 / 800.0));  /* ~1.9x speedup */
    }

    free(in_buf);
    free(out_buf);
    sp_qnn_destroy(&h);
    sp_qnn_shutdown();
    return (rc == SP_QNN_OK) ? 0 : 1;
}
