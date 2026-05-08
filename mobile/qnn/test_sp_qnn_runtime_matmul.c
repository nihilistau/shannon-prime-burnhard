/*
 * Phase 2.5 — runtime graph build, no .bin in sight.
 *
 * Builds a single MatMul op at runtime via QnnGraph_addNode, with weights
 * as APP_WRITE inputs (not STATIC baked into the prepared graph). Compares
 * against Phase 2.1 reference: 256x256 fp32 matmul = 81 µs on V69 HTP.
 *
 * If this works, the .bin AOT-compile flow becomes optional: our kernel
 * can build any op-by-op compute pattern at runtime, weights stream in
 * from our own DDR/UFS-backed buffer per execute(). Mode C + ISP fanout
 * follow as orchestration on top of this primitive.
 */
#include "sp_qnn.h"
#include "QnnTypes.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    uint32_t M = (argc > 1) ? (uint32_t)atoi(argv[1]) : 256;
    uint32_t K = (argc > 2) ? (uint32_t)atoi(argv[2]) : 256;
    uint32_t N = (argc > 3) ? (uint32_t)atoi(argv[3]) : 256;
    int n_iter = (argc > 4) ? atoi(argv[4]) : 50;
    /* arg 5: "fp32" (default) or "fp16" */
    Qnn_DataType_t dtype = QNN_DATATYPE_FLOAT_32;
    int bytes_per_el = 4;
    const char *dtype_name = "fp32";
    if (argc > 5 && strcmp(argv[5], "fp16") == 0) {
        dtype = QNN_DATATYPE_FLOAT_16;
        bytes_per_el = 2;
        dtype_name = "fp16";
    }

    fprintf(stderr, "=== Phase 2.5 runtime graph matmul (no .bin) ===\n");
    fprintf(stderr, "  shape: A[%u,%u] x B[%u,%u] = C[%u,%u] %s, %d iters\n\n",
            M, K, K, N, M, N, dtype_name, n_iter);
    fprintf(stderr, "  Reference (Phase 2.1, 256x256 fp32 .bin): 81 us steady min\n");
    fprintf(stderr, "  Goal: comparable latency, proving runtime-graph path is real.\n\n");

    if (sp_qnn_init(NULL, NULL) != SP_QNN_OK) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    sp_qnn_handle *h = NULL;
    sp_qnn_status rc = sp_qnn_runtime_matmul_create(M, K, N,
                                                    (uint32_t)dtype,
                                                    &h);
    if (rc != SP_QNN_OK) {
        fprintf(stderr, "runtime_matmul_create FAILED rc=%d\n", rc);
        sp_qnn_shutdown();
        return 2;
    }

    /* Allocate dummy A and B, output C. */
    size_t a_bytes = (size_t)M * K * bytes_per_el;
    size_t b_bytes = (size_t)K * N * bytes_per_el;
    size_t c_bytes = (size_t)M * N * bytes_per_el;
    void *a = calloc(1, a_bytes);
    void *b = calloc(1, b_bytes);
    void *c = calloc(1, c_bytes);

    /* Fill with non-zero pattern so the matmul has something to compute.
     * For fp16 we skip the fill — zeros are fine for latency bench, the
     * HTP doesn't shortcut on zero-valued inputs (it still does the FLOPs). */
    if (dtype == QNN_DATATYPE_FLOAT_32) {
        float *af = (float *)a;
        float *bf = (float *)b;
        for (size_t i = 0; i < (size_t)M*K; ++i) af[i] = (float)((i % 7) - 3) * 0.01f;
        for (size_t i = 0; i < (size_t)K*N; ++i) bf[i] = (float)((i % 5) - 2) * 0.01f;
    }

    const void *ins[]  = { a, b };
    const size_t in_sz[] = { a_bytes, b_bytes };
    void *outs[] = { c };
    const size_t out_sz[] = { c_bytes };

    sp_qnn_bench_result br = {0};
    rc = sp_qnn_bench(h, ins, in_sz, outs, out_sz, (uint32_t)n_iter, &br);
    if (rc != SP_QNN_OK) {
        fprintf(stderr, "bench failed rc=%d\n", rc);
    } else {
        fprintf(stderr, "=== bench result (runtime graph, %u iters) ===\n", br.n_iterations);
        fprintf(stderr, "  min:  %" PRIu64 " us  (%.3f ms)\n", br.min_us, br.min_us/1000.0);
        fprintf(stderr, "  avg:  %" PRIu64 " us  (%.3f ms)\n", br.avg_us, br.avg_us/1000.0);
        fprintf(stderr, "  max:  %" PRIu64 " us  (%.3f ms)\n", br.max_us, br.max_us/1000.0);
        fprintf(stderr, "\n  vs Phase 2.1 .bin reference 81 us: ");
        if (br.min_us < 200) fprintf(stderr, "MATCH (runtime path is real)\n");
        else fprintf(stderr, "%.1fx of reference\n", br.min_us / 81.0);
    }

    /* Sanity check: only meaningful for fp32 (we skipped the fp16 fill). */
    if (dtype == QNN_DATATYPE_FLOAT_32) {
        float *cf = (float *)c;
        int nonzero = 0;
        for (int i = 0; i < 16; ++i) if (cf[i] != 0.0f) nonzero++;
        fprintf(stderr, "\n  output sanity: %d/16 first elements non-zero\n", nonzero);
        fprintf(stderr, "  c[0..3]: %.4f %.4f %.4f %.4f\n", cf[0], cf[1], cf[2], cf[3]);
    }

    free(a); free(b); free(c);
    sp_qnn_destroy(&h);
    sp_qnn_shutdown();
    return (rc == SP_QNN_OK) ? 0 : 1;
}
