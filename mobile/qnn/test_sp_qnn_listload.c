/*
 * Test for sp_qnn_load_binary_list — uses QnnContext_createFromBinaryListAsync
 * with HTP shareResources=true so all 4 PP splits can co-reside in HTP
 * working memory by sharing kernel/workspace state. If this works, we get
 * "load all 4 once at startup" + 391 t/s exec = real production wall-clock.
 */
#include "sp_qnn.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define MAX_N 8

static size_t tensor_bytes(const sp_qnn_tensor_info *t) {
    size_t n = t->bytes_per_element;
    if (n == 0) n = 1;
    for (uint32_t d = 0; d < t->rank; ++d) n *= t->dims[d];
    return n;
}

static uint64_t now_us(void) {
    struct timeval t; gettimeofday(&t, NULL);
    return (uint64_t)t.tv_sec * 1000000ULL + (uint64_t)t.tv_usec;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <split1.bin> [split2.bin] ... [splitN.bin]\n", argv[0]);
        return 1;
    }
    int n = argc - 1;
    if (n > MAX_N) { fprintf(stderr, "max %d\n", MAX_N); return 1; }

    fprintf(stderr, "=== sp_qnn list-load test (n=%d) ===\n", n);
    fprintf(stderr, "Goal: all %d .bins co-resident via shareResources=true\n\n", n);

    if (sp_qnn_init(NULL, NULL) != SP_QNN_OK) return 1;

    const char *paths[MAX_N] = {0};
    sp_qnn_handle *handles[MAX_N] = {0};
    for (int i = 0; i < n; ++i) paths[i] = argv[1 + i];

    uint64_t t0 = now_us();
    sp_qnn_status rc = sp_qnn_load_binary_list(paths, NULL, n, handles);
    uint64_t t1 = now_us();
    if (rc != SP_QNN_OK) {
        fprintf(stderr, "\n=== list-load FAILED rc=%d ===\n", rc);
        sp_qnn_shutdown();
        return 2;
    }
    fprintf(stderr, "\n=== list-load OK in %.1f ms ===\n", (t1-t0)/1000.0);
    fprintf(stderr, "All %d contexts now HTP-resident simultaneously.\n\n", n);

    /* Now run each split's graphExecute in sequence to confirm they all
     * actually work (no swap, no reload). Allocate dummy buffers per split. */
    for (int i = 0; i < n; ++i) {
        size_t n_in = 0, n_out = 0;
        const sp_qnn_tensor_info *in_info = NULL, *out_info = NULL;
        sp_qnn_get_io_info(handles[i], &n_in, &in_info, &n_out, &out_info);

        void  **in_bufs  = calloc(n_in,  sizeof(void *));
        size_t *in_sz    = calloc(n_in,  sizeof(size_t));
        void  **out_bufs = calloc(n_out, sizeof(void *));
        size_t *out_sz   = calloc(n_out, sizeof(size_t));
        for (size_t k = 0; k < n_in;  ++k) {
            in_sz[k]  = tensor_bytes(&in_info[k]);
            in_bufs[k]  = calloc(1, in_sz[k]);
        }
        for (size_t k = 0; k < n_out; ++k) {
            out_sz[k] = tensor_bytes(&out_info[k]);
            out_bufs[k] = calloc(1, out_sz[k]);
        }

        uint64_t exec_us = 0;
        rc = sp_qnn_execute(handles[i],
                            (const void *const *)in_bufs, in_sz,
                            (void *const *)out_bufs, out_sz, &exec_us);
        if (rc != SP_QNN_OK) {
            fprintf(stderr, "  split[%d] exec FAILED rc=%d\n", i, rc);
        } else {
            fprintf(stderr, "  split[%d] exec %.2f ms (%zu in / %zu out)\n",
                    i, exec_us / 1000.0, n_in, n_out);
        }

        for (size_t k = 0; k < n_in;  ++k) free(in_bufs[k]);
        for (size_t k = 0; k < n_out; ++k) free(out_bufs[k]);
        free(in_bufs); free(in_sz); free(out_bufs); free(out_sz);
    }

    fprintf(stderr, "\n=== cleanup ===\n");
    for (int i = 0; i < n; ++i) sp_qnn_destroy(&handles[i]);
    sp_qnn_shutdown();
    return 0;
}
