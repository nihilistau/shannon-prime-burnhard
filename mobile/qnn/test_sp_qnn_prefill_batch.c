/*
 * Phase 2.4 — batch-amortized prefill bench. Tests the actual production
 * pattern given V69 HTP's ~1.5 GB context-residency ceiling.
 *
 * Pattern per 128-token prefill chunk:
 *   1. load split 1 → exec → load split 2 (now 1+2 resident) → exec
 *   2. destroy splits 1+2
 *   3. load split 3 → exec → load split 4 (now 3+4 resident) → exec
 *   4. destroy splits 3+4
 *
 * Total per chunk: 4 loads + 4 execs + 4 destroys. The loads dominate
 * wall-clock (HTP context create is 600-1500 ms each); execs are 90-140ms.
 * Tokens/sec = 128 / wall-clock-per-chunk.
 *
 * This is the realistic production rate for prefill on V69 HTP with the
 * current 4-split shape, until either:
 *   (a) shareResources lands in V73+ silicon
 *   (b) we re-export to balance split sizes (split 4 currently 63% bigger)
 *   (c) the full Mode C streaming pipeline supersedes static contexts
 */
#include "sp_qnn.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef struct {
    sp_qnn_handle *h;
    size_t   n_in, n_out;
    void   **in_bufs;
    size_t  *in_sz;
    void   **out_bufs;
    size_t  *out_sz;
    int      residual_in_idx;
    int      residual_out_idx;
} split_state;

static size_t tensor_bytes(const sp_qnn_tensor_info *t) {
    size_t n = t->bytes_per_element;
    if (n == 0) n = 1;
    for (uint32_t d = 0; d < t->rank; ++d) n *= t->dims[d];
    return n;
}
static int find_residual_idx(const sp_qnn_tensor_info *infos, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (infos[i].rank == 3 && infos[i].dims[0] == 1
            && infos[i].dims[1] == 128 && infos[i].dims[2] == 2560)
            return (int)i;
    return -1;
}
static void free_split(split_state *s) {
    if (s->in_bufs)  { for (size_t i = 0; i < s->n_in;  ++i) free(s->in_bufs[i]);  free(s->in_bufs); }
    if (s->out_bufs) { for (size_t i = 0; i < s->n_out; ++i) free(s->out_bufs[i]); free(s->out_bufs); }
    free(s->in_sz); free(s->out_sz);
    if (s->h) sp_qnn_destroy(&s->h);
    memset(s, 0, sizeof(*s));
}
static int load_split(const char *path, split_state *s) {
    memset(s, 0, sizeof(*s));
    if (sp_qnn_load_binary(path, NULL, &s->h) != SP_QNN_OK) return -1;
    const sp_qnn_tensor_info *in_info = NULL, *out_info = NULL;
    sp_qnn_get_io_info(s->h, &s->n_in, &in_info, &s->n_out, &out_info);
    s->in_bufs  = calloc(s->n_in,  sizeof(void *));
    s->in_sz    = calloc(s->n_in,  sizeof(size_t));
    s->out_bufs = calloc(s->n_out, sizeof(void *));
    s->out_sz   = calloc(s->n_out, sizeof(size_t));
    for (size_t i = 0; i < s->n_in;  ++i) {
        s->in_sz[i]  = tensor_bytes(&in_info[i]);
        s->in_bufs[i]  = calloc(1, s->in_sz[i]);
    }
    for (size_t i = 0; i < s->n_out; ++i) {
        s->out_sz[i] = tensor_bytes(&out_info[i]);
        s->out_bufs[i] = calloc(1, s->out_sz[i]);
    }
    s->residual_in_idx  = find_residual_idx(in_info,  s->n_in);
    s->residual_out_idx = find_residual_idx(out_info, s->n_out);
    return 0;
}
static uint64_t now_us(void) {
    struct timeval t; gettimeofday(&t, NULL);
    return (uint64_t)t.tv_sec * 1000000ULL + (uint64_t)t.tv_usec;
}

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s split1 split2 split3 split4\n", argv[0]); return 1; }
    fprintf(stderr, "=== Phase 2.4 — batch-amortized prefill bench ===\n");
    fprintf(stderr, "Pattern: load(1)→exec→load(2)→exec→destroy(1,2)→load(3)→exec→load(4)→exec→destroy(3,4)\n");
    fprintf(stderr, "Per chunk: 128 tokens prefilled, ~4 loads + 4 execs wall-clock.\n\n");

    if (sp_qnn_init(NULL, NULL) != SP_QNN_OK) return 1;

    static unsigned char host_residual[1 << 20];
    size_t host_residual_size = 0;

    const int N_CHUNKS = 3;
    uint64_t chunk_total_us[16] = {0};

    for (int c = 0; c < N_CHUNKS; ++c) {
        fprintf(stderr, "=== chunk %d (128 tokens) ===\n", c);
        host_residual_size = 0;
        uint64_t chunk_start = now_us();
        split_state slot_a = {0}, slot_b = {0};
        uint64_t load_us[4] = {0}, exec_us[4] = {0};

        /* Phase A: split 1 + split 2 */
        uint64_t t = now_us();
        if (load_split(argv[1], &slot_a) != 0) { fprintf(stderr, "load 1 fail\n"); return 2; }
        load_us[0] = now_us() - t;

        if (sp_qnn_execute(slot_a.h,
            (const void *const *)slot_a.in_bufs, slot_a.in_sz,
            (void *const *)slot_a.out_bufs, slot_a.out_sz, &exec_us[0]) != SP_QNN_OK) {
            fprintf(stderr, "exec 1 fail\n"); return 3;
        }
        if (slot_a.residual_out_idx >= 0) {
            host_residual_size = slot_a.out_sz[slot_a.residual_out_idx];
            memcpy(host_residual, slot_a.out_bufs[slot_a.residual_out_idx], host_residual_size);
        }

        t = now_us();
        if (load_split(argv[2], &slot_b) != 0) { fprintf(stderr, "load 2 fail\n"); return 2; }
        load_us[1] = now_us() - t;

        if (slot_b.residual_in_idx >= 0
            && host_residual_size == slot_b.in_sz[slot_b.residual_in_idx]) {
            memcpy(slot_b.in_bufs[slot_b.residual_in_idx], host_residual, host_residual_size);
        }
        if (sp_qnn_execute(slot_b.h,
            (const void *const *)slot_b.in_bufs, slot_b.in_sz,
            (void *const *)slot_b.out_bufs, slot_b.out_sz, &exec_us[1]) != SP_QNN_OK) {
            fprintf(stderr, "exec 2 fail\n"); return 3;
        }
        if (slot_b.residual_out_idx >= 0) {
            host_residual_size = slot_b.out_sz[slot_b.residual_out_idx];
            memcpy(host_residual, slot_b.out_bufs[slot_b.residual_out_idx], host_residual_size);
        }

        /* Phase B: destroy 1+2, load split 3+4 */
        free_split(&slot_a);
        free_split(&slot_b);

        t = now_us();
        if (load_split(argv[3], &slot_a) != 0) { fprintf(stderr, "load 3 fail\n"); return 2; }
        load_us[2] = now_us() - t;

        if (slot_a.residual_in_idx >= 0
            && host_residual_size == slot_a.in_sz[slot_a.residual_in_idx]) {
            memcpy(slot_a.in_bufs[slot_a.residual_in_idx], host_residual, host_residual_size);
        }
        if (sp_qnn_execute(slot_a.h,
            (const void *const *)slot_a.in_bufs, slot_a.in_sz,
            (void *const *)slot_a.out_bufs, slot_a.out_sz, &exec_us[2]) != SP_QNN_OK) {
            fprintf(stderr, "exec 3 fail\n"); return 3;
        }
        if (slot_a.residual_out_idx >= 0) {
            host_residual_size = slot_a.out_sz[slot_a.residual_out_idx];
            memcpy(host_residual, slot_a.out_bufs[slot_a.residual_out_idx], host_residual_size);
        }

        t = now_us();
        if (load_split(argv[4], &slot_b) != 0) { fprintf(stderr, "load 4 fail\n"); return 2; }
        load_us[3] = now_us() - t;

        if (slot_b.residual_in_idx >= 0
            && host_residual_size == slot_b.in_sz[slot_b.residual_in_idx]) {
            memcpy(slot_b.in_bufs[slot_b.residual_in_idx], host_residual, host_residual_size);
        }
        if (sp_qnn_execute(slot_b.h,
            (const void *const *)slot_b.in_bufs, slot_b.in_sz,
            (void *const *)slot_b.out_bufs, slot_b.out_sz, &exec_us[3]) != SP_QNN_OK) {
            fprintf(stderr, "exec 4 fail\n"); return 3;
        }

        free_split(&slot_a);
        free_split(&slot_b);

        chunk_total_us[c] = now_us() - chunk_start;
        uint64_t load_sum = load_us[0]+load_us[1]+load_us[2]+load_us[3];
        uint64_t exec_sum = exec_us[0]+exec_us[1]+exec_us[2]+exec_us[3];
        fprintf(stderr, "  loads: %.0f %.0f %.0f %.0f ms (sum %.0f)\n",
                load_us[0]/1000.0, load_us[1]/1000.0, load_us[2]/1000.0, load_us[3]/1000.0, load_sum/1000.0);
        fprintf(stderr, "  execs: %.0f %.0f %.0f %.0f ms (sum %.0f)\n",
                exec_us[0]/1000.0, exec_us[1]/1000.0, exec_us[2]/1000.0, exec_us[3]/1000.0, exec_sum/1000.0);
        fprintf(stderr, "  chunk wall: %.0f ms = %.1f tok/sec (128-token chunk)\n",
                chunk_total_us[c]/1000.0, 128.0 * 1e6 / (double)chunk_total_us[c]);
    }

    /* Steady-state: average over all but first chunk. */
    if (N_CHUNKS >= 2) {
        uint64_t sum = 0, mn = UINT64_MAX, mx = 0;
        for (int c = 1; c < N_CHUNKS; ++c) {
            uint64_t v = chunk_total_us[c];
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        uint64_t avg = sum / (N_CHUNKS - 1);
        fprintf(stderr, "\n=== steady-state (excl chunk 0) ===\n");
        fprintf(stderr, "  chunk[0]:        %.0f ms\n", chunk_total_us[0]/1000.0);
        fprintf(stderr, "  steady min:      %.0f ms = %.1f tok/sec\n", mn/1000.0, 128.0 * 1e6 / (double)mn);
        fprintf(stderr, "  steady avg:      %.0f ms = %.1f tok/sec\n", avg/1000.0, 128.0 * 1e6 / (double)avg);
        fprintf(stderr, "  steady max:      %.0f ms = %.1f tok/sec\n", mx/1000.0, 128.0 * 1e6 / (double)mx);
    }

    sp_qnn_shutdown();
    return 0;
}
