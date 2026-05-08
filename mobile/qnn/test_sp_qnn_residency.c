/*
 * Phase 2.3 stage 3 / Mode C — split-pair residency manager.
 *
 * The HTP can't hold all 4 transformer splits resident simultaneously
 * (~3 GB total contexts). The chain test that loads/destroys per split
 * paid 600-1500 ms per split = ~3.5 sec wall-clock per iter, masking
 * the 327 ms steady-state exec.
 *
 * That load overhead is a TEST ARTIFACT, not a production constraint.
 * Production pattern (Mode C at split granularity):
 *   Load split[1,2] → run, run → swap split[2]→split[3] → run → swap
 *   split[3]→split[4] → run. Always 2 resident. Per-token cost = 4 execs
 *   + 2 swaps (one swap is destroy(N) + load(N+2)).
 *
 * This test measures the real swap cost so we know if 2-pair Mode C is
 * viable, OR if we need to commit to the DLC re-compile + link path.
 *
 * Output reports:
 *   - per-iter total wall-clock (target: <500 ms)
 *   - swap cost (destroy(N) + load(N+2)) measured separately
 *   - per-split exec breakdown
 *
 * If swap cost < 200 ms each, total iter ~700 ms = 180 t/s. Good enough
 * to ship without link_job. If swap >> 500 ms, link_job is needed.
 */
#include "sp_qnn.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define N_SPLITS 4

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
    if (argc != 5) {
        fprintf(stderr, "usage: %s split1 split2 split3 split4\n", argv[0]);
        return 1;
    }
    fprintf(stderr, "=== sp_qnn split-pair residency bench ===\n");
    fprintf(stderr, "Pattern: load(1,2) → run(1) run(2) → swap(2→3) → run(3) → swap(3→4) → run(4)\n");
    fprintf(stderr, "Always 2 splits HTP-resident. 1 swap per 2 split runs.\n\n");

    if (sp_qnn_init(NULL, NULL) != SP_QNN_OK) return 1;

    /* Initial load: splits 1 and 2 resident. */
    fprintf(stderr, "=== initial load: split[0,1] ===\n");
    split_state slot_a = {0}, slot_b = {0};
    uint64_t t0 = now_us();
    if (load_split(argv[1], &slot_a) != 0) { fprintf(stderr, "load split 0 failed\n"); return 2; }
    uint64_t t_load_a = now_us() - t0;
    t0 = now_us();
    if (load_split(argv[2], &slot_b) != 0) { fprintf(stderr, "load split 1 failed\n"); return 2; }
    uint64_t t_load_b = now_us() - t0;
    fprintf(stderr, "  initial load split[0]: %.1f ms\n", t_load_a / 1000.0);
    fprintf(stderr, "  initial load split[1]: %.1f ms\n", t_load_b / 1000.0);

    static unsigned char host_residual[1 << 20];  /* 1 MB scratch */
    size_t host_residual_size = 0;

    const int N_ITER = 3;
    for (int it = 0; it < N_ITER; ++it) {
        fprintf(stderr, "\n=== iter %d ===\n", it);
        host_residual_size = 0;

        uint64_t iter_start = now_us();
        uint64_t exec_us[N_SPLITS] = {0};
        uint64_t swap_us[2]        = {0};

        /* Run split 0. */
        if (sp_qnn_execute(slot_a.h,
                           (const void *const *)slot_a.in_bufs, slot_a.in_sz,
                           (void *const *)slot_a.out_bufs,      slot_a.out_sz,
                           &exec_us[0]) != SP_QNN_OK) {
            fprintf(stderr, "  split 0 exec FAILED\n"); break;
        }
        if (slot_a.residual_out_idx >= 0) {
            host_residual_size = slot_a.out_sz[slot_a.residual_out_idx];
            memcpy(host_residual, slot_a.out_bufs[slot_a.residual_out_idx], host_residual_size);
        }

        /* Pass residual to slot_b's residual_in. */
        if (slot_b.residual_in_idx >= 0
            && host_residual_size == slot_b.in_sz[slot_b.residual_in_idx]) {
            memcpy(slot_b.in_bufs[slot_b.residual_in_idx], host_residual, host_residual_size);
        }

        /* Run split 1. */
        if (sp_qnn_execute(slot_b.h,
                           (const void *const *)slot_b.in_bufs, slot_b.in_sz,
                           (void *const *)slot_b.out_bufs,      slot_b.out_sz,
                           &exec_us[1]) != SP_QNN_OK) {
            fprintf(stderr, "  split 1 exec FAILED\n"); break;
        }
        if (slot_b.residual_out_idx >= 0) {
            host_residual_size = slot_b.out_sz[slot_b.residual_out_idx];
            memcpy(host_residual, slot_b.out_bufs[slot_b.residual_out_idx], host_residual_size);
        }

        /* Swap: destroy slot_a (was split 0), load split 2 into slot_a. */
        uint64_t sw0 = now_us();
        free_split(&slot_a);
        if (load_split(argv[3], &slot_a) != 0) {
            fprintf(stderr, "  swap to split 2 FAILED\n"); break;
        }
        swap_us[0] = now_us() - sw0;

        /* Pass residual to slot_a (now split 2). */
        if (slot_a.residual_in_idx >= 0
            && host_residual_size == slot_a.in_sz[slot_a.residual_in_idx]) {
            memcpy(slot_a.in_bufs[slot_a.residual_in_idx], host_residual, host_residual_size);
        }

        /* Run split 2 (in slot_a). */
        if (sp_qnn_execute(slot_a.h,
                           (const void *const *)slot_a.in_bufs, slot_a.in_sz,
                           (void *const *)slot_a.out_bufs,      slot_a.out_sz,
                           &exec_us[2]) != SP_QNN_OK) {
            fprintf(stderr, "  split 2 exec FAILED\n"); break;
        }
        if (slot_a.residual_out_idx >= 0) {
            host_residual_size = slot_a.out_sz[slot_a.residual_out_idx];
            memcpy(host_residual, slot_a.out_bufs[slot_a.residual_out_idx], host_residual_size);
        }

        /* Swap: destroy slot_b (was split 1), load split 3 into slot_b. */
        sw0 = now_us();
        free_split(&slot_b);
        if (load_split(argv[4], &slot_b) != 0) {
            fprintf(stderr, "  swap to split 3 FAILED\n"); break;
        }
        swap_us[1] = now_us() - sw0;

        /* Pass residual to slot_b (now split 3). */
        if (slot_b.residual_in_idx >= 0
            && host_residual_size == slot_b.in_sz[slot_b.residual_in_idx]) {
            memcpy(slot_b.in_bufs[slot_b.residual_in_idx], host_residual, host_residual_size);
        }

        /* Run split 3 (in slot_b). */
        if (sp_qnn_execute(slot_b.h,
                           (const void *const *)slot_b.in_bufs, slot_b.in_sz,
                           (void *const *)slot_b.out_bufs,      slot_b.out_sz,
                           &exec_us[3]) != SP_QNN_OK) {
            fprintf(stderr, "  split 3 exec FAILED\n"); break;
        }

        uint64_t iter_total = now_us() - iter_start;

        fprintf(stderr, "  exec: %.1f %.1f %.1f %.1f ms (sum %.1f)\n",
                exec_us[0]/1000.0, exec_us[1]/1000.0, exec_us[2]/1000.0, exec_us[3]/1000.0,
                (exec_us[0]+exec_us[1]+exec_us[2]+exec_us[3])/1000.0);
        fprintf(stderr, "  swaps:    %.1f                    %.1f      ms (sum %.1f)\n",
                swap_us[0]/1000.0, swap_us[1]/1000.0, (swap_us[0]+swap_us[1])/1000.0);
        fprintf(stderr, "  iter total wall-clock: %.1f ms\n", iter_total/1000.0);
        fprintf(stderr, "  effective rate (128-tok chunk): %.1f tok/sec\n",
                128.0 * 1e6 / (double)iter_total);

        /* For next iter, slot_a now holds split 2 and slot_b holds split 3.
         * To re-run the chain we need to swap them back to 0 and 1. */
        if (it + 1 < N_ITER) {
            uint64_t reset_start = now_us();
            free_split(&slot_a);
            free_split(&slot_b);
            if (load_split(argv[1], &slot_a) != 0
                || load_split(argv[2], &slot_b) != 0) {
                fprintf(stderr, "  iter reset FAILED\n"); break;
            }
            uint64_t reset_us = now_us() - reset_start;
            fprintf(stderr, "  iter reset (full reload of splits 0,1): %.1f ms\n", reset_us/1000.0);
        }
    }

    free_split(&slot_a);
    free_split(&slot_b);
    sp_qnn_shutdown();
    return 0;
}
