// Standalone smoke + micro-bench for sp_pointer_view.
//
//   sp_pointer_view_test <gguf>
//
// Reports:
//   - open time + #tensors + file size
//   - first/last tensor names + first 4 bytes of their data (sanity)
//   - hash lookup: per-call latency over 1e6 lookups (random tensor each)
//   - linear scan: per-call latency for the same workload (apples-to-apples
//     vs sp_optane_find_tensor's strcmp loop)
//   - 4K random-page read latency through the mmap (mean / p50 / p99) — a
//     real read of the same pages routed-MoE expert dispatch would touch

#include "sp_pointer_view.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#  include <windows.h>
static double now_s(void) {
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#  include <time.h>
static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

// Linear-scan reference (matches the existing sp_optane_find_tensor shape).
static const sp_pv_tensor_t *linear_find(const sp_pv_tensor_t *arr,
                                          uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; ++i)
        if (strcmp(arr[i].name, name) == 0) return &arr[i];
    return NULL;
}

// percentile of a sorted double[] array
static int dcmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y);
}
static double pct(double *xs, size_t n, double p) {
    size_t k = (size_t)((n - 1) * p);
    return xs[k];
}

struct gather_ctx { const sp_pv_tensor_t **a; uint32_t i; };
static int gather_cb(const sp_pv_tensor_t *t, void *u) {
    struct gather_ctx *c = (struct gather_ctx *)u;
    c->a[c->i++] = t;
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: %s <gguf>\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];

    double t0 = now_s();
    sp_pointer_view_t *pv = sp_pv_open(path);
    double open_s = now_s() - t0;
    if (!pv) {
        fprintf(stderr, "sp_pv_open(%s) failed\n", path);
        return 1;
    }

    uint32_t n = sp_pv_tensor_count(pv);
    printf("[pv] open OK in %.3f s  file=%.2f GiB  n_tensors=%u\n",
           open_s, sp_pv_file_size(pv) / (1024.0 * 1024.0 * 1024.0), n);

    // Collect all names into a flat array for benching.
    const sp_pv_tensor_t **all =
        (const sp_pv_tensor_t **)malloc(sizeof(*all) * n);
    struct gather_ctx cx; cx.a = all; cx.i = 0;
    sp_pv_iterate(pv, gather_cb, &cx);
    printf("[pv] first: %-48s dtype=%u  shape=%llu,%llu  bytes=%llu\n",
           all[0]->name, all[0]->dtype,
           (unsigned long long)all[0]->ne[0],
           (unsigned long long)all[0]->ne[1],
           (unsigned long long)all[0]->n_bytes);
    printf("[pv] last : %-48s dtype=%u  shape=%llu,%llu  bytes=%llu\n",
           all[n-1]->name, all[n-1]->dtype,
           (unsigned long long)all[n-1]->ne[0],
           (unsigned long long)all[n-1]->ne[1],
           (unsigned long long)all[n-1]->n_bytes);

    // Sanity: read first 16 bytes of two specific tensors via the pointer.
    {
        const sp_pv_tensor_t *te = sp_pv_get(pv, "token_embd.weight");
        if (te) {
            uint8_t *p = (uint8_t *)te->ptr;
            printf("[pv] token_embd.weight first 8 bytes: %02x %02x %02x %02x %02x %02x %02x %02x\n",
                   p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
        }
    }

    // Flatten name strings for the benches.
    char **names = (char **)malloc(sizeof(char *) * n);
    for (uint32_t i = 0; i < n; ++i) names[i] = (char *)all[i]->name;

    // ----- bench hash lookup vs linear scan ---------------------------
    const int trials = 1000000;
    unsigned r = 1u;
    double h0 = now_s();
    uint64_t hsum = 0;
    for (int i = 0; i < trials; ++i) {
        r = r * 1103515245u + 12345u;
        const char *q = names[(r >> 8) % n];
        const sp_pv_tensor_t *t = sp_pv_get(pv, q);
        hsum += (uint64_t)(t != NULL);
    }
    double h_us = (now_s() - h0) * 1e6 / trials;

    const int lin_trials = (n < 8000) ? trials : (n < 32000 ? 100000 : 10000);
    double l0 = now_s();
    uint64_t lsum = 0;
    r = 1u;
    for (int i = 0; i < lin_trials; ++i) {
        r = r * 1103515245u + 12345u;
        const char *q = names[(r >> 8) % n];
        const sp_pv_tensor_t *t = linear_find(all[0], n, q);
        lsum += (uint64_t)(t != NULL);
    }
    double l_us = (now_s() - l0) * 1e6 / lin_trials;

    printf("\n[bench] hash lookup    : %8.3f us / call  (1e6 calls, hit_rate=%.0f%%)\n",
           h_us, 100.0 * (double)hsum / trials);
    printf("[bench] linear scan    : %8.3f us / call  (%d calls, hit_rate=%.0f%%)\n",
           l_us, lin_trials, 100.0 * (double)lsum / lin_trials);
    printf("[bench] hash speedup   : %8.1fx faster\n", l_us / h_us);

    // ----- bench 4K random page-read latency through the mmap --------
    // Picks random pages within the data section and reads one byte from
    // each (forcing a fault if not paged in). Off is bounded by the
    // size of the data section, NOT the whole file — base points at
    // the data section, not at the file start.
    void *data = sp_pv_data_section(pv);
    size_t fsz = sp_pv_file_size(pv);
    uint8_t *base = (uint8_t *)data;
    size_t hdr_bytes = (size_t)(base - (uint8_t *)/*file base*/((uint8_t *)data - 0));
    // Available bytes for random reads via `base[off]`:
    //   the mmap extends from file_base..file_base+fsz; base is inside
    //   it at offset (base - file_base). Max safe off is (fsz - (base -
    //   file_base) - 4096). We don't have file_base exposed, but
    //   sp_pv_data_section + n_bytes of the last tensor gives an upper
    //   bound; safer: just clamp to the last tensor's end.
    size_t avail = 0;
    {
        // Compute the byte just past the last tensor.
        uint64_t end_off = 0;
        for (uint32_t i = 0; i < n; ++i) {
            uint64_t e = all[i]->offset + all[i]->n_bytes;
            if (e > end_off) end_off = e;
        }
        avail = (size_t)end_off;
        if (avail > 4096) avail -= 4096;
    }
    (void)hdr_bytes;

    // Per-page latency. Run 10k page faults.
    const int P = 10000;
    double *lats = (double *)malloc(sizeof(double) * P);
    uint64_t sink = 0;
    r = 42u;
    for (int i = 0; i < P; ++i) {
        r = r * 1103515245u + 12345u;
        size_t off = (size_t)(((uint64_t)r << 12) % avail);
        double a = now_s();
        sink += (uint64_t)base[off];
        lats[i] = (now_s() - a) * 1e6;  // us
    }
    qsort(lats, P, sizeof(double), dcmp);
    double mean = 0; for (int i = 0; i < P; ++i) mean += lats[i]; mean /= P;
    printf("\n[bench] 4K random read through mmap (%d page faults):\n", P);
    printf("           min  =%8.3f us\n", lats[0]);
    printf("           p50  =%8.3f us\n", pct(lats, P, 0.50));
    printf("           p90  =%8.3f us\n", pct(lats, P, 0.90));
    printf("           p99  =%8.3f us\n", pct(lats, P, 0.99));
    printf("           max  =%8.3f us\n", lats[P-1]);
    printf("           mean =%8.3f us\n", mean);
    printf("[bench] (sink=%llu)\n", (unsigned long long)(sink + hsum + lsum));

    free(lats);
    free(names);
    free(all);
    sp_pv_close(pv);
    return 0;
}
