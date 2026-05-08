// Shannon-Prime Beast Canyon: Diagnostics & Micro-Benchmarking — Implementation
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "sp_diagnostics_bc.h"
#include <stdio.h>
#include <string.h>
#include <float.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <time.h>
#  include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#  ifdef _MSC_VER
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#endif

static uint64_t sp_time_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (uint64_t)(now.QuadPart * 1000000ULL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
#endif
}

// ============================================================================
// Pulse Monitor
// ============================================================================

void sp_diag_init(sp_pulse_monitor_t *mon) {
    memset(mon, 0, sizeof(*mon));
    mon->optane_stride_min = DBL_MAX;
    mon->monitoring_active = true;
}

static void update_rolling(double *history, int *idx, double value, int size) {
    history[*idx % size] = value;
    (*idx)++;
}

static double rolling_avg(const double *history, int idx, int size) {
    int count = (idx < size) ? idx : size;
    if (count == 0) return 0.0;
    double sum = 0.0;
    int start = (idx < size) ? 0 : idx - size;
    for (int i = start; i < idx && i < start + size; i++) {
        sum += history[i % size];
    }
    return sum / count;
}

void sp_diag_record_optane_stride(sp_pulse_monitor_t *mon, double us) {
    update_rolling(mon->optane_stride_us, &mon->optane_stride_idx,
                   us, SP_DIAG_HISTORY_SIZE);
    mon->optane_stride_avg = rolling_avg(mon->optane_stride_us,
                                          mon->optane_stride_idx,
                                          SP_DIAG_HISTORY_SIZE);
    if (us < mon->optane_stride_min) mon->optane_stride_min = us;
    if (us > mon->optane_stride_max) mon->optane_stride_max = us;
}

void sp_diag_record_pcie_transfer(sp_pulse_monitor_t *mon, double us) {
    update_rolling(mon->pcie_transfer_us, &mon->pcie_transfer_idx,
                   us, SP_DIAG_HISTORY_SIZE);
    mon->pcie_transfer_avg = rolling_avg(mon->pcie_transfer_us,
                                          mon->pcie_transfer_idx,
                                          SP_DIAG_HISTORY_SIZE);
}

void sp_diag_record_barrier(sp_pulse_monitor_t *mon, double us) {
    update_rolling(mon->barrier_us, &mon->barrier_idx,
                   us, SP_DIAG_HISTORY_SIZE);
    mon->barrier_avg = rolling_avg(mon->barrier_us,
                                    mon->barrier_idx,
                                    SP_DIAG_HISTORY_SIZE);
}

void sp_diag_record_shredder(sp_pulse_monitor_t *mon, double gbps) {
    update_rolling(mon->shredder_gbps, &mon->shredder_idx,
                   gbps, SP_DIAG_HISTORY_SIZE);
}

void sp_diag_record_token(sp_pulse_monitor_t *mon,
                          double oracle_us, double shred_us,
                          double dispatch_us, double barrier_us,
                          double sum_us)
{
    mon->tok_oracle_us   = oracle_us;
    mon->tok_shred_us    = shred_us;
    mon->tok_dispatch_us = dispatch_us;
    mon->tok_barrier_us  = barrier_us;
    mon->tok_sum_us      = sum_us;
    mon->tok_total_us    = oracle_us + shred_us + dispatch_us + barrier_us + sum_us;
    mon->total_tokens_monitored++;
}

// ============================================================================
// Optane Audit — "Day Zero" validation
// ============================================================================

int sp_diag_optane_audit(const sp_optane_reservoir_t *res,
                         sp_pulse_monitor_t *mon)
{
    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  OPTANE AUDIT: DAY ZERO VALIDATION           ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════╝\n\n");

    if (!res || !res->base_ptr) {
        fprintf(stderr, "[AUDIT] ERROR: No reservoir mapped\n");
        return -1;
    }

    int pass = 0;
    int fail = 0;

    // ── Test 1: Sequential 4KB stride latency ───────────────────────
    fprintf(stderr, "[AUDIT] Test 1: Sequential 4KB stride latency...\n");
    {
        const uint8_t *base = (const uint8_t *)res->data_ptr;
        uint64_t data_size = res->file_size - res->data_offset;
        volatile uint8_t sink = 0;

        // Warm-up
        for (int i = 0; i < 8; i++) {
            sink ^= base[(uint64_t)i * SP_OPTANE_PAGE_SIZE];
        }

        // Measure 64 sequential pages
        int n_pages = 64;
        uint64_t t0 = sp_time_us();
        for (int i = 0; i < n_pages; i++) {
            uint64_t off = (uint64_t)i * SP_OPTANE_PAGE_SIZE;
            if (off >= data_size) break;
            sink ^= base[off];
        }
        uint64_t t1 = sp_time_us();
        (void)sink;

        double avg_us = (double)(t1 - t0) / n_pages;
        sp_diag_record_optane_stride(mon, avg_us);

        bool ok = avg_us < 15.0;
        fprintf(stderr, "  Average: %.2f us/page  [target < 15 us]  %s\n",
                avg_us, ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;
    }

    // ── Test 2: Random 4KB page access ──────────────────────────────
    fprintf(stderr, "[AUDIT] Test 2: Random 4KB page access latency...\n");
    {
        const uint8_t *base = (const uint8_t *)res->data_ptr;
        uint64_t data_size = res->file_size - res->data_offset;
        uint64_t n_pages_total = data_size / SP_OPTANE_PAGE_SIZE;
        volatile uint8_t sink = 0;

        // Simple LCG for pseudo-random page selection
        uint64_t seed = 0xDEADBEEF;
        int n_samples = 64;

        uint64_t t0 = sp_time_us();
        for (int i = 0; i < n_samples; i++) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            uint64_t page = (seed >> 16) % n_pages_total;
            sink ^= base[page * SP_OPTANE_PAGE_SIZE];
        }
        uint64_t t1 = sp_time_us();
        (void)sink;

        double avg_us = (double)(t1 - t0) / n_samples;
        bool ok = avg_us < 20.0;
        fprintf(stderr, "  Average: %.2f us/page  [target < 20 us]  %s\n",
                avg_us, ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;
    }

    // ── Test 3: Sustained sequential bandwidth ──────────────────────
    fprintf(stderr, "[AUDIT] Test 3: Sustained sequential read bandwidth...\n");
    {
        const uint8_t *base = (const uint8_t *)res->data_ptr;
        size_t read_size = 64 * 1024 * 1024;  // 64 MB
        if (read_size > res->file_size - res->data_offset) {
            read_size = (size_t)(res->file_size - res->data_offset);
        }

        volatile uint64_t sink = 0;
        uint64_t t0 = sp_time_us();

        // Read in 64-byte (cache-line) strides
        for (size_t off = 0; off < read_size; off += 64) {
            uint64_t v;
            memcpy(&v, base + off, 8);
            sink ^= v;
        }

        uint64_t t1 = sp_time_us();
        (void)sink;

        double seconds = (double)(t1 - t0) / 1000000.0;
        double gbps = ((double)read_size / (1024.0*1024.0*1024.0)) / seconds;
        bool ok = gbps > 1.0;
        fprintf(stderr, "  Bandwidth: %.2f GB/s over %.1f MB  [target > 1 GB/s]  %s\n",
                gbps, (double)read_size / (1024.0*1024.0), ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;
    }

    // ── Test 4: DAX status ──────────────────────────────────────────
    fprintf(stderr, "[AUDIT] Test 4: DAX status...\n");
    fprintf(stderr, "  DAX: %s\n",
            res->dax_enabled ? "ENABLED (direct Optane access)" :
                               "DISABLED (page cache, still fast)");
    // DAX is nice-to-have, not a hard requirement
    pass++;

    // ── Summary ─────────────────────────────────────────────────────
    fprintf(stderr, "\n[AUDIT] Results: %d PASS, %d FAIL\n", pass, fail);
    if (fail == 0) {
        fprintf(stderr, "[AUDIT] === OPTANE RESERVOIR: VALIDATED ===\n\n");
    } else {
        fprintf(stderr, "[AUDIT] === WARNING: %d tests failed — check M.2 slot assignment ===\n\n",
                fail);
    }

    return fail;
}

// ============================================================================
// Topology Discovery
// ============================================================================

int sp_diag_discover_topology(sp_topology_report_t *topo,
                              const sp_beast_engine_t *engine)
{
    memset(topo, 0, sizeof(*topo));

    // CPU info
#if defined(__x86_64__) || defined(_M_X64)
    {
        int info[4];
        // Brand string (leaves 0x80000002-4)
        char brand[49] = {0};
        for (int leaf = 0x80000002; leaf <= 0x80000004; leaf++) {
#ifdef _MSC_VER
            __cpuid(info, leaf);
#else
            __cpuid_count(leaf, 0, info[0], info[1], info[2], info[3]);
#endif
            memcpy(brand + (leaf - 0x80000002) * 16, info, 16);
        }
        brand[48] = '\0';
        // Trim leading spaces
        char *p = brand;
        while (*p == ' ') p++;
        strncpy(topo->cpu_name, p, sizeof(topo->cpu_name) - 1);

        // Check AVX-512
#ifdef _MSC_VER
        __cpuidex(info, 7, 0);
#else
        __cpuid_count(7, 0, info[0], info[1], info[2], info[3]);
#endif
        topo->has_avx512 = (info[1] & (1 << 16)) != 0;
    }
#endif

    // System info
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    topo->cpu_cores = si.dwNumberOfProcessors; // Logical processors
    topo->cpu_threads = si.dwNumberOfProcessors;
#else
    topo->cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);
    topo->cpu_threads = topo->cpu_cores;
#endif

    // Optane info from reservoir
    if (engine) {
        strncpy(topo->optane_path, engine->config.gguf_path ? engine->config.gguf_path : "",
                sizeof(topo->optane_path) - 1);
        topo->optane_capacity = engine->reservoir.file_size;
        topo->optane_dax = engine->reservoir.dax_enabled;
        topo->optane_measured_latency_us =
            sp_optane_measure_stride_latency(&engine->reservoir);

        // GPU info from barrier
        for (int i = 0; i < engine->barrier.n_gpus && i < 2; i++) {
            const sp_gpu_device_t *g = &engine->barrier.gpu[i];
            char *name = (i == 0) ? topo->gpu0_name : topo->gpu1_name;
            char *type = (i == 0) ? topo->gpu0_type : topo->gpu1_type;
            uint64_t *vram = (i == 0) ? &topo->gpu0_vram : &topo->gpu1_vram;
            strncpy(name, g->name, 127);
            *vram = g->vram_bytes;
            switch (g->type) {
            case SP_GPU_CUDA:       strncpy(type, "CUDA", 31); break;
            case SP_GPU_VULKAN:     strncpy(type, "Vulkan", 31); break;
            case SP_GPU_LEVEL_ZERO: strncpy(type, "Level Zero", 31); break;
            default:                strncpy(type, "CPU", 31); break;
            }
        }

        topo->sidecar_detected = (engine->sidecar.state == SP_SIDECAR_ONLINE);
    }

    return 0;
}

void sp_diag_print_topology(const sp_topology_report_t *topo) {
    fprintf(stderr, "\n=== HARDWARE TOPOLOGY ===\n");
    fprintf(stderr, "CPU:    %s (%d cores, AVX-512=%s)\n",
            topo->cpu_name, topo->cpu_cores,
            topo->has_avx512 ? "YES" : "no");
    fprintf(stderr, "Optane: %.1f GB%s (%.1f us stride)\n",
            (double)topo->optane_capacity / (1024.0*1024.0*1024.0),
            topo->optane_dax ? " DAX" : "",
            topo->optane_measured_latency_us);
    if (topo->gpu0_name[0]) {
        fprintf(stderr, "GPU 0:  %s [%s] (%.1f GB)\n",
                topo->gpu0_name, topo->gpu0_type,
                (double)topo->gpu0_vram / (1024.0*1024.0*1024.0));
    }
    if (topo->gpu1_name[0]) {
        fprintf(stderr, "GPU 1:  %s [%s] (%.1f GB)\n",
                topo->gpu1_name, topo->gpu1_type,
                (double)topo->gpu1_vram / (1024.0*1024.0*1024.0));
    }
    fprintf(stderr, "Sidecar: %s\n", topo->sidecar_detected ? "S22U ONLINE" : "none");
    fprintf(stderr, "=========================\n\n");
}

// ============================================================================
// Dashboard
// ============================================================================

void sp_diag_print_dashboard(const sp_pulse_monitor_t *mon,
                             const sp_beast_engine_t *engine)
{
    double tok_s = sp_beast_tok_per_sec(engine);
    double shred_gbps = sp_shredder_throughput_gbps(&engine->shredder);

    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  BEAST CANYON: REAL-TIME DASHBOARD                           ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║                                                              ║\n");
    fprintf(stderr, "║  [ OPTANE ] ──────> [ SHREDDER ] ──────> [ LLC ]             ║\n");
    fprintf(stderr, "║  %.1f us/pg          %.1f GB/s             24 MB             ║\n",
            mon->optane_stride_avg, shred_gbps);
    fprintf(stderr, "║                                            │                 ║\n");
    fprintf(stderr, "║                                    ┌───────┴───────┐         ║\n");
    fprintf(stderr, "║                                    ▼               ▼         ║\n");

    if (engine->barrier.n_gpus >= 2) {
        fprintf(stderr, "║                              [ GPU 0 ]       [ GPU 1 ]      ║\n");
        fprintf(stderr, "║                              %-14s %-14s    ║\n",
                engine->barrier.gpu[0].name, engine->barrier.gpu[1].name);
    } else if (engine->barrier.n_gpus == 1) {
        fprintf(stderr, "║                              [ GPU 0 ]       [ CPU ]        ║\n");
        fprintf(stderr, "║                              %-14s fallback        ║\n",
                engine->barrier.gpu[0].name);
    } else {
        fprintf(stderr, "║                              [ CPU-ONLY MODE ]              ║\n");
        fprintf(stderr, "║                                                              ║\n");
    }

    fprintf(stderr, "║                                    │               │         ║\n");
    fprintf(stderr, "║                                    └───────┬───────┘         ║\n");
    fprintf(stderr, "║                                            ▼                 ║\n");
    fprintf(stderr, "║                                    [ BARRIER: %.0f us ]       ║\n",
            mon->barrier_avg);
    fprintf(stderr, "║                                            │                 ║\n");
    fprintf(stderr, "║                                            ▼                 ║\n");
    fprintf(stderr, "║                                     [ SUM (AVX-512) ]        ║\n");
    fprintf(stderr, "║                                            │                 ║\n");
    fprintf(stderr, "║                                            ▼                 ║\n");
    fprintf(stderr, "║                                     [ TOKEN: %.1f t/s ]      ║\n", tok_s);
    fprintf(stderr, "║                                                              ║\n");

    if (engine->sidecar.state == SP_SIDECAR_ONLINE) {
        fprintf(stderr, "║  [ S22U SIDECAR: ONLINE — Prime-PE offloaded ]              ║\n");
    }

    fprintf(stderr, "╠══════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Per-Token Breakdown:                                        ║\n");
    fprintf(stderr, "║    Oracle:   %7.1f us                                      ║\n", mon->tok_oracle_us);
    fprintf(stderr, "║    Shredder: %7.1f us                                      ║\n", mon->tok_shred_us);
    fprintf(stderr, "║    Dispatch: %7.1f us                                      ║\n", mon->tok_dispatch_us);
    fprintf(stderr, "║    Barrier:  %7.1f us                                      ║\n", mon->tok_barrier_us);
    fprintf(stderr, "║    Sum:      %7.1f us                                      ║\n", mon->tok_sum_us);
    fprintf(stderr, "║    Total:    %7.1f us (%.1f tok/s)                         ║\n",
            mon->tok_total_us, mon->tok_total_us > 0 ? 1000000.0 / mon->tok_total_us : 0.0);
    fprintf(stderr, "╚══════════════════════════════════════════════════════════════╝\n\n");
}

void sp_diag_print_inline(const sp_pulse_monitor_t *mon,
                          const sp_beast_engine_t *engine)
{
    double tok_s = sp_beast_tok_per_sec(engine);
    fprintf(stderr, "\r[BC] %.1f t/s | optane=%.1fus | shred=%.1fGB/s | barrier=%.0fus | tok=%llu  ",
            tok_s, mon->optane_stride_avg,
            rolling_avg(mon->shredder_gbps, mon->shredder_idx, SP_DIAG_HISTORY_SIZE),
            mon->barrier_avg,
            (unsigned long long)mon->total_tokens_monitored);
    fflush(stderr);
}
