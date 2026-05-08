// Shannon-Prime Beast Canyon: Diagnostics & Micro-Benchmarking
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Pulse Monitor: real-time instrumentation for the Beast Canyon engine.
// Three critical metrics (from Gemini's safeguard spec):
//   1. Optane-to-LLC Stride: 4KB page fetch latency (target: < 15us)
//   2. PCIe Saturation: expert weight transfer to VRAM
//   3. Barrier Wait Time: CPU idle between GPU dispatches
//
// Also provides:
//   - Optane audit script (validates CPU-attached lane latency)
//   - Hardware topology report
//   - Per-token timing breakdown
//   - Throughput histograms

#ifndef SP_DIAGNOSTICS_BC_H
#define SP_DIAGNOSTICS_BC_H

#include "sp_beast_canyon.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Pulse Monitor
// ============================================================================

#define SP_DIAG_HISTORY_SIZE  256   // Rolling window for latency tracking

typedef struct {
    // Optane stride latency (microseconds per 4KB page)
    double  optane_stride_us[SP_DIAG_HISTORY_SIZE];
    int     optane_stride_idx;
    double  optane_stride_avg;
    double  optane_stride_min;
    double  optane_stride_max;

    // PCIe transfer time (microseconds per expert)
    double  pcie_transfer_us[SP_DIAG_HISTORY_SIZE];
    int     pcie_transfer_idx;
    double  pcie_transfer_avg;

    // Barrier wait time (microseconds)
    double  barrier_us[SP_DIAG_HISTORY_SIZE];
    int     barrier_idx;
    double  barrier_avg;

    // Shredder throughput (GB/s, rolling)
    double  shredder_gbps[SP_DIAG_HISTORY_SIZE];
    int     shredder_idx;

    // Per-token timing breakdown (microseconds)
    double  tok_oracle_us;     // Router/expert selection
    double  tok_shred_us;      // AVX-512 dequantization
    double  tok_dispatch_us;   // GPU kernel launch
    double  tok_barrier_us;    // Barrier wait
    double  tok_sum_us;        // Result merging
    double  tok_total_us;

    // Token counter
    uint64_t total_tokens_monitored;
    bool     monitoring_active;
} sp_pulse_monitor_t;

// ============================================================================
// Hardware Topology Report
// ============================================================================

typedef struct {
    // CPU
    char    cpu_name[128];
    int     cpu_cores;
    int     cpu_threads;
    bool    has_avx512;
    uint64_t llc_size_bytes;    // L3/LLC size

    // Optane
    char    optane_path[256];
    uint64_t optane_capacity;
    bool    optane_dax;
    double  optane_measured_latency_us;

    // GPU 0 (typically NVIDIA)
    char    gpu0_name[128];
    uint64_t gpu0_vram;
    char    gpu0_type[32];      // "CUDA", "Vulkan", etc.

    // GPU 1 (typically Intel Xe)
    char    gpu1_name[128];
    uint64_t gpu1_vram;
    char    gpu1_type[32];

    // Sidecar
    bool    sidecar_detected;
    char    sidecar_device[64];
} sp_topology_report_t;

// ============================================================================
// Public API
// ============================================================================

// Initialize the pulse monitor.
void sp_diag_init(sp_pulse_monitor_t *mon);

// Record a single measurement for each metric.
void sp_diag_record_optane_stride(sp_pulse_monitor_t *mon, double us);
void sp_diag_record_pcie_transfer(sp_pulse_monitor_t *mon, double us);
void sp_diag_record_barrier(sp_pulse_monitor_t *mon, double us);
void sp_diag_record_shredder(sp_pulse_monitor_t *mon, double gbps);

// Record per-token timing breakdown.
void sp_diag_record_token(sp_pulse_monitor_t *mon,
                          double oracle_us, double shred_us,
                          double dispatch_us, double barrier_us,
                          double sum_us);

// --- Optane Audit ---

// Run the "Day Zero" Optane audit. Tests:
//   1. 4KB sequential stride latency
//   2. Random 4KB page access latency
//   3. Sustained sequential read bandwidth
//   4. DAX status check
// Returns 0 if all targets met.
int sp_diag_optane_audit(const sp_optane_reservoir_t *res,
                         sp_pulse_monitor_t *mon);

// --- Topology ---

// Discover hardware topology and populate report.
int sp_diag_discover_topology(sp_topology_report_t *topo,
                              const sp_beast_engine_t *engine);

// Print topology report.
void sp_diag_print_topology(const sp_topology_report_t *topo);

// --- Dashboard ---

// Print the full ASCII dashboard (one-shot, for terminals).
void sp_diag_print_dashboard(const sp_pulse_monitor_t *mon,
                             const sp_beast_engine_t *engine);

// Print a compact one-line status (for inline updates).
void sp_diag_print_inline(const sp_pulse_monitor_t *mon,
                          const sp_beast_engine_t *engine);

#ifdef __cplusplus
}
#endif

#endif // SP_DIAGNOSTICS_BC_H
