// Shannon-Prime Beast Canyon: Heterogeneous MoE Orchestrator
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// The Beast Canyon Engine manages a heterogeneous compute cluster:
//   - Optane M10/M15 (16/32/64GB) as persistent model reservoir
//   - i9 AVX-512 cores as the "Shredder" (dequantization pump)
//   - Intel Xe iGPU via Vulkan/Level-Zero (Expert A)
//   - NVIDIA RTX 2060/3090/4090 via CUDA (Expert B)
//   - S22U Hexagon DSP via USB-C/ADB sidecar (optional, Prime-PE offload)
//
// The engine is designed to be hardware-flexible:
//   - Any Optane capacity (16GB → 64GB+)
//   - Any GPU combination (CUDA-only, Vulkan-only, dual, or CPU-only)
//   - Optional phone sidecar (graceful degradation if absent)
//   - Any MoE model that fits on Optane (or non-MoE with layer-sharding)
//
// Three golden rules:
//   1. Memory is the Enemy — pointer swap over memcpy, always.
//   2. Optane is RAM — treat it as address space, not filesystem.
//   3. LLC is the Finish Line — every result terminates in the 24MB Smart Cache.

#ifndef SP_BEAST_CANYON_H
#define SP_BEAST_CANYON_H

#include "sp_optane.h"
#include "sp_avx512_shredder.h"
#include "sp_hetero_sync.h"
#include "sp_shadow_steal.h"
#include "sp_thermal_throttle.h"
#include "sp_prefetch_telemetry.h"
#include "sp_shredder_crt.h"
#include "sp_sidecar_proto.h"
#include "../../core/include/shannon_prime.h"

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Engine Configuration
// ============================================================================

typedef struct {
    // --- Storage ---
    const char  *gguf_path;          // Path to GGUF model on Optane (or any NVMe)

    // --- GPU assignment ---
    int          cuda_device;        // CUDA device ID (-1 = none)
    int          vulkan_device;      // Vulkan device ID (-1 = auto-detect Intel Xe)
    bool         force_cpu_only;     // Disable all GPU dispatch

    // --- MoE routing ---
    int          n_experts_per_token; // Top-K (default: use model's value)
    bool         expert_round_robin;  // Alternate expert assignment across GPUs

    // --- Sidecar ---
    bool         enable_sidecar;     // Try to connect S22U via ADB
    int          sidecar_port;       // ADB forward port (default 9876)

    // --- Shredder ---
    int          shredder_prefetch;  // Pages to prefetch (default 8)
    size_t       staging_elements;   // Staging buffer size (0 = auto)

    // --- Memory management ---
    size_t       optane_budget;      // Max bytes to use from Optane (0 = all)
    bool         preload_hot_layers; // Pre-shred attention/router into RAM at boot

    // --- Shannon-Prime compression ---
    sp_config_t  sp_config;          // VHT2 + band config for KV cache

    // --- Shadow-Steal ---
    float        shadow_steal_tau;    // Confidence threshold (0 = disabled)
    bool         enable_shadow_steal; // Master switch for speculative UHD dispatch

    // --- Thermal ---
    float        thermal_tau_base;    // Base tau for thermal homeostat (default 0.50)

    // --- Diagnostics ---
    bool         enable_dashboard;   // Real-time ASCII dashboard
    int          dashboard_interval_ms; // Dashboard refresh rate
} sp_beast_config_t;

// Initialize with sane defaults.
void sp_beast_config_init(sp_beast_config_t *cfg);

// ============================================================================
// Ping-Pong Double Buffer
// ============================================================================
//
// Two staging buffers in LLC-resident memory. While one is being consumed
// by the GPU, the Shredder fills the other from Optane. Zero-copy pointer
// flip between cycles.

typedef struct {
    uint16_t  *buffers[2];    // Two fp16 staging areas (64-byte aligned)
    size_t     buf_size;      // Size per buffer in bytes
    size_t     buf_elements;  // Elements per buffer
    int        active;        // Which buffer the GPU is reading (0 or 1)
    int        filling;       // Which buffer the Shredder is writing (0 or 1)
    bool       fill_ready;    // True when filling buffer is complete
} sp_pingpong_t;

// ============================================================================
// Sidecar Connection (S22U via USB-C/ADB)
// ============================================================================

typedef enum {
    SP_SIDECAR_DISCONNECTED = 0,
    SP_SIDECAR_CONNECTING   = 1,
    SP_SIDECAR_ONLINE       = 2,
    SP_SIDECAR_ERROR        = 3,
} sp_sidecar_state_t;

typedef struct {
    sp_sidecar_state_t state;
    int                socket_fd;    // TCP socket via ADB port-forward
    int                port;
    uint64_t           total_offloads;
    uint64_t           total_offload_us;
    uint64_t           total_bytes_sent;
    uint64_t           total_bytes_recv;
} sp_sidecar_t;

// ============================================================================
// MoE Expert Router
// ============================================================================

typedef struct {
    int     expert_ids[SP_OPTANE_MAX_EXPERTS]; // Sorted by router logit
    float   expert_scores[SP_OPTANE_MAX_EXPERTS];
    int     n_selected;                        // Top-K
    int     gpu_assignment[SP_OPTANE_MAX_EXPERTS]; // Which GPU gets which expert
} sp_expert_routing_t;

// ============================================================================
// Beast Canyon Engine
// ============================================================================

typedef struct {
    // Sub-systems
    sp_optane_reservoir_t  reservoir;    // Optane mmap + expert pointers
    sp_shredder_t          shredder;     // AVX-512 dequantization
    sp_hetero_barrier_t    barrier;      // Cross-GPU sync
    sp_sidecar_t           sidecar;      // S22U optional sidecar

    // MoE speculation + telemetry subsystems
    sp_shadow_steal_t      shadow_steal;    // Speculative UHD dispatch
    sp_thermal_homeostat_t thermal;         // Thermal-aware throttle
    sp_prefetch_telemetry_t telemetry;      // Speculation metrics
    sp_shredder_crt_t      shredder_crt;   // Residue-aware CRT shredder

    // Ping-pong buffers (one per GPU)
    sp_pingpong_t          pingpong[2];  // [0] = GPU A, [1] = GPU B

    // KV Cache (Shannon-Prime compressed)
    sp_shadow_cache_t     *kv_cache;     // CPU-side compressed cache
    // GPU-resident caches are managed by the respective backends

    // Configuration
    sp_beast_config_t      config;

    // Current state
    int                    current_pos;  // Sequence position
    int                    current_layer; // Transformer layer
    sp_expert_routing_t    routing;      // Current token's expert selection

    // Performance counters
    uint64_t    total_tokens;
    uint64_t    total_inference_us;
    uint64_t    total_shred_us;
    uint64_t    total_gpu_us;
    uint64_t    total_barrier_us;
    uint64_t    total_sidecar_us;
    uint64_t    boot_time_us;

    // Forward pass runtime state (allocated by sp_beast_generate)
    float       *beast_k_cache;     // [n_attn_layers * n_kv_dim * max_ctx]
    float       *beast_v_cache;     // same shape
    float       *beast_scratch;     // dequant row scratch [max(n_embd, vocab_size)]
    float       *beast_x;           // residual stream [n_embd]
    float       *beast_xn;          // normed residual [n_embd]
    float       *beast_logits;      // [vocab_size]
    int          beast_max_ctx;
    int          beast_n_pos;

    // SSM (Mamba2) recurrent state — persists across tokens
    float       *beast_conv_state;  // [n_ssm_layers * ssm_inner * (conv_kernel-1)]
    float       *beast_ssm_state;   // [n_ssm_layers * dt_rank * ssm_state_size]
    int          beast_n_ssm_layers;
} sp_beast_engine_t;

// ============================================================================
// Public API — Lifecycle
// ============================================================================

// Boot the engine: mmap Optane, detect GPUs, init Shredder, connect sidecar.
int sp_beast_init(sp_beast_engine_t *engine, const sp_beast_config_t *cfg);

// Shutdown: release all resources in the correct order.
// (Level Zero before CUDA, as per Gemini's safeguard spec.)
void sp_beast_free(sp_beast_engine_t *engine);

// ============================================================================
// Public API — Inference
// ============================================================================

// Run the MoE MLP for one transformer layer.
// router_logits: [n_experts] fp32 logits from the router/gate network.
// hidden_states: [hidden_dim] fp32 input hidden states.
// output:        [hidden_dim] fp32 output (accumulated expert results).
//
// This is the core "Execution Pulse":
//   1. Oracle: CPU routes experts from router_logits
//   2. Shredder: AVX-512 dequants selected experts from Optane → staging
//   3. Dispatch: Launch on CUDA + Vulkan simultaneously
//   4. Barrier: Wait for both GPUs, pre-shred next expert during wait
//   5. Sum: AVX-512 merges results in LLC
int sp_beast_moe_forward(sp_beast_engine_t *engine,
                         const float *router_logits,
                         const float *hidden_states,
                         float *output,
                         int layer);

// Run attention for one layer (non-MoE path).
// K/V are written to the Shannon-Prime compressed cache.
int sp_beast_attention_forward(sp_beast_engine_t *engine,
                               const float *q, const float *k, const float *v,
                               float *output,
                               int layer, int pos, int kv_len);

// Full forward pass for one token (all layers).
// input_token: token ID.
// logits: [vocab_size] output logits.
int sp_beast_forward(sp_beast_engine_t *engine,
                     int input_token,
                     float *logits);

// Generate tokens autoregressively.
// prompt_tokens: [n_prompt] token IDs.
// output_tokens: [max_tokens] generated token IDs (caller-allocated).
// Returns number of tokens generated.
int sp_beast_generate(sp_beast_engine_t *engine,
                      const int *prompt_tokens, int n_prompt,
                      int *output_tokens, int max_tokens,
                      float temperature, float top_p);

// ============================================================================
// Public API — Expert Routing
// ============================================================================

// Run the router/oracle on logits and populate engine->routing.
// Assigns experts to GPUs based on hardware capabilities.
void sp_beast_route_experts(sp_beast_engine_t *engine,
                            const float *router_logits, int n_experts);

// ============================================================================
// Public API — Sidecar
// ============================================================================

// Connect to S22U sidecar via ADB port-forward.
// Returns 0 on success, -1 if sidecar not available.
int sp_beast_sidecar_connect(sp_beast_engine_t *engine);
void sp_beast_sidecar_disconnect(sp_beast_engine_t *engine);

// Offload Prime-PE transform to S22U DSP.
// Returns 0 on success, -1 if sidecar offline (CPU fallback triggered).
int sp_beast_sidecar_prime_pe(sp_beast_engine_t *engine,
                              const float *hidden_states, int dim,
                              float *pe_output);

// Ping the sidecar. Returns 0 if responsive, -1 if dead.
int sp_beast_sidecar_ping(sp_beast_engine_t *engine);

// Tell sidecar to load a draft model (path on the phone's filesystem).
int sp_beast_sidecar_load_draft(sp_beast_engine_t *engine,
                                const char *phone_gguf_path);

// Prefill the draft model on the phone with the same prompt as the host.
int sp_beast_sidecar_prefill(sp_beast_engine_t *engine,
                             const int32_t *tokens, int n_tokens);

// Draft N speculative tokens from the phone's draft model.
// current_tok = the token just decoded by the host. n_draft = how many to predict.
// Result written to *out. Returns 0 on success, -1 on error/fallback.
int sp_beast_sidecar_draft(sp_beast_engine_t *engine,
                           int32_t current_tok, int n_draft,
                           sp_sidecar_draft_result_t *out);

// Accept verified tokens / resync after speculative mismatch.
int sp_beast_sidecar_accept(sp_beast_engine_t *engine,
                            const int32_t *verified, int n_accepted);

// Reset draft model state (new conversation).
int sp_beast_sidecar_reset(sp_beast_engine_t *engine);

// ============================================================================
// Public API — Diagnostics
// ============================================================================

// Print full engine status.
void sp_beast_print_status(const sp_beast_engine_t *engine);

// Get tokens per second.
static inline double sp_beast_tok_per_sec(const sp_beast_engine_t *engine) {
    if (engine->total_inference_us == 0) return 0.0;
    return (double)engine->total_tokens / ((double)engine->total_inference_us / 1000000.0);
}

// Print real-time ASCII dashboard to stderr.
// Call in a loop from a monitoring thread, or after each token.
void sp_beast_dashboard(const sp_beast_engine_t *engine);

#ifdef __cplusplus
}
#endif

#endif // SP_BEAST_CANYON_H
