// Shannon-Prime Beast Canyon: Heterogeneous MoE Orchestrator — Implementation
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "sp_beast_canyon.h"
#include "sp_shredder_crt.h"
#include "sp_level_zero.h"
#include "../crt/sp_crt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#  pragma comment(lib, "ws2_32.lib")
#else
#  include <time.h>
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#endif

// ============================================================================
// Timing
// ============================================================================

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
// Config defaults
// ============================================================================

void sp_beast_config_init(sp_beast_config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->cuda_device = -1;       // Auto-detect
    cfg->vulkan_device = -1;     // Auto-detect
    cfg->force_cpu_only = false;
    cfg->n_experts_per_token = 0; // Use model's default
    cfg->expert_round_robin = true;
    cfg->enable_sidecar = false;
    cfg->sidecar_port = 9876;
    cfg->shredder_prefetch = 8;
    cfg->staging_elements = 0;   // Auto-size at init
    cfg->optane_budget = 0;      // Use all available
    cfg->preload_hot_layers = false;
    cfg->enable_dashboard = true;
    cfg->dashboard_interval_ms = 500;

    // Shannon-Prime defaults
    sp_config_init(&cfg->sp_config, 128, 32, 8); // head_dim=128, n_layers=32, n_kv=8
}

// ============================================================================
// Ping-pong buffer management
// ============================================================================

static int sp_pingpong_init(sp_pingpong_t *pp, size_t elements) {
    pp->buf_elements = elements;
    pp->buf_size = elements * sizeof(uint16_t);
    pp->active = 0;
    pp->filling = 1;
    pp->fill_ready = false;

    for (int i = 0; i < 2; i++) {
#ifdef _WIN32
        pp->buffers[i] = (uint16_t *)_aligned_malloc(pp->buf_size, 64);
#else
        void *ptr = NULL;
        if (posix_memalign(&ptr, 64, pp->buf_size) != 0) ptr = NULL;
        pp->buffers[i] = (uint16_t *)ptr;
#endif
        if (!pp->buffers[i]) {
            fprintf(stderr, "[sp-beast] ERROR: failed to allocate ping-pong buffer %d\n", i);
            return -1;
        }
        // Pre-fault pages
        memset(pp->buffers[i], 0, pp->buf_size);
    }

    return 0;
}

static void sp_pingpong_free(sp_pingpong_t *pp) {
    for (int i = 0; i < 2; i++) {
        if (pp->buffers[i]) {
#ifdef _WIN32
            _aligned_free(pp->buffers[i]);
#else
            free(pp->buffers[i]);
#endif
            pp->buffers[i] = NULL;
        }
    }
}

// Flip: the buffer that was being filled becomes active for GPU,
//       and vice versa. This is the "zero-copy pointer swap".
static inline void sp_pingpong_flip(sp_pingpong_t *pp) {
    pp->active  = 1 - pp->active;
    pp->filling = 1 - pp->filling;
    pp->fill_ready = false;
}

// ============================================================================
// Boot sequence
// ============================================================================

int sp_beast_init(sp_beast_engine_t *engine, const sp_beast_config_t *cfg) {
    memset(engine, 0, sizeof(*engine));
    engine->config = *cfg;
    uint64_t t0 = sp_time_us();

    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  SHANNON-PRIME: BEAST CANYON ENGINE          ║\n");
    fprintf(stderr, "║  Heterogeneous MoE Orchestrator              ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════╝\n\n");

    // ── Stage 0: Probe hardware, pick a run mode ────────────────────
    engine->hw_caps  = sp_probe_hardware();
    engine->run_mode = sp_select_mode(&engine->hw_caps);
    sp_log_mode(engine->run_mode, &engine->hw_caps);

    // ── Stage 1: Map the Reservoir ──────────────────────────────────
    fprintf(stderr, "[BOOT] Stage 1: Mapping Optane reservoir...\n");
    // LEGACY mode → no Optane available; force RAM-tier load so the engine
    // still comes up. The caller's cfg->gguf_path remains the source.
    sp_optane_tier_t requested =
        (engine->run_mode == SP_MODE_LEGACY) ? SP_OPTANE_TIER_RAM
                                              : SP_OPTANE_TIER_UNKNOWN;
    int rc = sp_optane_init_tier(&engine->reservoir, cfg->gguf_path, requested);
    if (rc != 0) {
        fprintf(stderr, "[BOOT] FATAL: reservoir load failed (rc=%d)\n", rc);
        return rc;
    }

    // ── Stage 2: Initialize the Shredder ────────────────────────────
    fprintf(stderr, "[BOOT] Stage 2: Initializing AVX-512 Shredder...\n");
    sp_shredder_config_t shred_cfg;
    sp_shredder_config_init(&shred_cfg);
    shred_cfg.prefetch_pages = cfg->shredder_prefetch;

    // Auto-size staging: enough for one expert's largest tensor
    size_t staging_elems = cfg->staging_elements;
    if (staging_elems == 0) {
        // Find the largest expert projection tensor.
        // Qwen3.6+ uses fused expert tensors (ffn_down_exps) rather than
        // per-expert gate/up/down — most expert slots will be empty.
        // Guard against NULL pointers and corrupted sizes.
        if (engine->reservoir.is_moe) {
            for (int e = 0; e < engine->reservoir.n_experts; e++) {
                const sp_optane_expert_t *exp = &engine->reservoir.experts[e];
                if (!exp->gate_proj) continue;
                size_t nbytes = exp->gate_proj->n_bytes;
                // Sanity: reject sizes > 2 GB (corrupt parse artifact)
                if (nbytes == 0 || nbytes > (size_t)2 * 1024 * 1024 * 1024) continue;
                if (nbytes > staging_elems * 2) {
                    // Estimate elements from bytes (Q4_0: 32 elems per 18 bytes)
                    staging_elems = nbytes * 32 / 18;
                }
            }
        }
        // Fallback: enough for a 14336-wide intermediate × 4096 hidden
        if (staging_elems == 0) {
            staging_elems = 14336 * 4096;
        }
        // Double for ping-pong
        staging_elems *= 2;
    }

    rc = sp_shredder_init(&engine->shredder, &shred_cfg, staging_elems);
    if (rc != 0) {
        sp_optane_free(&engine->reservoir);
        return rc;
    }

    // ── Stage 3: Detect and initialize GPUs ─────────────────────────
    fprintf(stderr, "[BOOT] Stage 3: Detecting GPU hardware...\n");
    sp_hetero_barrier_init(&engine->barrier);

    if (!cfg->force_cpu_only) {
        // Try auto-detection first — finds CUDA discrete + Vulkan integrated
        int n_detected = sp_hetero_detect_gpus(&engine->barrier);

        // If auto-detect didn't find GPUs, allow manual override
        if (n_detected == 0) {
            if (cfg->cuda_device >= 0) {
                sp_hetero_add_gpu(&engine->barrier, SP_GPU_CUDA, cfg->cuda_device);
            }
            if (cfg->vulkan_device >= 0) {
                sp_hetero_add_gpu(&engine->barrier, SP_GPU_VULKAN, cfg->vulkan_device);
            }
        }
    } else {
        fprintf(stderr, "[BOOT] CPU-only mode (GPUs disabled)\n");
    }

    // ── Stage 4: Initialize ping-pong buffers ───────────────────────
    fprintf(stderr, "[BOOT] Stage 4: Initializing ping-pong buffers...\n");
    size_t pp_elements = staging_elems / 2;  // Each half gets half the staging
    for (int i = 0; i < 2; i++) {
        rc = sp_pingpong_init(&engine->pingpong[i], pp_elements);
        if (rc != 0) {
            fprintf(stderr, "[BOOT] WARNING: ping-pong buffer %d failed, degraded mode\n", i);
        }
    }

    // ── Stage 5: Connect sidecar (optional) ─────────────────────────
    engine->sidecar.state = SP_SIDECAR_DISCONNECTED;
    // Boot-probe found the phone → opportunistically enable sidecar even if
    // the caller left enable_sidecar=false. They can still force it off via
    // SP_RUN_MODE=DESKTOP_SOLO (which makes hw_caps.has_sidecar irrelevant).
    bool want_sidecar = cfg->enable_sidecar ||
                        (engine->run_mode == SP_MODE_FULL_PULSE);
    if (want_sidecar) {
        fprintf(stderr, "[BOOT] Stage 5: Connecting S22U sidecar...\n");
        sp_beast_sidecar_connect(engine);
    } else {
        fprintf(stderr, "[BOOT] Stage 5: Sidecar disabled (mode=%s)\n",
                sp_mode_name(engine->run_mode));
    }

    // ── Stage 5b: Initialize Shadow-Steal (Level Zero UHD dispatch) ──
    {
        // Shadow-steal uses L0 to speculatively pre-compute MoE experts
        // on the Intel UHD iGPU via Ring Bus USM.  Falls back to stub mode
        // (instant-ready) if ze_loader.dll is absent or no iGPU detected.
        size_t expert_dim = (engine->reservoir.n_embd > 0)
                          ? (size_t)engine->reservoir.n_embd
                          : 4096;
        float tau = 0.7f;  // Default speculation threshold
        int ss_rc = sp_shadow_steal_init(&engine->shadow_steal, expert_dim, tau);
        if (ss_rc == 0) {
            fprintf(stderr, "[BOOT] Shadow-Steal: initialized (expert_dim=%zu, tau=%.2f)\n",
                    expert_dim, tau);

            // If L0 found the UHD and barrier has no GPUs yet, register it.
            // The UHD is our M2 ring target via Ring Bus USM.
            sp_l0_t *l0 = (sp_l0_t *)engine->shadow_steal.l0_command_queue;
            if (l0 && l0->loaded && l0->device_found &&
                engine->barrier.n_gpus < 2)
            {
                // Register CPU as GPU[0] (M1 ring — AVX-512 modular matmul)
                if (engine->barrier.n_gpus == 0) {
                    int cpu_idx = sp_hetero_add_gpu(&engine->barrier,
                                                    SP_GPU_CPU_ONLY, 0);
                    if (cpu_idx >= 0) {
                        snprintf(engine->barrier.gpu[cpu_idx].name,
                                 sizeof(engine->barrier.gpu[cpu_idx].name),
                                 "i9-11900KB AVX-512 (M1 Mersenne)");
                        fprintf(stderr, "[BOOT] Registered CPU as M1 Primary Pulse\n");
                    }
                }
                // Register UHD as GPU[1] (M2 ring — Level Zero USM)
                int l0_idx = sp_hetero_add_gpu(&engine->barrier,
                                               SP_GPU_LEVEL_ZERO, 0);
                if (l0_idx >= 0) {
                    snprintf(engine->barrier.gpu[l0_idx].name,
                             sizeof(engine->barrier.gpu[l0_idx].name),
                             "Intel UHD 750 (M2 via L0 Ring Bus)");
                    engine->barrier.gpu[l0_idx].context = l0;
                    engine->barrier.gpu[l0_idx].command_list = l0;
                    fprintf(stderr, "[BOOT] Registered UHD 750 as M2 Shadow Steal\n");
                }
            }
        } else {
            fprintf(stderr, "[BOOT] Shadow-Steal: init failed (rc=%d) — speculation disabled\n", ss_rc);
        }
    }

    // ── Stage 6: Initialize KV cache ────────────────────────────────
    fprintf(stderr, "[BOOT] Stage 6: Initializing Shannon-Prime KV cache...\n");
    // The KV cache uses the model's actual dimensions
    if (engine->reservoir.n_layer > 0 && engine->reservoir.head_dim > 0) {
        sp_config_t kv_cfg = cfg->sp_config;
        kv_cfg.head_dim = engine->reservoir.head_dim;
        kv_cfg.n_layers = engine->reservoir.n_layer;
        kv_cfg.n_heads_kv = engine->reservoir.n_head_kv;
        // KV cache allocation happens at first inference (lazy)
        fprintf(stderr, "[BOOT] KV cache: %u layers × %u heads × hd=%u (lazy alloc)\n",
                kv_cfg.n_layers, kv_cfg.n_heads_kv, kv_cfg.head_dim);
    }

    // ── Boot complete ───────────────────────────────────────────────
    engine->boot_time_us = sp_time_us() - t0;

    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  BEAST CANYON ENGINE: ONLINE                 ║\n");
    fprintf(stderr, "║  Boot time: %8.2f ms                       ║\n",
            (double)engine->boot_time_us / 1000.0);
    fprintf(stderr, "║  Reservoir: %.1f MB (%s)%*s║\n",
            (double)engine->reservoir.file_size / (1024.0*1024.0),
            engine->reservoir.architecture,
            (int)(20 - strlen(engine->reservoir.architecture)), "");
    fprintf(stderr, "║  GPUs:      %d configured                     ║\n",
            engine->barrier.n_gpus);
    fprintf(stderr, "║  Sidecar:   %s                       ║\n",
            engine->sidecar.state == SP_SIDECAR_ONLINE ? "ONLINE " :
            engine->sidecar.state == SP_SIDECAR_ERROR  ? "ERROR  " : "OFFLINE");
    fprintf(stderr, "╚══════════════════════════════════════════════╝\n\n");

    return 0;
}

void sp_beast_free(sp_beast_engine_t *engine) {
    fprintf(stderr, "[sp-beast] Shutting down Beast Canyon engine...\n");

    // Order matters: Level Zero/Vulkan before CUDA (per safeguard spec)
    sp_shadow_steal_log_efficiency(&engine->shadow_steal);
    sp_shadow_steal_free(&engine->shadow_steal);

    sp_beast_sidecar_disconnect(engine);

    // Free ping-pong buffers
    for (int i = 0; i < 2; i++) {
        sp_pingpong_free(&engine->pingpong[i]);
    }

    sp_hetero_barrier_free(&engine->barrier);
    sp_shredder_free(&engine->shredder);
    sp_optane_free(&engine->reservoir);

    fprintf(stderr, "[sp-beast] Engine shutdown complete.\n");
}

// ============================================================================
// Expert Routing (The Oracle)
// ============================================================================

void sp_beast_route_experts(sp_beast_engine_t *engine,
                            const float *router_logits, int n_experts)
{
    sp_expert_routing_t *r = &engine->routing;
    int top_k = engine->config.n_experts_per_token;
    if (top_k <= 0) top_k = engine->reservoir.n_experts_per_token;
    if (top_k <= 0) top_k = 2;  // Default top-2
    if (top_k > n_experts) top_k = n_experts;

    // Softmax over router logits (for scores)
    float max_logit = -1e30f;
    for (int i = 0; i < n_experts; i++) {
        if (router_logits[i] > max_logit) max_logit = router_logits[i];
    }
    float sum_exp = 0.0f;
    float exp_vals[SP_OPTANE_MAX_EXPERTS];
    for (int i = 0; i < n_experts; i++) {
        exp_vals[i] = expf(router_logits[i] - max_logit);
        sum_exp += exp_vals[i];
    }

    // Top-K selection (simple partial sort)
    bool selected[SP_OPTANE_MAX_EXPERTS];
    memset(selected, 0, sizeof(selected));

    r->n_selected = top_k;
    for (int k = 0; k < top_k; k++) {
        int best = -1;
        float best_score = -1e30f;
        for (int i = 0; i < n_experts; i++) {
            if (!selected[i] && exp_vals[i] / sum_exp > best_score) {
                best = i;
                best_score = exp_vals[i] / sum_exp;
            }
        }
        r->expert_ids[k] = best;
        r->expert_scores[k] = best_score;
        selected[best] = true;

        // GPU assignment: round-robin across available GPUs
        if (engine->barrier.n_gpus >= 2 && engine->config.expert_round_robin) {
            r->gpu_assignment[k] = k % engine->barrier.n_gpus;
        } else if (engine->barrier.n_gpus == 1) {
            r->gpu_assignment[k] = 0;
        } else {
            r->gpu_assignment[k] = -1;  // CPU-only
        }
    }

    // Re-normalize top-K scores to sum to 1.0
    // Without this, the weighted expert outputs are massively attenuated
    // (e.g. top-8 of 256 experts → scores sum to ~0.3 instead of 1.0)
    float score_sum = 0.0f;
    for (int k = 0; k < top_k; k++) score_sum += r->expert_scores[k];
    if (score_sum > 0.0f) {
        float inv = 1.0f / score_sum;
        for (int k = 0; k < top_k; k++) r->expert_scores[k] *= inv;
    }
}

// Forward declarations for helpers defined later in this file
static void beast_matvec(const sp_optane_tensor_t *W,
                         const float *x, float *out, float *scratch,
                         int n_out, int n_in);
static void beast_silu_mul(float *gate, const float *up, int n);

// ============================================================================
// MoE Forward — The Execution Pulse
// ============================================================================

int sp_beast_moe_forward(sp_beast_engine_t *engine,
                         const float *router_logits,
                         const float *hidden_states,
                         float *output,
                         int layer)
{
    uint64_t t0 = sp_time_us();
    int n_experts = engine->reservoir.n_experts;
    if (n_experts <= 0) return -1;

    size_t hidden_dim = engine->reservoir.n_embd;

    // ── Step 1: Oracle — route experts ──────────────────────────────
    sp_beast_route_experts(engine, router_logits, n_experts);
    const sp_expert_routing_t *r = &engine->routing;

    // Zero the output accumulator
    memset(output, 0, hidden_dim * sizeof(float));

    bool crt_armed = (engine->barrier.n_gpus >= 2 &&
                      engine->shredder_crt.config.use_avx512);

    // ── Step 2: Per-expert FFN (gate→SiLU·up→down) ──────────────────
    for (int k = 0; k < r->n_selected; k++) {
        int eid = r->expert_ids[k];
        float score = r->expert_scores[k];
        const sp_optane_expert_t *exp = sp_optane_layer_expert(&engine->reservoir, layer, eid);
        if (!exp) continue;

        // Prefetch next expert (layer-aware flat index)
        if (k + 1 < r->n_selected) {
            int next_eid = r->expert_ids[k+1];
            int next_flat = layer * engine->reservoir.n_experts + next_eid;
            if (next_flat < SP_OPTANE_MAX_EXPERTS) {
                const sp_optane_expert_t *next_exp = &engine->reservoir.experts[next_flat];
                if (next_exp->gate_proj) sp_optane_prefetch_tensor(next_exp->gate_proj, 0, next_exp->gate_proj->n_bytes);
                if (next_exp->up_proj)   sp_optane_prefetch_tensor(next_exp->up_proj,   0, next_exp->up_proj->n_bytes);
                if (next_exp->down_proj) sp_optane_prefetch_tensor(next_exp->down_proj, 0, next_exp->down_proj->n_bytes);
            }
        }

        // All three projections required for a real expert FFN
        if (!exp->gate_proj || !exp->up_proj || !exp->down_proj) continue;

        int n_ff   = (int)exp->gate_proj->ne[1];
        int in_dim = (int)exp->gate_proj->ne[0];
        if (n_ff <= 0 || in_dim <= 0) continue;

        // CRT Shredder telemetry: shred gate weights into residue streams
        if (crt_armed) {
            sp_shredder_crt_q8_expert(&engine->shredder_crt,
                                      exp->gate_proj->ptr,
                                      n_ff, in_dim);
        }

        // Allocate per-expert scratch
        float *gate_buf = (float *)malloc((size_t)n_ff * sizeof(float));
        float *up_buf   = (float *)malloc((size_t)n_ff * sizeof(float));
        float *down_out = (float *)malloc(hidden_dim * sizeof(float));
        float *scratch  = engine->beast_scratch;
        if (!gate_buf || !up_buf || !down_out) {
            free(gate_buf); free(up_buf); free(down_out);
            continue;
        }

        // Gate projection: [n_ff, n_embd] × [n_embd] → [n_ff]
        beast_matvec(exp->gate_proj, hidden_states, gate_buf, scratch,
                     n_ff, (int)hidden_dim);

        // Up projection: [n_ff, n_embd] × [n_embd] → [n_ff]
        beast_matvec(exp->up_proj, hidden_states, up_buf, scratch,
                     n_ff, (int)hidden_dim);

        // SiLU(gate) * up
        beast_silu_mul(gate_buf, up_buf, n_ff);

        // Down projection: [n_embd, n_ff] × [n_ff] → [n_embd]
        beast_matvec(exp->down_proj, gate_buf, down_out, scratch,
                     (int)hidden_dim, n_ff);

        // Weighted accumulation into output
        for (size_t i = 0; i < hidden_dim; i++) {
            output[i] += score * down_out[i];
        }

        free(gate_buf);
        free(up_buf);
        free(down_out);
    }

    // ── Step 3: CRT dispatch + barrier (telemetry for dual-GPU) ─────
    if (crt_armed) {
        for (int g = 0; g < engine->barrier.n_gpus; g++) {
            sp_hetero_mark_dispatched(&engine->barrier, g);
        }

        // GPU 0 (CPU/AVX-512 — M1 Mersenne ring)
        sp_hetero_mark_done(&engine->barrier, 0);

        // GPU 1 (UHD 750 — M2 ring via L0 USM)
        if (engine->barrier.n_gpus >= 2) {
            sp_l0_t *l0 = (sp_l0_t*)engine->shadow_steal.l0_command_queue;
            if (l0 && l0->loaded && l0->device_found) {
                sp_l0_append_barrier(l0, NULL);
            }
            sp_hetero_mark_done(&engine->barrier, 1);
        }
    }

    uint64_t barrier_us = sp_hetero_barrier_wait(&engine->barrier);

    // Update counters
    engine->total_tokens++;
    uint64_t elapsed = sp_time_us() - t0;
    engine->total_inference_us += elapsed;
    engine->total_barrier_us += barrier_us;

    return 0;
}

// ============================================================================
// Sidecar — S22U via ADB USB-C
// ============================================================================

int sp_beast_sidecar_connect(sp_beast_engine_t *engine) {
    sp_sidecar_t *sc = &engine->sidecar;
    sc->port = engine->config.sidecar_port;
    sc->state = SP_SIDECAR_CONNECTING;

    // Try to connect to localhost:<port> (ADB port-forwarded to S22U)
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) {
        sc->state = SP_SIDECAR_DISCONNECTED;
        fprintf(stderr, "[sp-sidecar] Socket creation failed\n");
        return -1;
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons((u_short)sc->port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        closesocket(sock);
        sc->state = SP_SIDECAR_DISCONNECTED;
        fprintf(stderr, "[sp-sidecar] S22U not available at port %d (run: adb forward tcp:%d tcp:%d)\n",
                sc->port, sc->port, sc->port);
        return -1;
    }

    sc->socket_fd = (int)sock;
#else
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        sc->state = SP_SIDECAR_DISCONNECTED;
        return -1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(sc->port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        sc->state = SP_SIDECAR_DISCONNECTED;
        fprintf(stderr, "[sp-sidecar] S22U not available at port %d\n", sc->port);
        return -1;
    }

    sc->socket_fd = sock;
#endif

    sc->state = SP_SIDECAR_ONLINE;
    fprintf(stderr, "[sp-sidecar] S22U connected at port %d\n", sc->port);
    return 0;
}

void sp_beast_sidecar_disconnect(sp_beast_engine_t *engine) {
    sp_sidecar_t *sc = &engine->sidecar;
    if (sc->state == SP_SIDECAR_ONLINE || sc->socket_fd > 0) {
#ifdef _WIN32
        closesocket((SOCKET)sc->socket_fd);
        WSACleanup();
#else
        close(sc->socket_fd);
#endif
        sc->socket_fd = -1;
    }
    sc->state = SP_SIDECAR_DISCONNECTED;
}

int sp_beast_sidecar_prime_pe(sp_beast_engine_t *engine,
                              const float *hidden_states, int dim,
                              float *pe_output)
{
    if (engine->sidecar.state != SP_SIDECAR_ONLINE) {
        // CPU fallback: run Prime-PE on host
        memcpy(pe_output, hidden_states, dim * sizeof(float));
        return -1;
    }

    int fd = engine->sidecar.socket_fd;
    uint64_t t0 = sp_time_us();

    // Build payload: [dim:u32][hidden:f32*dim]
    uint32_t payload_len = (uint32_t)(sizeof(uint32_t) + dim * sizeof(float));
    uint8_t *payload = (uint8_t *)malloc(payload_len);
    if (!payload) {
        memcpy(pe_output, hidden_states, dim * sizeof(float));
        return -1;
    }
    uint32_t udim = (uint32_t)dim;
    memcpy(payload, &udim, sizeof(udim));
    memcpy(payload + sizeof(udim), hidden_states, dim * sizeof(float));

    if (sp_sidecar_send(fd, SP_CMD_PRIME_PE, payload, payload_len) != 0) {
        free(payload);
        engine->sidecar.state = SP_SIDECAR_ERROR;
        memcpy(pe_output, hidden_states, dim * sizeof(float));
        return -1;
    }
    free(payload);
    engine->sidecar.total_bytes_sent += sizeof(sp_sidecar_hdr_t) + payload_len;

    // Receive response: [dim:u32][pe_out:f32*dim]
    void *resp = NULL;
    uint32_t resp_len = 0;
    int status = sp_sidecar_recv_msg(fd, &resp, &resp_len);
    if (status != SP_RESP_OK || !resp ||
        resp_len < sizeof(uint32_t) + dim * sizeof(float))
    {
        free(resp);
        engine->sidecar.state = SP_SIDECAR_ERROR;
        memcpy(pe_output, hidden_states, dim * sizeof(float));
        return -1;
    }

    memcpy(pe_output, (uint8_t *)resp + sizeof(uint32_t), dim * sizeof(float));
    free(resp);

    engine->sidecar.total_bytes_recv += sizeof(sp_sidecar_hdr_t) + resp_len;
    engine->sidecar.total_offloads++;
    engine->sidecar.total_offload_us += sp_time_us() - t0;

    return 0;
}

// ── Sidecar: Ping ────────────────────────────────────────────────────

int sp_beast_sidecar_ping(sp_beast_engine_t *engine) {
    if (engine->sidecar.state != SP_SIDECAR_ONLINE) return -1;

    int fd = engine->sidecar.socket_fd;
    if (sp_sidecar_send(fd, SP_CMD_PING, NULL, 0) != 0) {
        engine->sidecar.state = SP_SIDECAR_ERROR;
        return -1;
    }

    void *resp = NULL;
    uint32_t resp_len = 0;
    int status = sp_sidecar_recv_msg(fd, &resp, &resp_len);
    free(resp);

    if (status != SP_RESP_OK) {
        engine->sidecar.state = SP_SIDECAR_ERROR;
        return -1;
    }
    return 0;
}

// ── Sidecar: Load draft model ─────────────────────────────────────────

int sp_beast_sidecar_load_draft(sp_beast_engine_t *engine,
                                const char *phone_gguf_path)
{
    if (engine->sidecar.state != SP_SIDECAR_ONLINE) return -1;

    int fd = engine->sidecar.socket_fd;
    uint32_t path_len = (uint32_t)(strlen(phone_gguf_path) + 1); // include NUL

    if (sp_sidecar_send(fd, SP_CMD_LOAD_DRAFT, phone_gguf_path, path_len) != 0) {
        engine->sidecar.state = SP_SIDECAR_ERROR;
        return -1;
    }
    engine->sidecar.total_bytes_sent += sizeof(sp_sidecar_hdr_t) + path_len;

    void *resp = NULL;
    uint32_t resp_len = 0;
    int status = sp_sidecar_recv_msg(fd, &resp, &resp_len);
    free(resp);

    if (status != SP_RESP_OK) {
        fprintf(stderr, "[sp-sidecar] Load draft failed (status=%d)\n", status);
        return -1;
    }

    fprintf(stderr, "[sp-sidecar] Draft model loaded: %s\n", phone_gguf_path);
    return 0;
}

// ── Sidecar: Prefill draft model ──────────────────────────────────────

int sp_beast_sidecar_prefill(sp_beast_engine_t *engine,
                             const int32_t *tokens, int n_tokens)
{
    if (engine->sidecar.state != SP_SIDECAR_ONLINE) return -1;

    int fd = engine->sidecar.socket_fd;

    // Payload: [n_tokens:u32][token_ids:i32*n]
    uint32_t payload_len = (uint32_t)(sizeof(uint32_t) + n_tokens * sizeof(int32_t));
    uint8_t *payload = (uint8_t *)malloc(payload_len);
    if (!payload) return -1;

    uint32_t un = (uint32_t)n_tokens;
    memcpy(payload, &un, sizeof(un));
    memcpy(payload + sizeof(un), tokens, n_tokens * sizeof(int32_t));

    if (sp_sidecar_send(fd, SP_CMD_PREFILL, payload, payload_len) != 0) {
        free(payload);
        engine->sidecar.state = SP_SIDECAR_ERROR;
        return -1;
    }
    free(payload);
    engine->sidecar.total_bytes_sent += sizeof(sp_sidecar_hdr_t) + payload_len;

    void *resp = NULL;
    uint32_t resp_len = 0;
    int status = sp_sidecar_recv_msg(fd, &resp, &resp_len);
    free(resp);

    if (status != SP_RESP_OK) {
        fprintf(stderr, "[sp-sidecar] Prefill failed (status=%d)\n", status);
        return -1;
    }

    fprintf(stderr, "[sp-sidecar] Draft prefilled with %d tokens\n", n_tokens);
    return 0;
}

// ── Sidecar: Draft speculative tokens ─────────────────────────────────

int sp_beast_sidecar_draft(sp_beast_engine_t *engine,
                           int32_t current_tok, int n_draft,
                           sp_sidecar_draft_result_t *out)
{
    memset(out, 0, sizeof(*out));

    if (engine->sidecar.state != SP_SIDECAR_ONLINE) return -1;
    if (n_draft > SP_SIDECAR_MAX_DRAFT) n_draft = SP_SIDECAR_MAX_DRAFT;

    int fd = engine->sidecar.socket_fd;
    uint64_t t0 = sp_time_us();

    // Payload: [current_token:i32][n_draft:u32]
    uint8_t payload[8];
    memcpy(payload, &current_tok, sizeof(int32_t));
    uint32_t un = (uint32_t)n_draft;
    memcpy(payload + 4, &un, sizeof(uint32_t));

    if (sp_sidecar_send(fd, SP_CMD_DRAFT, payload, 8) != 0) {
        engine->sidecar.state = SP_SIDECAR_ERROR;
        return -1;
    }
    engine->sidecar.total_bytes_sent += sizeof(sp_sidecar_hdr_t) + 8;

    // Response: [n_drafted:u32][draft_ids:i32*n]
    void *resp = NULL;
    uint32_t resp_len = 0;
    int status = sp_sidecar_recv_msg(fd, &resp, &resp_len);
    if (status != SP_RESP_OK || !resp || resp_len < sizeof(uint32_t)) {
        free(resp);
        if (status == (int)SP_RESP_NOT_LOADED) {
            fprintf(stderr, "[sp-sidecar] Draft model not loaded on phone\n");
        }
        return -1;
    }

    uint32_t n_got = 0;
    memcpy(&n_got, resp, sizeof(uint32_t));
    if (n_got > SP_SIDECAR_MAX_DRAFT) n_got = SP_SIDECAR_MAX_DRAFT;
    if (resp_len < sizeof(uint32_t) + n_got * sizeof(int32_t)) {
        free(resp);
        return -1;
    }

    memcpy(out->tokens, (uint8_t *)resp + sizeof(uint32_t),
           n_got * sizeof(int32_t));
    out->n_drafted = (int)n_got;
    free(resp);

    engine->sidecar.total_bytes_recv += sizeof(sp_sidecar_hdr_t) + resp_len;
    out->round_trip_us = sp_time_us() - t0;
    engine->sidecar.total_offloads++;
    engine->sidecar.total_offload_us += out->round_trip_us;

    return 0;
}

// ── Sidecar: Accept verified tokens ───────────────────────────────────

int sp_beast_sidecar_accept(sp_beast_engine_t *engine,
                            const int32_t *verified, int n_accepted)
{
    if (engine->sidecar.state != SP_SIDECAR_ONLINE) return -1;

    int fd = engine->sidecar.socket_fd;

    uint32_t payload_len = (uint32_t)(sizeof(uint32_t) + n_accepted * sizeof(int32_t));
    uint8_t *payload = (uint8_t *)malloc(payload_len);
    if (!payload) return -1;

    uint32_t un = (uint32_t)n_accepted;
    memcpy(payload, &un, sizeof(un));
    memcpy(payload + sizeof(un), verified, n_accepted * sizeof(int32_t));

    if (sp_sidecar_send(fd, SP_CMD_ACCEPT, payload, payload_len) != 0) {
        free(payload);
        engine->sidecar.state = SP_SIDECAR_ERROR;
        return -1;
    }
    free(payload);
    engine->sidecar.total_bytes_sent += sizeof(sp_sidecar_hdr_t) + payload_len;

    // Don't wait for response on accept — fire and forget for latency
    return 0;
}

// ── Sidecar: Reset draft state ────────────────────────────────────────

int sp_beast_sidecar_reset(sp_beast_engine_t *engine) {
    if (engine->sidecar.state != SP_SIDECAR_ONLINE) return -1;

    int fd = engine->sidecar.socket_fd;
    if (sp_sidecar_send(fd, SP_CMD_RESET, NULL, 0) != 0) {
        engine->sidecar.state = SP_SIDECAR_ERROR;
        return -1;
    }

    void *resp = NULL;
    uint32_t resp_len = 0;
    int status = sp_sidecar_recv_msg(fd, &resp, &resp_len);
    free(resp);

    return (status == SP_RESP_OK) ? 0 : -1;
}

// ============================================================================
// Attention forward (non-MoE layers)
// ============================================================================

int sp_beast_attention_forward(sp_beast_engine_t *engine,
                               const float *q, const float *k, const float *v,
                               float *output,
                               int layer, int pos, int kv_len)
{
    // Write K/V to Shannon-Prime compressed cache
    // sp_shadow_write_k(cache, layer, head, pos, k);
    // sp_shadow_write_v(cache, layer, head, pos, v);

    // Read back full KV for attention
    // Compute attention scores
    // Output = softmax(Q @ K^T / sqrt(d)) @ V

    // TODO: actual implementation using existing engine infrastructure
    // For skeleton testing, passthrough Q
    size_t head_dim = engine->reservoir.head_dim;
    memcpy(output, q, head_dim * sizeof(float));

    return 0;
}

// ============================================================================
// Self-contained dequant — no ggml dependency
// ============================================================================

static float beast_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (frac == 0) { bits = sign; }
        else { exp = 1; while (!(frac & 0x400)) { frac <<= 1; exp--; }
               frac &= 0x3FF; bits = sign | ((exp + 112) << 23) | (frac << 13); }
    } else if (exp == 31) { bits = sign | 0x7F800000 | (frac << 13); }
    else { bits = sign | ((exp + 112) << 23) | (frac << 13); }
    float f; memcpy(&f, &bits, 4); return f;
}

// Row byte size for GGUF quant types (matches sp_optane.c tables)
static size_t beast_row_bytes(uint32_t type, int ne0) {
    size_t blck, tsize;
    switch (type) {
    case SP_GGML_TYPE_F32:     return (size_t)ne0 * 4;
    case SP_GGML_TYPE_F16:     return (size_t)ne0 * 2;
    case SP_GGML_TYPE_Q4_0:    blck = 32;  tsize = 18;  break;
    case SP_GGML_TYPE_Q4_1:    blck = 32;  tsize = 20;  break;
    case SP_GGML_TYPE_Q5_0:    blck = 32;  tsize = 22;  break;
    case SP_GGML_TYPE_Q5_1:    blck = 32;  tsize = 24;  break;
    case SP_GGML_TYPE_Q8_0:    blck = 32;  tsize = 34;  break;
    case SP_GGML_TYPE_Q8_1:    blck = 32;  tsize = 36;  break;
    case SP_GGML_TYPE_Q2_K:    blck = 256; tsize = 84;  break;
    case SP_GGML_TYPE_Q3_K:    blck = 256; tsize = 110; break;
    case SP_GGML_TYPE_Q4_K:    blck = 256; tsize = 144; break;
    case SP_GGML_TYPE_Q5_K:    blck = 256; tsize = 176; break;
    case SP_GGML_TYPE_Q6_K:    blck = 256; tsize = 210; break;
    case SP_GGML_TYPE_Q8_K:    blck = 256; tsize = 292; break;
    default:                   return (size_t)ne0 * 4;
    }
    return ((size_t)ne0 / blck) * tsize;
}

// --- Q8_0 dequant: 34 bytes per 32 elements ---
// Layout: [f16 scale][32 × int8 values]
static void beast_dequant_q8_0(const void *src, float *dst, int n) {
    const uint8_t *p = (const uint8_t *)src;
    for (int b = 0; b < n / 32; b++) {
        float scale = beast_fp16_to_fp32(*(const uint16_t *)p);
        const int8_t *qs = (const int8_t *)(p + 2);
        for (int i = 0; i < 32; i++) dst[b * 32 + i] = qs[i] * scale;
        p += 34;
    }
}

// --- Q4_K scale unpacker ---
static void beast_get_scale_min_k4(int j, const uint8_t *q,
                                    uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

// --- Q4_K dequant: 144 bytes per 256 elements ---
// Layout: [f16 d][f16 dmin][12 scales][128 qs (4-bit)]
static void beast_dequant_q4_K(const void *src, float *dst, int n) {
    const uint8_t *p = (const uint8_t *)src;
    for (int b = 0; b < n / 256; b++) {
        float d    = beast_fp16_to_fp32(*(const uint16_t *)(p + 0));
        float dmin = beast_fp16_to_fp32(*(const uint16_t *)(p + 2));
        const uint8_t *scales = p + 4;
        const uint8_t *qs = p + 16;
        float *y = dst + b * 256;
        int is = 0;
        for (int j = 0; j < 4; j++) {  // 4 groups of 64
            uint8_t sc, m;
            beast_get_scale_min_k4(is++, scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            for (int l = 0; l < 32; l++) *y++ = d1 * (qs[l] & 0xF) - m1;
            beast_get_scale_min_k4(is++, scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; l++) *y++ = d2 * (qs[l] >> 4) - m2;
            qs += 32;
        }
        p += 144;
    }
}

// --- Q6_K dequant: 210 bytes per 256 elements ---
// Layout: [128 ql (lower 4-bit)][64 qh (upper 2-bit)][16 scales (int8)][f16 d]
static void beast_dequant_q6_K(const void *src, float *dst, int n) {
    const uint8_t *p = (const uint8_t *)src;
    for (int b = 0; b < n / 256; b++) {
        const uint8_t *ql = p;
        const uint8_t *qh = p + 128;
        const int8_t  *sc = (const int8_t *)(p + 192);
        float d = beast_fp16_to_fp32(*(const uint16_t *)(p + 208));
        float *y = dst + b * 256;

        for (int j = 0; j < 2; j++) {  // 2 halves of 128
            for (int l = 0; l < 32; l++) {
                int q1 = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
                y[l +  0] = d * sc[0] * ((int)q1 - 32);
                int q2 = (ql[l] >> 4)  | (((qh[l] >> 2) & 3) << 4);
                y[l + 32] = d * sc[1] * ((int)q2 - 32);
                int q3 = (ql[l + 32] & 0xF) | (((qh[l] >> 4) & 3) << 4);
                y[l + 64] = d * sc[2] * ((int)q3 - 32);
                int q4 = (ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4);
                y[l + 96] = d * sc[3] * ((int)q4 - 32);
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 4;
        }
        p += 210;
    }
}

// --- Q5_K dequant: 176 bytes per 256 elements ---
// Layout: [f16 d][f16 dmin][12 scales][32 qh (high bits)][128 qs (4-bit)]
static void beast_dequant_q5_K(const void *src, float *dst, int n) {
    const uint8_t *p = (const uint8_t *)src;
    for (int b = 0; b < n / 256; b++) {
        float d    = beast_fp16_to_fp32(*(const uint16_t *)(p + 0));
        float dmin = beast_fp16_to_fp32(*(const uint16_t *)(p + 2));
        const uint8_t *scales = p + 4;
        const uint8_t *qh = p + 16;
        const uint8_t *qs = p + 48;
        float *y = dst + b * 256;
        int is = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc, m;
            beast_get_scale_min_k4(is++, scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            for (int l = 0; l < 32; l++) {
                uint8_t hbit = (qh[l] >> j) & 1;
                *y++ = d1 * ((qs[l] & 0xF) | (hbit << 4)) - m1;
            }
            beast_get_scale_min_k4(is++, scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; l++) {
                uint8_t hbit = (qh[l] >> (j + 4)) & 1;
                *y++ = d2 * ((qs[l] >> 4) | (hbit << 4)) - m2;
            }
            qs += 32;
        }
        p += 176;
    }
}

// Dispatch dequant by type. Writes n float values to dst.
static void beast_dequant_row(const void *src, float *dst, uint32_t type, int n) {
    switch (type) {
    case SP_GGML_TYPE_F32:
        memcpy(dst, src, (size_t)n * sizeof(float));
        break;
    case SP_GGML_TYPE_F16: {
        const uint16_t *s = (const uint16_t *)src;
        for (int i = 0; i < n; i++) dst[i] = beast_fp16_to_fp32(s[i]);
        break;
    }
    case SP_GGML_TYPE_Q8_0:  beast_dequant_q8_0(src, dst, n); break;
    case SP_GGML_TYPE_Q4_K:  beast_dequant_q4_K(src, dst, n); break;
    case SP_GGML_TYPE_Q5_K:  beast_dequant_q5_K(src, dst, n); break;
    case SP_GGML_TYPE_Q6_K:  beast_dequant_q6_K(src, dst, n); break;
    default:
        fprintf(stderr, "[sp-beast] Unsupported dequant type %u — zeroing\n", type);
        memset(dst, 0, (size_t)n * sizeof(float));
        break;
    }
}

// ============================================================================
// Forward Pass — CPU Ops (used for attention layers; MoE goes through CRT)
// ============================================================================

static void beast_rms_norm(const float *x, const float *scale,
                           int n, float eps, float *out) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv_rms * scale[i];
}

// Mat-vec: out[n_out] = W[n_out, n_in] @ x[n_in]
// Dequants one row at a time into scratch, then dot product.
static void beast_matvec(const sp_optane_tensor_t *W,
                         const float *x, float *out, float *scratch,
                         int n_out, int n_in) {
    const uint8_t *w_data = (const uint8_t *)W->ptr;
    size_t rb = beast_row_bytes(W->type, n_in);

    for (int i = 0; i < n_out; i++) {
        beast_dequant_row(w_data + (size_t)i * rb, scratch, W->type, n_in);
        float sum = 0.0f;
        for (int j = 0; j < n_in; j++) sum += scratch[j] * x[j];
        out[i] = sum;
    }
}

// mRoPE-aware rotation: only first rope_dim dimensions are rotated.
// For standard RoPE, rope_dim == head_dim (rotate all).
// For mRoPE (Qwen3.6), rope_dim < head_dim (e.g. 64 of 128 — upper half pass-through).
// Text-only: all mRoPE sections use the same position, so no partition logic needed.
// The freq denominator uses rope_dim, not head_dim, per the mRoPE spec.
static void beast_rope(float *x, int head_dim, int n_heads,
                       int pos, float freq_base, int rope_dim) {
    int n_rot = rope_dim > 0 ? rope_dim : head_dim;  // dims to rotate
    int n_pairs = n_rot / 2;
    for (int h = 0; h < n_heads; h++) {
        float *xh = x + h * head_dim;
        for (int i = 0; i < n_pairs; i++) {
            float freq = 1.0f / powf(freq_base, 2.0f * i / (float)n_rot);
            float angle = pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float a = xh[2*i], b = xh[2*i + 1];
            xh[2*i]     = a * c - b * s;
            xh[2*i + 1] = a * s + b * c;
        }
        // Dims [n_rot .. head_dim-1] are untouched (pass-through)
    }
}

static void beast_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static void beast_silu_mul(float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

static int beast_argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

// Embed lookup: dequant row `token_id` from the embedding tensor
static int beast_embed(const sp_optane_tensor_t *tok_embd,
                       int token_id, int n_embd, float *out) {
    if (!tok_embd || !tok_embd->ptr) return -1;
    size_t row_bytes = beast_row_bytes(tok_embd->type, n_embd);
    if (row_bytes == 0) return -1;
    const uint8_t *row = (const uint8_t *)tok_embd->ptr +
                         (size_t)token_id * row_bytes;

    if (tok_embd->type == SP_GGML_TYPE_F32) {
        memcpy(out, row, (size_t)n_embd * sizeof(float));
    } else {
        beast_dequant_row(row, out, tok_embd->type, n_embd);
    }
    return 0;
}

// ============================================================================
// Full forward pass — one token through all layers
// ============================================================================

// ── Shared expert FFN (runs in every layer alongside routed MoE) ──
static void beast_shared_expert_ffn(sp_beast_engine_t *engine,
                                     const float *xn, float *ffn_out,
                                     int l, int n_embd) {
    sp_optane_reservoir_t *res = &engine->reservoir;
    float *scratch = engine->beast_scratch;
    char name[128];

    // Shared expert gate (scalar → sigmoid for gating the shared expert output)
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", l);
    const sp_optane_tensor_t *shexp_gate = sp_optane_find_tensor(res, name);

    snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", l);
    const sp_optane_tensor_t *sg = sp_optane_find_tensor(res, name);
    snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", l);
    const sp_optane_tensor_t *su = sp_optane_find_tensor(res, name);
    snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", l);
    const sp_optane_tensor_t *sd = sp_optane_find_tensor(res, name);

    if (!sg || !su || !sd || !sg->ptr || !su->ptr || !sd->ptr) return;

    int shexp_ff = (int)sg->ne[1];
    float *sg_buf = (float *)malloc((size_t)shexp_ff * sizeof(float));
    float *su_buf = (float *)malloc((size_t)shexp_ff * sizeof(float));
    float *sd_out = (float *)malloc((size_t)n_embd * sizeof(float));
    if (!sg_buf || !su_buf || !sd_out) {
        free(sg_buf); free(su_buf); free(sd_out);
        return;
    }

    beast_matvec(sg, xn, sg_buf, scratch, shexp_ff, n_embd);
    beast_matvec(su, xn, su_buf, scratch, shexp_ff, n_embd);
    beast_silu_mul(sg_buf, su_buf, shexp_ff);
    beast_matvec(sd, sg_buf, sd_out, scratch, n_embd, shexp_ff);

    // Apply shared expert gate if present (sigmoid scaling)
    if (shexp_gate && shexp_gate->ptr) {
        // Gate is [n_embd] — dot product with xn → scalar → sigmoid
        const float *gw = (const float *)shexp_gate->ptr;
        float gate_val = 0.0f;
        for (int i = 0; i < n_embd; i++) gate_val += gw[i] * xn[i];
        gate_val = 1.0f / (1.0f + expf(-gate_val));
        for (int i = 0; i < n_embd; i++) sd_out[i] *= gate_val;
    }

    for (int i = 0; i < n_embd; i++) ffn_out[i] += sd_out[i];

    free(sg_buf); free(su_buf); free(sd_out);
}

// ── MoE FFN block (routed experts + shared expert) ──
static void beast_moe_ffn(sp_beast_engine_t *engine, const float *xn,
                           float *ffn_out, int l, int n_embd) {
    sp_optane_reservoir_t *res = &engine->reservoir;
    float *scratch = engine->beast_scratch;
    char name[128];
    float *router_logits = (float *)malloc((size_t)res->n_experts * sizeof(float));
    if (!router_logits) return;

    // Router
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
    const sp_optane_tensor_t *gate_inp = sp_optane_find_tensor(res, name);
    if (gate_inp && gate_inp->ptr) {
        beast_matvec(gate_inp, xn, router_logits, scratch, res->n_experts, n_embd);
    }

    // Routed expert dispatch
    sp_beast_moe_forward(engine, router_logits, xn, ffn_out, l);

    // Shared expert (always-on alongside routed experts)
    beast_shared_expert_ffn(engine, xn, ffn_out, l, n_embd);

    free(router_logits);
}

// ── Gated Delta Net layer (Qwen3.6 "SSM" layers) ──
// NOT Mamba2 — this is a Delta Net linear attention with learned decay + delta rule.
//
// Tensor mapping:
//   attn_qkv [n_embd → 8192]  = in_proj for Q(2048)+K(2048)+V(4096)
//   attn_gate [n_embd → 4096]  = z (output gate)
//   ssm_alpha [n_embd → 32]    = decay gate (softplus → exp)
//   ssm_beta  [n_embd → 32]    = delta learning rate (sigmoid)
//   ssm_a     [32]              = negative scaling for decay
//   ssm_dt.bias [32]            = bias added to alpha before softplus
//   ssm_conv1d [4 × 8192]      = depthwise conv on Q+K+V
//   ssm_norm  [128]             = per-head RMS norm on output
//   ssm_out   [4096 → n_embd]  = output projection
//
// State: S_h [head_k_dim × head_v_dim] = [128 × 128] per head, 32 heads per layer
//        conv_state [conv_kernel-1, conv_dim] = [3, 8192]
//
// Recurrence (per head h):
//   gate = exp(softplus(alpha[h] + dt_bias[h]) * ssm_a[h])   // scalar decay
//   beta_h = sigmoid(beta_raw[h])                              // learning rate
//   S_h = gate * S_h + outer(k, (v - S_h^T k) * beta_h)       // delta rule update
//   o_h = S_h @ q                                               // query retrieval
static void beast_ssm_layer(sp_beast_engine_t *engine, float *x,
                             int l, int pos) {
    sp_optane_reservoir_t *res = &engine->reservoir;
    const int n_embd = (int)res->n_embd;
    const int n_v_heads = (int)res->ssm_dt_rank;    // 32
    const int n_k_heads = (int)res->ssm_n_groups;    // 16
    const int head_k_dim = (int)res->ssm_state_size; // 128
    const int d_inner = (int)res->ssm_inner_size;    // 4096
    const int head_v_dim = d_inner / n_v_heads;      // 128
    const int key_dim = head_k_dim * n_k_heads;      // 2048
    const int conv_dim = key_dim * 2 + d_inner;      // 8192
    const int conv_k = (int)res->ssm_conv_kernel;    // 4
    const float eps = res->rms_norm_eps;
    float *scratch = engine->beast_scratch;
    float *xn = engine->beast_xn;
    char name[128];

    // Compute SSM layer index (for state addressing)
    int attn_interval = (int)res->full_attn_interval;
    // SSM layers are all layers where (l+1)%interval != 0
    // Map to contiguous index for state arrays
    int ssm_idx = l;  // simplified: use layer index directly since we allocate for all layers

    // 1. Attention norm
    snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
    const sp_optane_tensor_t *an = sp_optane_find_tensor(res, name);
    if (!an || !an->ptr) return;
    beast_rms_norm(x, (const float *)an->ptr, n_embd, eps, xn);

    // 2. Input projections
    snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", l);
    const sp_optane_tensor_t *wqkv = sp_optane_find_tensor(res, name);
    snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", l);
    const sp_optane_tensor_t *wz = sp_optane_find_tensor(res, name);
    snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", l);
    const sp_optane_tensor_t *walpha = sp_optane_find_tensor(res, name);
    snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", l);
    const sp_optane_tensor_t *wbeta = sp_optane_find_tensor(res, name);

    if (!wqkv || !wz || !walpha || !wbeta ||
        !wqkv->ptr || !wz->ptr || !walpha->ptr || !wbeta->ptr) return;

    float *qkv_buf = (float *)calloc(conv_dim, sizeof(float));     // [8192]
    float *z_buf   = (float *)calloc(d_inner, sizeof(float));      // [4096]
    float *alpha_raw = (float *)calloc(n_v_heads, sizeof(float));  // [32]
    float *beta_raw  = (float *)calloc(n_v_heads, sizeof(float));  // [32]
    float *output  = (float *)calloc(d_inner, sizeof(float));      // [4096]
    float *attn_out = (float *)calloc(n_embd, sizeof(float));

    if (!qkv_buf || !z_buf || !alpha_raw || !beta_raw || !output || !attn_out) {
        free(qkv_buf); free(z_buf); free(alpha_raw); free(beta_raw);
        free(output); free(attn_out);
        return;
    }

    beast_matvec(wqkv, xn, qkv_buf, scratch, conv_dim, n_embd);
    beast_matvec(wz, xn, z_buf, scratch, d_inner, n_embd);
    beast_matvec(walpha, xn, alpha_raw, scratch, n_v_heads, n_embd);
    beast_matvec(wbeta, xn, beta_raw, scratch, n_v_heads, n_embd);

    // 3. Compute decay gate (alpha) and learning rate (beta)
    snprintf(name, sizeof(name), "blk.%d.ssm_a", l);
    const sp_optane_tensor_t *ta = sp_optane_find_tensor(res, name);
    snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", l);
    const sp_optane_tensor_t *tdt = sp_optane_find_tensor(res, name);

    float gate_vals[64];   // decay per head (max 64 heads)
    float beta_vals[64];   // learning rate per head
    for (int h = 0; h < n_v_heads && h < 64; h++) {
        // Alpha: add dt_bias, softplus, multiply by ssm_a (negative), exp
        float a_raw = alpha_raw[h];
        if (tdt && tdt->ptr) a_raw += ((const float *)tdt->ptr)[h];
        float sp = logf(1.0f + expf(a_raw));  // softplus
        float a_neg = (ta && ta->ptr) ? ((const float *)ta->ptr)[h] : -1.0f;
        gate_vals[h] = expf(sp * a_neg);  // decay factor (0 < gate < 1 since a_neg < 0)

        // Beta: sigmoid → learning rate
        beta_vals[h] = 1.0f / (1.0f + expf(-beta_raw[h]));
    }

    // 4. Conv1d (depthwise, causal)
    snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", l);
    const sp_optane_tensor_t *conv_w = sp_optane_find_tensor(res, name);

    // Conv state: [conv_k-1, conv_dim] = [3, 8192]
    int conv_state_stride = conv_dim * (conv_k - 1);
    float *conv_state = engine->beast_conv_state + (size_t)ssm_idx * conv_state_stride;

    if (conv_w && conv_w->ptr) {
        const float *cw = (const float *)conv_w->ptr;
        // Shift conv state: drop oldest, append new
        // State layout: [slot0][slot1][slot2] each [conv_dim]
        // Shift left: slot0=slot1, slot1=slot2, slot2=new
        for (int s = 0; s < conv_k - 2; s++) {
            memcpy(conv_state + s * conv_dim,
                   conv_state + (s + 1) * conv_dim,
                   (size_t)conv_dim * sizeof(float));
        }
        memcpy(conv_state + (conv_k - 2) * conv_dim, qkv_buf,
               (size_t)conv_dim * sizeof(float));

        // Apply depthwise conv: for each channel, dot product over kernel
        // conv_w layout: [kernel_size × channels] = [4 × 8192]
        for (int c = 0; c < conv_dim; c++) {
            float sum = 0.0f;
            for (int k = 0; k < conv_k - 1; k++) {
                sum += conv_state[k * conv_dim + c] * cw[k * conv_dim + c];
            }
            // Current token (slot conv_k-1) is qkv_buf itself
            sum += qkv_buf[c] * cw[(conv_k - 1) * conv_dim + c];
            qkv_buf[c] = sum;
        }

        // SiLU activation on conv output
        for (int i = 0; i < conv_dim; i++) {
            float v = qkv_buf[i];
            qkv_buf[i] = v / (1.0f + expf(-v));
        }
    }

    // 5. Split conv output → Q [key_dim], K [key_dim], V [d_inner]
    float *q_all = qkv_buf;                    // [2048]
    float *k_all = qkv_buf + key_dim;          // [2048]
    float *v_all = qkv_buf + key_dim * 2;      // [4096]

    // L2 normalize Q and K per head
    for (int h = 0; h < n_k_heads; h++) {
        float *qh = q_all + h * head_k_dim;
        float *kh = k_all + h * head_k_dim;
        float qnorm = 0.0f, knorm = 0.0f;
        for (int i = 0; i < head_k_dim; i++) {
            qnorm += qh[i] * qh[i];
            knorm += kh[i] * kh[i];
        }
        qnorm = 1.0f / (sqrtf(qnorm) + 1e-12f);
        knorm = 1.0f / (sqrtf(knorm) + 1e-12f);
        for (int i = 0; i < head_k_dim; i++) { qh[i] *= qnorm; kh[i] *= knorm; }
    }

    // 6. Delta Net recurrence — per head
    // SSM state: S_h [head_k_dim × head_v_dim] = [128 × 128] per head
    int state_per_head = head_k_dim * head_v_dim;  // 128*128 = 16384
    float *ssm_state = engine->beast_ssm_state +
                       (size_t)ssm_idx * n_v_heads * state_per_head;

    int heads_per_group = n_v_heads / n_k_heads;  // 32/16 = 2

    for (int h = 0; h < n_v_heads; h++) {
        int kh_idx = h / heads_per_group;  // which K group this head uses
        float *qh = q_all + kh_idx * head_k_dim;   // Q from K-head group
        float *kh = k_all + kh_idx * head_k_dim;   // K from K-head group
        float *vh = v_all + h * head_v_dim;          // V per v-head
        float *Sh = ssm_state + h * state_per_head;  // [128 × 128] state
        float *oh = output + h * head_v_dim;          // output [128]

        float gate_h = gate_vals[h];
        float beta_h = beta_vals[h];
        float scale = 1.0f / sqrtf((float)head_k_dim);

        // Decay state: S = gate * S
        for (int i = 0; i < state_per_head; i++) Sh[i] *= gate_h;

        // Retrieve: sk = S^T @ k → [head_v_dim]
        // S is [head_k_dim × head_v_dim], S^T @ k = sum over head_k_dim
        float sk[256]; // max head_v_dim
        for (int j = 0; j < head_v_dim; j++) {
            float sum = 0.0f;
            for (int i = 0; i < head_k_dim; i++) {
                sum += Sh[i * head_v_dim + j] * kh[i];
            }
            sk[j] = sum;
        }

        // Delta: d = (v - sk) * beta
        float d[256];
        for (int j = 0; j < head_v_dim; j++) {
            d[j] = (vh[j] - sk[j]) * beta_h;
        }

        // Update: S += outer(k, d)
        for (int i = 0; i < head_k_dim; i++) {
            for (int j = 0; j < head_v_dim; j++) {
                Sh[i * head_v_dim + j] += kh[i] * d[j];
            }
        }

        // Query: o = S @ q (scaled)
        for (int j = 0; j < head_v_dim; j++) {
            float sum = 0.0f;
            for (int i = 0; i < head_k_dim; i++) {
                sum += Sh[i * head_v_dim + j] * qh[i];
            }
            oh[j] = sum * scale;
        }
    }

    // 7. Group RMS norm on output (per head)
    snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", l);
    const sp_optane_tensor_t *ssm_norm = sp_optane_find_tensor(res, name);
    if (ssm_norm && ssm_norm->ptr) {
        const float *nw = (const float *)ssm_norm->ptr;
        int norm_dim = (int)ssm_norm->ne[0];  // 128 = head_v_dim
        for (int h = 0; h < n_v_heads; h++) {
            float *seg = output + h * head_v_dim;
            int slen = (norm_dim < head_v_dim) ? norm_dim : head_v_dim;
            float ss = 0.0f;
            for (int i = 0; i < slen; i++) ss += seg[i] * seg[i];
            float irms = 1.0f / sqrtf(ss / slen + eps);
            for (int i = 0; i < slen; i++) seg[i] *= irms * nw[i % norm_dim];
        }
    }

    // 8. Gate: output *= SiLU(z)
    for (int i = 0; i < d_inner; i++) {
        float g = z_buf[i];
        output[i] *= g / (1.0f + expf(-g));
    }

    // 9. Output projection: [d_inner → n_embd]
    snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", l);
    const sp_optane_tensor_t *wout = sp_optane_find_tensor(res, name);
    if (wout && wout->ptr) {
        beast_matvec(wout, output, attn_out, scratch, n_embd, d_inner);
        for (int i = 0; i < n_embd; i++) x[i] += attn_out[i];
    }

    free(qkv_buf); free(z_buf); free(alpha_raw); free(beta_raw);
    free(output); free(attn_out);
}

// ── Full attention layer (every full_attn_interval-th layer) ──
static void beast_attn_layer(sp_beast_engine_t *engine, float *x,
                              int l, int pos) {
    sp_optane_reservoir_t *res = &engine->reservoir;
    const int n_embd = (int)res->n_embd;
    const int n_head = (int)res->n_head;
    const int n_head_kv = (int)res->n_head_kv;
    // Use key_length for per-head K/V dim if available
    const int kv_head_dim = (res->key_length > 0) ? (int)res->key_length : (int)res->head_dim;
    const int n_kv_dim = n_head_kv * kv_head_dim;
    const int max_ctx = engine->beast_max_ctx;
    const float eps = res->rms_norm_eps;
    float *scratch = engine->beast_scratch;
    float *xn = engine->beast_xn;
    float *k_cache = engine->beast_k_cache;
    float *v_cache = engine->beast_v_cache;
    char name[128];

    // 1. Attention norm
    snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
    const sp_optane_tensor_t *an = sp_optane_find_tensor(res, name);
    if (!an || !an->ptr) return;
    beast_rms_norm(x, (const float *)an->ptr, n_embd, eps, xn);

    // 2. Q/K/V projections (separate for full attention layers)
    snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
    const sp_optane_tensor_t *wq = sp_optane_find_tensor(res, name);
    snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
    const sp_optane_tensor_t *wk = sp_optane_find_tensor(res, name);
    snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
    const sp_optane_tensor_t *wv = sp_optane_find_tensor(res, name);

    if (!wq || !wk || !wv || !wq->ptr || !wk->ptr || !wv->ptr) return;

    // Q output dim from weight shape
    int n_q_out = (int)wq->ne[1];
    int q_head_dim = n_q_out / n_head;  // per-head Q dim (may differ from kv_head_dim)

    float *q_buf = (float *)malloc((size_t)n_q_out * sizeof(float));
    float *k_buf = (float *)malloc((size_t)n_kv_dim * sizeof(float));
    float *v_buf = (float *)malloc((size_t)n_kv_dim * sizeof(float));
    float *scores = (float *)malloc((size_t)(pos + 1) * sizeof(float));
    float *head_out = (float *)malloc((size_t)n_q_out * sizeof(float));
    float *attn_out = (float *)malloc((size_t)n_embd * sizeof(float));

    if (!q_buf || !k_buf || !v_buf || !scores || !head_out || !attn_out) {
        free(q_buf); free(k_buf); free(v_buf); free(scores);
        free(head_out); free(attn_out);
        return;
    }

    beast_matvec(wq, xn, q_buf, scratch, n_q_out, n_embd);
    beast_matvec(wk, xn, k_buf, scratch, n_kv_dim, n_embd);
    beast_matvec(wv, xn, v_buf, scratch, n_kv_dim, n_embd);

    // 3. Q/K head norms (Qwen3.6 attention layers have per-head normalization)
    snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l);
    const sp_optane_tensor_t *qn = sp_optane_find_tensor(res, name);
    snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
    const sp_optane_tensor_t *kn = sp_optane_find_tensor(res, name);

    if (qn && qn->ptr) {
        const float *nw = (const float *)qn->ptr;
        int norm_dim = (int)qn->ne[0];  // per-head norm dim (kv_head_dim)
        for (int h = 0; h < n_head; h++) {
            float *qh = q_buf + h * q_head_dim;
            // Norm the first norm_dim elements of each Q head
            float ss = 0.0f;
            for (int i = 0; i < norm_dim && i < q_head_dim; i++) ss += qh[i] * qh[i];
            float irms = 1.0f / sqrtf(ss / norm_dim + eps);
            for (int i = 0; i < norm_dim && i < q_head_dim; i++) qh[i] *= irms * nw[i];
        }
    }
    if (kn && kn->ptr) {
        const float *nw = (const float *)kn->ptr;
        int norm_dim = (int)kn->ne[0];
        for (int h = 0; h < n_head_kv; h++) {
            float *kh = k_buf + h * kv_head_dim;
            float ss = 0.0f;
            for (int i = 0; i < norm_dim && i < kv_head_dim; i++) ss += kh[i] * kh[i];
            float irms = 1.0f / sqrtf(ss / norm_dim + eps);
            for (int i = 0; i < norm_dim && i < kv_head_dim; i++) kh[i] *= irms * nw[i];
        }
    }

    // 4. RoPE on Q and K (only first rope_dim dimensions)
    beast_rope(q_buf, q_head_dim, n_head, pos, res->rope_freq_base, res->rope_dim);
    beast_rope(k_buf, kv_head_dim, n_head_kv, pos, res->rope_freq_base, res->rope_dim);

    // 5. Write K, V to cache
    // KV cache layout: [n_attn_layers, n_kv_dim, max_ctx]
    // Map layer index to attention-layer index for cache addressing
    int attn_interval = (int)res->full_attn_interval;
    int attn_idx = (attn_interval > 0) ? l / attn_interval : l;

    for (int h = 0; h < n_kv_dim; h++) {
        k_cache[((size_t)attn_idx * n_kv_dim + h) * max_ctx + (pos % max_ctx)] = k_buf[h];
        v_cache[((size_t)attn_idx * n_kv_dim + h) * max_ctx + (pos % max_ctx)] = v_buf[h];
    }

    // 6. Attention: Q @ K^T / sqrt(dim) → softmax → @ V
    int kv_len = (pos + 1 < max_ctx) ? pos + 1 : max_ctx;
    int heads_per_kv = (n_head_kv > 0) ? n_head / n_head_kv : 1;
    memset(head_out, 0, (size_t)n_q_out * sizeof(float));

    // For attention dot product, Q heads use kv_head_dim elements to match K
    // If q_head_dim > kv_head_dim, only the first kv_head_dim dims participate
    int attn_dim = (kv_head_dim < q_head_dim) ? kv_head_dim : q_head_dim;

    for (int qh = 0; qh < n_head; qh++) {
        int kvh = qh / heads_per_kv;
        float *q_head = q_buf + qh * q_head_dim;

        for (int p = 0; p < kv_len; p++) {
            float dot = 0.0f;
            for (int d = 0; d < attn_dim; d++) {
                float kval = k_cache[((size_t)attn_idx * n_kv_dim + kvh * kv_head_dim + d) * max_ctx + p];
                dot += q_head[d] * kval;
            }
            scores[p] = dot / sqrtf((float)attn_dim);
        }

        beast_softmax(scores, kv_len);

        // Weighted sum of V — output into q_head_dim space
        float *out_head = head_out + qh * q_head_dim;
        for (int p = 0; p < kv_len; p++) {
            float s = scores[p];
            for (int d = 0; d < kv_head_dim && d < q_head_dim; d++) {
                out_head[d] += s * v_cache[((size_t)attn_idx * n_kv_dim + kvh * kv_head_dim + d) * max_ctx + p];
            }
        }
    }

    // 7. Output projection
    snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
    const sp_optane_tensor_t *wo = sp_optane_find_tensor(res, name);
    if (wo && wo->ptr) {
        beast_matvec(wo, head_out, attn_out, scratch, n_embd, n_q_out);
    }

    // 8. Residual
    for (int i = 0; i < n_embd; i++) x[i] += attn_out[i];

    free(q_buf); free(k_buf); free(v_buf);
    free(scores); free(head_out); free(attn_out);
}

int sp_beast_forward(sp_beast_engine_t *engine,
                     int input_token,
                     float *logits)
{
    sp_optane_reservoir_t *res = &engine->reservoir;
    const int n_embd    = (int)res->n_embd;
    const int n_layer   = (int)res->n_layer;
    const int vocab     = (int)res->vocab_size;
    const int pos       = engine->beast_n_pos;

    float *x       = engine->beast_x;
    float *xn      = engine->beast_xn;
    float *scratch  = engine->beast_scratch;
    char name[128];

    // Determine layer type dispatch
    int attn_interval = (int)res->full_attn_interval;
    bool hybrid = res->is_hybrid && (attn_interval > 0);

    // 1. Token embedding
    const sp_optane_tensor_t *tok_embd = sp_optane_find_tensor(res, "token_embd.weight");
    if (beast_embed(tok_embd, input_token, n_embd, x) != 0) {
        fprintf(stderr, "[sp-beast] Forward: embedding lookup failed for token %d\n", input_token);
        return -1;
    }

    float *ffn_out = (float *)malloc((size_t)n_embd * sizeof(float));
    if (!ffn_out) return -1;

    // 2. Layer loop — hybrid SSM+Attention dispatch
    for (int l = 0; l < n_layer; l++) {
        // Determine if this is a full attention layer or SSM layer
        bool is_full_attn = false;
        if (hybrid) {
            // Full attention at layers: attn_interval-1, 2*attn_interval-1, ...
            // i.e. layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 for interval=4
            is_full_attn = ((l + 1) % attn_interval == 0);
        } else {
            is_full_attn = true;  // All layers are attention for standard models
        }

        // ── Attention or SSM block ──
        if (is_full_attn) {
            beast_attn_layer(engine, x, l, pos);
        } else {
            beast_ssm_layer(engine, x, l, pos);
        }

        // ── Post-attention/SSM norm (Qwen3.6 uses post_attention_norm) ──
        snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", l);
        const sp_optane_tensor_t *pan = sp_optane_find_tensor(res, name);
        if (pan && pan->ptr) {
            beast_rms_norm(x, (const float *)pan->ptr, n_embd, res->rms_norm_eps, xn);
        } else {
            // Fallback to ffn_norm if post_attention_norm doesn't exist
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
            const sp_optane_tensor_t *fn = sp_optane_find_tensor(res, name);
            if (fn && fn->ptr) {
                beast_rms_norm(x, (const float *)fn->ptr, n_embd, res->rms_norm_eps, xn);
            } else {
                memcpy(xn, x, (size_t)n_embd * sizeof(float));
            }
        }

        // ── MoE FFN block ──
        memset(ffn_out, 0, (size_t)n_embd * sizeof(float));

        if (res->is_moe) {
            beast_moe_ffn(engine, xn, ffn_out, l, n_embd);
        } else {
            // Dense FFN fallback
            snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
            const sp_optane_tensor_t *fg = sp_optane_find_tensor(res, name);
            snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
            const sp_optane_tensor_t *fu = sp_optane_find_tensor(res, name);
            snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
            const sp_optane_tensor_t *fd = sp_optane_find_tensor(res, name);
            if (fg && fu && fd && fg->ptr && fu->ptr && fd->ptr) {
                int n_ff = (int)fg->ne[1];
                float *gate_buf = (float *)malloc((size_t)n_ff * sizeof(float));
                float *up_buf   = (float *)malloc((size_t)n_ff * sizeof(float));
                if (gate_buf && up_buf) {
                    beast_matvec(fg, xn, gate_buf, scratch, n_ff, n_embd);
                    beast_matvec(fu, xn, up_buf,   scratch, n_ff, n_embd);
                    beast_silu_mul(gate_buf, up_buf, n_ff);
                    beast_matvec(fd, gate_buf, ffn_out, scratch, n_embd, n_ff);
                }
                free(gate_buf); free(up_buf);
            }
        }

        // ── FFN residual ──
        for (int i = 0; i < n_embd; i++) x[i] += ffn_out[i];

        if (l == 0 || (l + 1) == n_layer || (l % 10) == 0) {
            fprintf(stderr, "\r[sp-beast] Layer %d/%d  pos=%d  %s",
                    l + 1, n_layer, pos, is_full_attn ? "ATTN" : "SSM");
        }
    }

    // 3. Final RMSNorm
    const sp_optane_tensor_t *out_norm = sp_optane_find_tensor(res, "output_norm.weight");
    if (out_norm && out_norm->ptr) {
        beast_rms_norm(x, (const float *)out_norm->ptr, n_embd, res->rms_norm_eps, xn);
    } else {
        memcpy(xn, x, (size_t)n_embd * sizeof(float));
    }

    // 4. LM head → logits
    const sp_optane_tensor_t *output_w = sp_optane_find_tensor(res, "output.weight");
    if (!output_w) output_w = sp_optane_find_tensor(res, "token_embd.weight"); // tied
    if (output_w && output_w->ptr) {
        beast_matvec(output_w, xn, logits, scratch, vocab, n_embd);
    }

    engine->beast_n_pos = pos + 1;

    free(ffn_out);
    return 0;
}

// ============================================================================
// Generate — autoregressive token loop with timing
// ============================================================================

int sp_beast_generate(sp_beast_engine_t *engine,
                      const int *prompt_tokens, int n_prompt,
                      int *output_tokens, int max_tokens,
                      float temperature, float top_p)
{
    sp_optane_reservoir_t *res = &engine->reservoir;
    const int n_embd    = (int)res->n_embd;
    const int n_head_kv = (int)res->n_head_kv;
    const int head_dim  = (int)res->head_dim;
    const int n_layer   = (int)res->n_layer;
    const int vocab     = (int)res->vocab_size;

    if (n_embd == 0 || n_layer == 0 || vocab == 0) {
        fprintf(stderr, "[sp-beast] Generate: model metadata incomplete "
                "(n_embd=%d n_layer=%d vocab=%d)\n", n_embd, n_layer, vocab);
        return -1;
    }

    // ── Allocate runtime state ──
    // KV cache uses key_length (256 for Qwen3.6) not head_dim (128)
    int kv_head_dim = (res->key_length > 0) ? (int)res->key_length : head_dim;
    int kv_dim = n_head_kv * kv_head_dim;
    // Only attention layers need KV cache (10 of 40 for Qwen3.6)
    int attn_interval = (res->full_attn_interval > 0) ? (int)res->full_attn_interval : 1;
    int n_attn_layers = (attn_interval > 1) ? (n_layer / attn_interval) : n_layer;
    if (n_attn_layers < 1) n_attn_layers = n_layer;
    engine->beast_max_ctx = 256;
    engine->beast_n_pos   = 0;

    size_t kv_size = (size_t)n_attn_layers * kv_dim * engine->beast_max_ctx;
    engine->beast_k_cache = (float *)calloc(kv_size, sizeof(float));
    engine->beast_v_cache = (float *)calloc(kv_size, sizeof(float));
    int scratch_dim = (vocab > n_embd) ? vocab : n_embd;
    // scratch needs to hold at least conv_dim (8192) for GDN matvec
    int gdn_conv_dim = (int)res->ssm_inner_size + 2 * (int)res->ssm_n_groups * (int)res->ssm_state_size;
    if (gdn_conv_dim > scratch_dim) scratch_dim = gdn_conv_dim;
    engine->beast_scratch = (float *)malloc((size_t)scratch_dim * sizeof(float));
    engine->beast_x       = (float *)malloc((size_t)n_embd * sizeof(float));
    engine->beast_xn      = (float *)malloc((size_t)n_embd * sizeof(float));
    engine->beast_logits  = (float *)malloc((size_t)vocab * sizeof(float));

    // GDN (Gated Delta Net) recurrent state for hybrid SSM layers
    // Matches engine's GdnStateCache from gdn_state.h
    engine->beast_conv_state = NULL;
    engine->beast_ssm_state  = NULL;
    engine->beast_n_ssm_layers = 0;
    if (res->is_hybrid && res->ssm_state_size > 0) {
        int d_inner    = (int)res->ssm_inner_size;    // 4096
        int conv_k     = (int)res->ssm_conv_kernel;   // 4
        int n_groups   = (int)res->ssm_n_groups;       // 16
        int d_state    = (int)res->ssm_state_size;     // 128
        int dt_rank    = (int)res->ssm_dt_rank;        // 32 (num_v_heads)
        int conv_channels = d_inner + 2 * n_groups * d_state;  // 8192
        int head_v_dim    = d_inner / dt_rank;                  // 128

        // Allocate for ALL layers (indexed by layer number; attn layers waste ~0 bytes)
        engine->beast_n_ssm_layers = n_layer;
        size_t conv_per_layer = (size_t)(conv_k - 1) * conv_channels;     // 3*8192 = 24576
        size_t ssm_per_layer  = (size_t)dt_rank * head_v_dim * head_v_dim; // 32*128*128 = 524288

        engine->beast_conv_state = (float *)calloc((size_t)n_layer * conv_per_layer, sizeof(float));
        engine->beast_ssm_state  = (float *)calloc((size_t)n_layer * ssm_per_layer, sizeof(float));

        if (!engine->beast_conv_state || !engine->beast_ssm_state) {
            fprintf(stderr, "[sp-beast] Generate: OOM allocating GDN state "
                    "(conv=%.1f MB, ssm=%.1f MB)\n",
                    (double)n_layer * conv_per_layer * sizeof(float) / (1024.0*1024.0),
                    (double)n_layer * ssm_per_layer * sizeof(float) / (1024.0*1024.0));
            goto cleanup;
        }
        fprintf(stderr, "[sp-beast] GDN state: conv=%.1f MB  ssm=%.1f MB  (%d layers)\n",
                (double)n_layer * conv_per_layer * sizeof(float) / (1024.0*1024.0),
                (double)n_layer * ssm_per_layer * sizeof(float) / (1024.0*1024.0),
                n_layer - n_attn_layers);
    }

    if (!engine->beast_k_cache || !engine->beast_v_cache ||
        !engine->beast_scratch || !engine->beast_x ||
        !engine->beast_xn || !engine->beast_logits) {
        fprintf(stderr, "[sp-beast] Generate: OOM allocating runtime state\n");
        goto cleanup;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  BEAST CANYON INFERENCE                              ║\n");
    fprintf(stderr, "║  Model: %-42s  ║\n", res->architecture);
    fprintf(stderr, "║  Layers: %3d  Embd: %4d  Vocab: %6d              ║\n",
            n_layer, n_embd, vocab);
    if (res->is_moe) {
        fprintf(stderr, "║  MoE: %d experts (top-%d)  CRT: %s                  ║\n",
                res->n_experts, res->n_experts_per_token,
                (engine->barrier.n_gpus >= 2) ? "DUAL-GPU" : "CPU");
    }
    if (res->is_hybrid) {
        fprintf(stderr, "║  Hybrid: %d SSM + %d ATTN layers (interval=%d)       ║\n",
                n_layer - n_attn_layers, n_attn_layers, attn_interval);
    }
    fprintf(stderr, "║  KV cache: %d attn × %d × %d ctx = %.1f MB         ║\n",
            n_attn_layers, kv_dim, engine->beast_max_ctx,
            (double)kv_size * sizeof(float) / (1024.0 * 1024.0));
    fprintf(stderr, "╚══════════════════════════════════════════════════════╝\n\n");

    // ── Prefill: process prompt tokens ──
    engine->total_tokens = 0;
    engine->total_inference_us = 0;

    fprintf(stderr, "[sp-beast] Prefill: %d tokens\n", n_prompt);
    uint64_t prefill_start = sp_time_us();

    for (int i = 0; i < n_prompt; i++) {
        int rc = sp_beast_forward(engine, prompt_tokens[i], engine->beast_logits);
        if (rc != 0) {
            fprintf(stderr, "\n[sp-beast] Forward failed at prompt token %d\n", i);
            goto cleanup;
        }
        engine->total_tokens++;
    }

    uint64_t prefill_end = sp_time_us();
    double prefill_ms = (double)(prefill_end - prefill_start) / 1000.0;
    double prefill_tps = (n_prompt > 0) ?
        (double)n_prompt / (prefill_ms / 1000.0) : 0.0;

    fprintf(stderr, "\n[sp-beast] Prefill done: %.1f ms (%.2f tok/s)\n",
            prefill_ms, prefill_tps);

    // ── Prefill sidecar draft model if online ──
    bool sidecar_drafting = false;
    int spec_n_draft = 4;  // Tunable: how many tokens to draft per step
    int spec_accepted_total = 0;
    int spec_drafted_total  = 0;

    if (engine->sidecar.state == SP_SIDECAR_ONLINE) {
        // Prefill the phone's draft model with the same prompt
        if (sp_beast_sidecar_prefill(engine, prompt_tokens, n_prompt) == 0) {
            sidecar_drafting = true;
            fprintf(stderr, "[sp-beast] Speculative decode ENABLED: draft=%d tokens/step\n",
                    spec_n_draft);
        } else {
            fprintf(stderr, "[sp-beast] Sidecar prefill failed, falling back to standard decode\n");
        }
    }

    // ── Decode: autoregressive generation ──
    fprintf(stderr, "[sp-beast] Generating %d tokens...\n", max_tokens);
    int n_generated = 0;

    // Helper macro for EOS check
    #define IS_EOS(tok) ((tok) == 0 || (tok) == 2 || (tok) == 151643 || (tok) == 151645)

    while (n_generated < max_tokens) {
        int next_token = beast_argmax(engine->beast_logits, vocab);
        if (output_tokens) output_tokens[n_generated] = next_token;

        if (IS_EOS(next_token)) {
            fprintf(stderr, "\n[sp-beast] EOS token %d at position %d\n",
                    next_token, n_generated);
            n_generated++;
            break;
        }

        // ── Speculative decode path ──
        if (sidecar_drafting) {
            sp_sidecar_draft_result_t draft_result;
            if (sp_beast_sidecar_draft(engine, (int32_t)next_token,
                                       spec_n_draft, &draft_result) == 0 &&
                draft_result.n_drafted > 0)
            {
                spec_drafted_total += draft_result.n_drafted;

                // Run verified forward for the accepted token
                uint64_t tok_start = sp_time_us();
                int rc = sp_beast_forward(engine, next_token, engine->beast_logits);
                uint64_t tok_end = sp_time_us();
                if (rc != 0) break;
                engine->total_tokens++;
                engine->total_inference_us += (tok_end - tok_start);
                n_generated++;

                // Verify each draft token against the full model
                int n_accepted = 0;
                int32_t verified[SP_SIDECAR_MAX_DRAFT];

                for (int d = 0; d < draft_result.n_drafted && n_generated < max_tokens; d++) {
                    int host_pred = beast_argmax(engine->beast_logits, vocab);

                    if (host_pred == draft_result.tokens[d]) {
                        // Draft matches — accept, run forward to advance KV
                        verified[n_accepted] = (int32_t)host_pred;
                        n_accepted++;

                        if (output_tokens) output_tokens[n_generated] = host_pred;

                        tok_start = sp_time_us();
                        rc = sp_beast_forward(engine, host_pred, engine->beast_logits);
                        tok_end = sp_time_us();
                        if (rc != 0) break;
                        engine->total_tokens++;
                        engine->total_inference_us += (tok_end - tok_start);
                        n_generated++;

                        if (IS_EOS(host_pred)) break;
                    } else {
                        // Mismatch — reject remaining drafts, use host's prediction
                        break;
                    }
                }

                spec_accepted_total += n_accepted;

                // Tell sidecar how many we accepted
                if (n_accepted > 0) {
                    sp_beast_sidecar_accept(engine, verified, n_accepted);
                }

                double acc = (spec_drafted_total > 0) ?
                    100.0 * spec_accepted_total / spec_drafted_total : 0.0;
                fprintf(stderr, "\r[sp-beast] Token %d  spec: %d/%d accepted (%.0f%% hit)  "
                        "rt=%.0fus  %.2f tok/s",
                        n_generated, n_accepted, draft_result.n_drafted, acc,
                        (double)draft_result.round_trip_us,
                        (engine->total_inference_us > 0) ?
                            (double)n_generated / ((double)engine->total_inference_us / 1e6) : 0.0);

                continue;  // Next decode step
            }
            // Draft failed — fall through to standard decode for this step
        }

        // ── Standard single-token decode path ──
        uint64_t tok_start = sp_time_us();
        int rc = sp_beast_forward(engine, next_token, engine->beast_logits);
        uint64_t tok_end = sp_time_us();

        if (rc != 0) {
            fprintf(stderr, "\n[sp-beast] Forward failed at generated token %d\n",
                    n_generated);
            break;
        }

        engine->total_tokens++;
        engine->total_inference_us += (tok_end - tok_start);
        n_generated++;

        double tok_ms = (double)(tok_end - tok_start) / 1000.0;
        double avg_tps = (engine->total_inference_us > 0) ?
            (double)(n_generated) /
            ((double)engine->total_inference_us / 1e6) : 0.0;

        fprintf(stderr, "\r[sp-beast] Token %d: id=%6d  %.0f ms  (%.2f tok/s avg)",
                n_generated, next_token, tok_ms, avg_tps);
    }

    #undef IS_EOS

    // ── Final report ──
    fprintf(stderr, "\n\n");
    sp_beast_print_status(engine);

    double total_s = (double)engine->total_inference_us / 1e6;
    fprintf(stderr, "\n╔══════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  RESULTS                                              ║\n");
    fprintf(stderr, "║  Prefill:   %4d tokens in %8.1f ms (%6.2f tok/s)  ║\n",
            n_prompt, prefill_ms, prefill_tps);
    fprintf(stderr, "║  Decode:    %4d tokens in %8.1f ms (%6.2f tok/s)  ║\n",
            n_generated, total_s * 1000.0,
            (total_s > 0) ? (double)n_generated / total_s : 0.0);
    fprintf(stderr, "║  CRT shreds: %llu  elements: %llu                       ║\n",
            (unsigned long long)engine->shredder_crt.total_shreds,
            (unsigned long long)engine->shredder_crt.total_elements);
    if (spec_drafted_total > 0) {
        double acc = 100.0 * spec_accepted_total / spec_drafted_total;
        fprintf(stderr, "║  Sidecar:   %4d drafted, %4d accepted (%.1f%%)       ║\n",
                spec_drafted_total, spec_accepted_total, acc);
        fprintf(stderr, "║  Sidecar:   %llu offloads, avg %.0f us/rt              ║\n",
                (unsigned long long)engine->sidecar.total_offloads,
                engine->sidecar.total_offloads > 0 ?
                    (double)engine->sidecar.total_offload_us /
                    (double)engine->sidecar.total_offloads : 0.0);
    }
    fprintf(stderr, "╚══════════════════════════════════════════════════════╝\n");

cleanup:
    free(engine->beast_k_cache);    engine->beast_k_cache = NULL;
    free(engine->beast_v_cache);    engine->beast_v_cache = NULL;
    free(engine->beast_conv_state); engine->beast_conv_state = NULL;
    free(engine->beast_ssm_state);  engine->beast_ssm_state = NULL;
    free(engine->beast_scratch);    engine->beast_scratch = NULL;
    free(engine->beast_x);          engine->beast_x = NULL;
    free(engine->beast_xn);         engine->beast_xn = NULL;
    free(engine->beast_logits);     engine->beast_logits = NULL;

    return n_generated;
}

// ============================================================================
// ASCII Dashboard
// ============================================================================

void sp_beast_dashboard(const sp_beast_engine_t *engine) {
    double tok_s = sp_beast_tok_per_sec(engine);
    double shred_gbps = sp_shredder_throughput_gbps(&engine->shredder);
    double avg_barrier = sp_hetero_avg_barrier_us(&engine->barrier);

    fprintf(stderr, "\r");
    fprintf(stderr, "[ OPTANE ] ────── ");
    fprintf(stderr, "[ SHREDDER %.1f GB/s ] ────── ", shred_gbps);
    fprintf(stderr, "[ LLC ] ────── ");

    if (engine->barrier.n_gpus >= 1) {
        fprintf(stderr, "[ GPU0: %s ] ", engine->barrier.gpu[0].name);
    }
    if (engine->barrier.n_gpus >= 2) {
        fprintf(stderr, "[ GPU1: %s ] ", engine->barrier.gpu[1].name);
    }

    fprintf(stderr, "  %.1f tok/s  barrier=%.0fus  ", tok_s, avg_barrier);

    if (engine->sidecar.state == SP_SIDECAR_ONLINE) {
        fprintf(stderr, "[ S22U: ONLINE ]");
    }

    fflush(stderr);
}

// ============================================================================
// Full status print
// ============================================================================

void sp_beast_print_status(const sp_beast_engine_t *engine) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  BEAST CANYON ENGINE STATUS                  ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Tokens:     %10llu                       ║\n",
            (unsigned long long)engine->total_tokens);
    fprintf(stderr, "║  Tok/s:      %10.1f                       ║\n",
            sp_beast_tok_per_sec(engine));
    fprintf(stderr, "║  Inference:  %10.2f ms                    ║\n",
            (double)engine->total_inference_us / 1000.0);
    fprintf(stderr, "║  Barrier:    %10.2f ms (avg %.0f us)       ║\n",
            (double)engine->total_barrier_us / 1000.0,
            sp_hetero_avg_barrier_us(&engine->barrier));
    fprintf(stderr, "╠══════════════════════════════════════════════╣\n");

    sp_optane_print_status(&engine->reservoir);
    sp_shredder_print_status(&engine->shredder);
    sp_hetero_barrier_print_status(&engine->barrier);
}
