// Shannon-Prime BurnHard — GGML_OP_BURN_MOE custom op (impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Host-side implementation of the routed-expert FFN block via the BurnHard
// residue-matmul kernel (Sessions 1-2.5). One ggml_custom_4d node replaces
// the three ggml_mul_mat_id + SiLU + mul + weighted-sum sequence. The
// shared expert (ffn_*_shexp) stays on the standard ggml path and is
// added by the caller AFTER this op produces its output.
//
// Performance note: this is the host-roundtrip path. On Qwen3.6 35B-A3B
// it'll be slow (each MoE layer pulls cur + 8 expert weight tensors to
// host, dequants, runs three matmuls per (token, expert), pushes result
// back). Session 3 swaps the inner residue matmul for CUDA / Level Zero
// kernels; the surrounding op machinery stays.

#include "burn_moe_op.h"

#include "burnhard_residue_matmul.h"
#include "burnhard_crt.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ----------------------------------------------------------------------------
// SiLU on a flat float buffer.
// ----------------------------------------------------------------------------
static inline void apply_silu_inplace(float *buf, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const float x = buf[i];
        buf[i] = x / (1.0f + std::exp(-x));
    }
}

// ----------------------------------------------------------------------------
// Pull a quantised expert slice to host fp32 via ggml's public type_traits.
//
// ggml stores ffn_gate_exps / ffn_up_exps as 3D [n_embd, n_ff, n_expert] —
// expert e occupies a contiguous block of n_embd*n_ff elements at byte
// offset e * nb[2]. ffn_down_exps is [n_ff, n_embd, n_expert], same layout
// principle; the per-expert slice is still n_ff*n_embd elements.
//
// `tensor` is the parent expert-bank tensor. `expert_idx` selects which
// expert. `out_fp32` is caller-allocated; size must equal n_inner*n_outer.
// ----------------------------------------------------------------------------
static int dequant_expert_slice(const ggml_tensor *tensor, int expert_idx,
                                 size_t n_elements, float *out_fp32) {
    const ggml_type t = tensor->type;
    const struct ggml_type_traits *traits = ggml_get_type_traits(t);
    if (!traits || !traits->to_float) return -1;

    // Bytes per expert = product of ne[0] * ne[1] elements / blck_size * type_size.
    const int64_t blk = (int64_t)traits->blck_size;
    const int64_t bytes_per_expert =
        (int64_t)(tensor->ne[0] * tensor->ne[1] / blk) * (int64_t)traits->type_size;
    const uint8_t *base = (const uint8_t *)tensor->data;
    if (!base) {
        // Tensor data may live on a backend buffer (CUDA/Vulkan). We need
        // to copy it to host first. ggml_backend_tensor_get into a temp
        // buffer of size bytes_per_expert at offset expert_idx*bytes.
        std::vector<uint8_t> tmp((size_t)bytes_per_expert);
        ggml_backend_tensor_get(tensor, tmp.data(),
                                 (size_t)expert_idx * (size_t)bytes_per_expert,
                                 (size_t)bytes_per_expert);
        traits->to_float(tmp.data(), out_fp32, (int64_t)n_elements);
        return 0;
    }
    const uint8_t *expert_bytes = base + (size_t)expert_idx * (size_t)bytes_per_expert;
    traits->to_float(expert_bytes, out_fp32, (int64_t)n_elements);
    return 0;
}

// ----------------------------------------------------------------------------
// Per-expert FFN forward (one token):
//   gate_out = gate_W @ x          (residue matmul)
//   up_out   = up_W   @ x          (residue matmul)
//   par      = silu(gate_out) * up_out
//   contrib  = down_W @ par        (residue matmul)
// Returns contrib in `out_fp32` (caller allocated, n_embd elements).
// ----------------------------------------------------------------------------
static int expert_ffn_one_token(const float *x, size_t n_embd, size_t n_ff,
                                  const float *gate_W, const float *up_W,
                                  const float *down_W,
                                  float *out_fp32) {
    std::vector<float> gate_out(n_ff);
    std::vector<float> up_out  (n_ff);
    std::vector<float> par     (n_ff);
    int rc = 0;
    rc |= sp_burnhard_residue_matmul_fp32(gate_W, x, gate_out.data(),
                                            n_ff, n_embd, 1);
    rc |= sp_burnhard_residue_matmul_fp32(up_W,   x, up_out.data(),
                                            n_ff, n_embd, 1);
    if (rc != 0) return rc;
    apply_silu_inplace(gate_out.data(), n_ff);
    for (size_t i = 0; i < n_ff; ++i) par[i] = gate_out[i] * up_out[i];
    rc |= sp_burnhard_residue_matmul_fp32(down_W, par.data(), out_fp32,
                                            n_embd, n_ff, 1);
    return rc;
}

// ----------------------------------------------------------------------------
// User data carried through the custom op. The graph builder allocates one
// of these per layer and threads it through ggml_custom_4d. Lifetime ties
// to the graph's gctx (we leak intentionally; gctx_free reclaims).
// ----------------------------------------------------------------------------
struct burn_moe_userdata {
    int n_expert_used;
};

// ----------------------------------------------------------------------------
// Memory safety cap. Per-call dequant cache is bounded to this many unique
// experts. Each entry holds three fp32 buffers of n_embd*n_ff. For Qwen3.6
// 35B-A3B (n_embd=2048, n_ff=512) that's 3 × 4 MiB = 12 MiB per cached
// expert. Cap × 12 MiB = worst-case per-call host scratch.
//
// 32 entries → ~384 MiB ceiling — safe on a 32 GB host even with the
// 19.7 GB GGUF mmap + ggml CUDA backend + Beast Canyon co-resident.
// Realistic decode (n_tokens=1) needs only 8 unique experts per call so
// the cap rarely matters; the cap exists to bound prefill of long
// sequences where naive caching would saturate at n_expert (256 = ~3 GB).
//
// A previous integration test on 35B with audit mode (which runs BOTH
// the host residue path AND ggml's mul_mat_id) saturated host RAM and
// took down the system. This cap (plus the budget log below) means the
// Furnace can be invoked on any model size without that risk.
static constexpr size_t SP_BURN_MOE_DEQUANT_CACHE_CAP = 32;

// First-call diagnostic guard. Set to 1 by burn_moe_compute on first
// invocation so we don't spam the budget line per layer per token.
static int s_burn_moe_logged_budget = 0;

// ----------------------------------------------------------------------------
// The compute callback. Runs once per graph compute (single-thread for
// now — ith == 0 path; multi-thread parallelisation is a Session-2.7
// optimisation once we have correctness pinned).
// ----------------------------------------------------------------------------
static void burn_moe_compute(ggml_tensor *dst, int ith, int nth, void *userdata) {
    if (ith != 0) return;
    (void)nth;
    burn_moe_userdata *ud = (burn_moe_userdata *)userdata;
    const int n_expert_used = ud->n_expert_used;

    // dst layout: [n_embd, n_tokens] fp32.
    const ggml_tensor *cur       = dst->src[0];   // [n_embd, n_tokens] f32
    const ggml_tensor *selected  = dst->src[1];   // [k, n_tokens] I32
    const ggml_tensor *weights   = dst->src[2];   // [1, k, n_tokens] f32
    const ggml_tensor *gate_exps = dst->src[3];
    const ggml_tensor *up_exps   = dst->src[4];
    const ggml_tensor *down_exps = dst->src[5];

    const size_t n_embd   = (size_t)cur->ne[0];
    const size_t n_tokens = (size_t)cur->ne[1];
    const size_t n_ff     = (size_t)gate_exps->ne[1];

    // First-call budget log — surfaces the per-call host scratch ceiling
    // BEFORE any allocation, so OOMs are predictable instead of silent.
    if (!s_burn_moe_logged_budget) {
        const size_t per_expert_fp32_bytes = 3 * n_embd * n_ff * sizeof(float);
        const size_t cache_cap_bytes =
            SP_BURN_MOE_DEQUANT_CACHE_CAP * per_expert_fp32_bytes;
        const size_t io_buffers_bytes =
            (n_embd * n_tokens * sizeof(float))           // cur_host
            + (n_expert_used * n_tokens * sizeof(int32_t))// sel_host
            + (n_expert_used * n_tokens * sizeof(float))  // wgt_host
            + (n_embd * n_tokens * sizeof(float));        // out_host
        std::fprintf(stderr,
            "[burn_moe] budget per call: I/O=%.1f MiB  "
            "dequant_cache_cap=%zu × %.1f MiB = %.1f MiB  "
            "(n_embd=%zu n_ff=%zu n_tokens=%zu k=%d)\n",
            (double)io_buffers_bytes / (1024.0 * 1024.0),
            SP_BURN_MOE_DEQUANT_CACHE_CAP,
            (double)per_expert_fp32_bytes / (1024.0 * 1024.0),
            (double)cache_cap_bytes / (1024.0 * 1024.0),
            n_embd, n_ff, n_tokens, n_expert_used);
        s_burn_moe_logged_budget = 1;
    }

    // Sanity — these must match what the graph claimed.
    if ((size_t)dst->ne[0] != n_embd || (size_t)dst->ne[1] != n_tokens) {
        std::fprintf(stderr,
            "[burn_moe] dst shape mismatch: dst=[%lld,%lld] cur=[%lld,%lld]\n",
            (long long)dst->ne[0], (long long)dst->ne[1],
            (long long)cur->ne[0], (long long)cur->ne[1]);
        return;
    }

    // Pull cur, selected, weights to host. dst's data is the output buffer
    // (already allocated by the gallocr); we write directly into it.
    std::vector<float>   cur_host(n_embd * n_tokens);
    std::vector<int32_t> sel_host(n_expert_used * n_tokens);
    std::vector<float>   wgt_host(n_expert_used * n_tokens);
    ggml_backend_tensor_get(cur,      cur_host.data(), 0,
                              cur_host.size() * sizeof(float));
    ggml_backend_tensor_get(selected, sel_host.data(), 0,
                              sel_host.size() * sizeof(int32_t));
    ggml_backend_tensor_get(weights,  wgt_host.data(), 0,
                              wgt_host.size() * sizeof(float));

    // Output buffer (host scratch — we'll set into dst at the end).
    std::vector<float> out_host(n_embd * n_tokens, 0.0f);

    // Per-expert dequant scratch — reused across (token, expert) steps.
    std::vector<float> gate_W(n_embd * n_ff);
    std::vector<float> up_W  (n_embd * n_ff);
    std::vector<float> down_W(n_embd * n_ff);
    std::vector<float> contrib(n_embd);

    // Bounded LRU dequant cache. Many tokens select the same experts so
    // we cache the dequantised fp32 weights and reuse on hit. Cap is
    // SP_BURN_MOE_DEQUANT_CACHE_CAP; on miss + full cache we evict the
    // least-recently-used entry so memory stays bounded regardless of
    // prompt length or expert-routing pathology.
    struct expert_cache_entry {
        int expert_idx;
        uint64_t lru_tick;          // monotonic counter; smallest = oldest
        std::vector<float> gate;
        std::vector<float> up;
        std::vector<float> down;
    };
    std::vector<expert_cache_entry> cache;
    cache.reserve(SP_BURN_MOE_DEQUANT_CACHE_CAP);
    uint64_t lru_clock = 0;
    auto find_or_dequant = [&](int e) -> const expert_cache_entry * {
        // Hit?
        for (auto &c : cache) {
            if (c.expert_idx == e) {
                c.lru_tick = ++lru_clock;
                return &c;
            }
        }
        // Miss. Evict if at cap.
        if (cache.size() >= SP_BURN_MOE_DEQUANT_CACHE_CAP) {
            size_t oldest_idx = 0;
            uint64_t oldest_tick = cache[0].lru_tick;
            for (size_t i = 1; i < cache.size(); ++i) {
                if (cache[i].lru_tick < oldest_tick) {
                    oldest_tick = cache[i].lru_tick;
                    oldest_idx  = i;
                }
            }
            // Swap-and-pop to keep amortised O(1) eviction.
            cache[oldest_idx] = std::move(cache.back());
            cache.pop_back();
        }
        cache.emplace_back();
        auto &c = cache.back();
        c.expert_idx = e;
        c.lru_tick   = ++lru_clock;
        c.gate.resize(n_embd * n_ff);
        c.up  .resize(n_embd * n_ff);
        c.down.resize(n_embd * n_ff);
        if (dequant_expert_slice(gate_exps, e, n_embd * n_ff, c.gate.data()) != 0 ||
            dequant_expert_slice(up_exps,   e, n_embd * n_ff, c.up.data())   != 0 ||
            dequant_expert_slice(down_exps, e, n_embd * n_ff, c.down.data()) != 0) {
            std::fprintf(stderr, "[burn_moe] dequant failed for expert %d\n", e);
            cache.pop_back();
            return nullptr;
        }
        return &c;
    };

    // Per-token loop: dispatch the n_expert_used selected experts and
    // accumulate weighted contributions into out_host[t*n_embd : ...].
    for (size_t t = 0; t < n_tokens; ++t) {
        const float *x = &cur_host[t * n_embd];
        float       *y = &out_host[t * n_embd];
        for (int k = 0; k < n_expert_used; ++k) {
            const int32_t e = sel_host[t * n_expert_used + k];
            const float    w = wgt_host[t * n_expert_used + k];
            const auto *ce = find_or_dequant(e);
            if (!ce) continue;
            if (expert_ffn_one_token(x, n_embd, n_ff,
                                       ce->gate.data(), ce->up.data(), ce->down.data(),
                                       contrib.data()) != 0) {
                continue;
            }
            for (size_t i = 0; i < n_embd; ++i) y[i] += w * contrib[i];
        }
    }

    // Push the host buffer into dst.
    ggml_backend_tensor_set(dst, out_host.data(), 0,
                              out_host.size() * sizeof(float));
}

// ----------------------------------------------------------------------------
// Public — graph builder.
// ----------------------------------------------------------------------------
extern "C" ggml_tensor *sp_build_burn_moe_op(ggml_context *gctx,
                                              ggml_tensor *cur,
                                              ggml_tensor *selected,
                                              ggml_tensor *weights,
                                              ggml_tensor *gate_exps,
                                              ggml_tensor *up_exps,
                                              ggml_tensor *down_exps,
                                              int n_expert_used) {
    // User data lives in heap memory tied to the graph's lifetime via a
    // simple leak — gctx doesn't track our allocations. The Engine resets
    // its forward graph each call so this leak is bounded to one graph's
    // worth of MoE layers (~40 × ~96 bytes = 4 KB). Not worth a free hook.
    burn_moe_userdata *ud = new burn_moe_userdata;
    ud->n_expert_used = n_expert_used;

    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];

    ggml_tensor *args[6] = { cur, selected, weights, gate_exps, up_exps, down_exps };
    return ggml_custom_4d(gctx, GGML_TYPE_F32,
                           n_embd, n_tokens, 1, 1,
                           args, 6,
                           burn_moe_compute,
                           /*n_tasks=*/1, ud);
}
