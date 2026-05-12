// Shannon-Prime BurnHard — GGML_OP_BURN_MOE custom op.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Replaces the routed-expert FFN block (3 × ggml_mul_mat_id + SiLU + mul +
// weighted sum) with a host-side residue-matmul implementation. The shared
// expert (ffn_*_shexp) is left on the standard ggml path.
//
// API:
//   sp_build_burn_moe_op(gctx, cur, selected, weights,
//                         gate_exps, up_exps, down_exps,
//                         n_expert_used)
//
// Returns a [n_embd, n_tokens] tensor that is the routed-expert FFN output —
// the same shape build_moe_ffn would produce for the routed experts ALONE.
// The caller is responsible for adding the shared-expert contribution after.
//
// During graph compute, the callback dequantises the selected experts on
// the host (via ggml's public type_traits->to_float), runs the BurnHard
// residue matmul (Sessions 1-2.5: M1+M2 chambers, Garner-joined), applies
// SiLU + multiply + weighted sum, and writes the result. All inputs are
// pulled to host via ggml_backend_tensor_get; the result is written via
// ggml_backend_tensor_set. One host round-trip per MoE layer per token.

#ifndef SP_BURN_MOE_OP_H
#define SP_BURN_MOE_OP_H

#include "ggml.h"
#include "sp_crt_dispatch.h"   // sp_crt_dispatch_t — pulled in for the public API

#ifdef __cplusplus
extern "C" {
#endif

// Build the BURN_MOE custom op into the graph. All `ggml_tensor *` args
// follow ggml's normal lifetime — owned by the caller's gctx. n_expert_used
// is the top-K selection count (e.g. 8 for Qwen3.6 35B-A3B).
// gate/up/down_exps   — GPU-resident sub-bank covering experts [pivot, n_expert).
//                       Pass the full bundled tensor when no split is active
//                       (pivot=0); BURN_MOE then iterates over all n_expert.
// gate/up/down_exps_cpu — CPU-resident sub-bank covering experts [0, pivot).
//                       Pass nullptr when no split is active. When non-null,
//                       BURN_MOE dispatches per-expert: ids < pivot go to the
//                       CPU bank (host residue matmul, hot path stays on host),
//                       ids >= pivot go to the GPU bank (residue dispatched
//                       through `crt` to dual-ring when armed).
// pivot              — number of experts kept CPU-side. 0 = no split.
struct ggml_tensor *sp_build_burn_moe_op(struct ggml_context *gctx,
                                          struct ggml_tensor *cur,            // [n_embd, n_tokens]
                                          struct ggml_tensor *selected,       // [k, n_tokens] I32
                                          struct ggml_tensor *weights,        // [1, k, n_tokens] f32
                                          struct ggml_tensor *gate_exps,      // [n_embd, n_ff, n_expert - pivot]
                                          struct ggml_tensor *up_exps,        // [n_embd, n_ff, n_expert - pivot]
                                          struct ggml_tensor *down_exps,      // [n_ff, n_embd, n_expert - pivot]
                                          struct ggml_tensor *gate_exps_cpu,  // [n_embd, n_ff, pivot] or NULL
                                          struct ggml_tensor *up_exps_cpu,    // [n_embd, n_ff, pivot] or NULL
                                          struct ggml_tensor *down_exps_cpu,  // [n_ff, n_embd, pivot] or NULL
                                          int n_expert_used,
                                          int pivot,
                                          sp_crt_dispatch_t *crt);   // optional, NULL=host fallback

// ----------------------------------------------------------------------------
// BEAST_MOE custom op — wraps sp_beast_moe_forward.
//
// When the chat path is run with --beast, the Beast Canyon orchestrator boots
// alongside but the per-token MoE FFN still goes through the standard ggml
// path. This op bridges that gap: it replaces the MoE FFN block with a single
// host-side custom op that calls sp_beast_moe_forward(beast_engine, ...) once
// per token. Inside, Beast Canyon does its own routing, Optane-resident
// expert lookup, AVX-512 matvec / dual-GPU shred, and accumulation.
//
// Inputs:
//   cur            — [n_embd, n_tokens] f32 hidden states
//   router_logits  — [n_expert, n_tokens] f32 (gate_inp @ cur)
// Output:
//   [n_embd, n_tokens] f32 routed-expert FFN result (no shared expert).
//
// `engine` is opaque to the engine-side TU (forward declaration); the C
// header sp_beast_canyon.h provides the real type.
struct sp_beast_engine_t;
struct ggml_tensor *sp_build_beast_moe_op(struct ggml_context *gctx,
                                           struct ggml_tensor *cur,
                                           struct ggml_tensor *router_logits,
                                           struct sp_beast_engine_t *engine,
                                           int layer_idx);

#ifdef __cplusplus
}
#endif

#endif // SP_BURN_MOE_OP_H
