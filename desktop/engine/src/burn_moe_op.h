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
struct ggml_tensor *sp_build_burn_moe_op(struct ggml_context *gctx,
                                          struct ggml_tensor *cur,        // [n_embd, n_tokens]
                                          struct ggml_tensor *selected,   // [k, n_tokens] I32
                                          struct ggml_tensor *weights,    // [1, k, n_tokens] f32
                                          struct ggml_tensor *gate_exps,  // [n_embd, n_ff, n_expert]
                                          struct ggml_tensor *up_exps,    // [n_embd, n_ff, n_expert]
                                          struct ggml_tensor *down_exps,  // [n_ff, n_embd, n_expert]
                                          int n_expert_used,
                                          sp_crt_dispatch_t *crt);   // optional, NULL=host fallback

#ifdef __cplusplus
}
#endif

#endif // SP_BURN_MOE_OP_H
