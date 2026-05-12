// Shannon-Prime Engine — SP-Flash speculative decoding oracle (Phase 1)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Phase 1 exact-match speculative decoding using the dflash-draft-3.6
// 8-layer bidirectional transformer. Target hidden states are captured
// from SpFlashCaptureCtx, projected via FC, and fed into the draft model
// alongside mask-token embeddings. Draft tokens are accepted against the
// target model's greedy choices.

#include "sp_flash_oracle.h"
#include "dflash_weights.h"
#include "forward.h"  // SpFlashCaptureCtx

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace sp::engine {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static int32_t argmax_f32(const float* data, int n) {
    int32_t best = 0;
    float   best_val = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_val) {
            best_val = data[i];
            best    = i;
        }
    }
    return best;
}

// Exact-match acceptance: compare greedy draft vs greedy target logits.
// Returns tokens accepted (all matching) plus the target's choice at the
// first divergence (or the target's choice after the last matching position).
static std::vector<int32_t> accept_exact(
    const std::vector<int32_t>& draft,
    const std::vector<float>&   target_logits,  // [block_size * n_vocab]
    int n_vocab)
{
    std::vector<int32_t> result;
    result.reserve(draft.size() + 1);
    const int block = (int)draft.size();
    for (int i = 0; i < block; ++i) {
        int32_t target_tok = argmax_f32(target_logits.data() + (size_t)i * n_vocab, n_vocab);
        if (draft[i] == target_tok) {
            result.push_back(draft[i]);
        } else {
            result.push_back(target_tok);  // resampled from target at mismatch
            return result;
        }
    }
    // All draft tokens matched — append the target's next token after the block.
    // The last target logits row (index block-1) already yielded the matching
    // token; we need nothing further here per the spec (length = block_size when
    // all match, caller handles the next prefill). Return as-is.
    return result;
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

// Descriptor-only arena for ggml_init(no_alloc=true).
// ~200 tensors × ~400 bytes each = ~80 KB max; 512 KB is safe headroom.
static constexpr size_t kCtxSize = 512 * 1024;

struct SpFlashOracle::Impl {
    std::unique_ptr<DFlashWeights> weights;
    SpFlashCaptureCtx*             capture      = nullptr;

    // Draft model config (cached from weights)
    int hidden_size  = 2048;
    int n_heads      = 32;
    int n_heads_kv   = 4;
    int head_dim     = 128;
    int block_size   = 8;
    int mask_tok_id  = 0;
    int n_target_layers = 5;   // number of target layers captured

    // Target model tensors (non-owning, must outlive Impl)
    ggml_tensor* target_tok_embd = nullptr;
    ggml_tensor* target_output   = nullptr;

    // ggml CPU backend + graph allocator (reused across steps)
    ggml_backend_t   backend = nullptr;
    ggml_gallocr_t   allocr  = nullptr;

    // Scratch arena for graph tensor descriptors (rebuilt each step)
    std::vector<uint8_t> ctx_mem;

    // Adaptive EMA tracking
    bool  adaptive      = false;
    float adaptive_hi   = 0.75f;
    float adaptive_lo   = 0.50f;
    int   adaptive_steps= 50;
    int   cur_block_size= 8;
    float ema_accept    = 0.0f;
    int   step_count    = 0;

    ~Impl() {
        if (allocr)  ggml_gallocr_free(allocr);
        if (backend) ggml_backend_free(backend);
        // weights is owned by unique_ptr and cleaned up via unload_dflash_weights
    }
};

// ---------------------------------------------------------------------------
// SpFlashOracle public API
// ---------------------------------------------------------------------------

SpFlashOracle::SpFlashOracle() : impl_(std::make_unique<Impl>()) {}
SpFlashOracle::~SpFlashOracle() = default;

std::unique_ptr<SpFlashOracle> SpFlashOracle::load(const Config& cfg) {
    if (cfg.draft_path.empty()) {
        std::fprintf(stderr, "[sp-oracle] draft_path is empty\n");
        return nullptr;
    }
    if (!cfg.target_tok_embd || !cfg.target_output) {
        std::fprintf(stderr, "[sp-oracle] target_tok_embd / target_output must be non-null\n");
        return nullptr;
    }

    auto oracle = std::unique_ptr<SpFlashOracle>(new SpFlashOracle());
    Impl* I = oracle->impl_.get();

    // Load draft weights
    I->weights = load_dflash_weights(cfg.draft_path);
    if (!I->weights) {
        std::fprintf(stderr, "[sp-oracle] failed to load draft weights from '%s'\n",
                     cfg.draft_path.c_str());
        return nullptr;
    }
    auto* W = I->weights.get();

    // Cache hparams
    I->hidden_size       = W->hidden_size > 0 ? W->hidden_size : 2048;
    I->n_heads           = W->n_heads     > 0 ? W->n_heads     : 32;
    I->n_heads_kv        = W->n_heads_kv  > 0 ? W->n_heads_kv  : 4;
    I->head_dim          = W->head_dim    > 0 ? W->head_dim     : 128;
    I->block_size        = cfg.block_size > 0 ? cfg.block_size  : (W->block_size > 0 ? W->block_size : 8);
    I->mask_tok_id       = W->mask_token_id;
    I->n_target_layers   = (int)W->target_layer_ids.size();
    if (I->n_target_layers == 0) I->n_target_layers = 5;

    I->target_tok_embd   = cfg.target_tok_embd;
    I->target_output     = cfg.target_output;

    I->adaptive          = cfg.adaptive;
    I->adaptive_hi       = cfg.adaptive_hi;
    I->adaptive_lo       = cfg.adaptive_lo;
    I->adaptive_steps    = cfg.adaptive_steps;
    I->cur_block_size    = I->block_size;

    // CPU backend
    I->backend = ggml_backend_cpu_init();
    if (!I->backend) {
        std::fprintf(stderr, "[sp-oracle] failed to init CPU backend\n");
        return nullptr;
    }

    // Allocator (reused across steps — topology is fixed for constant block_size)
    I->allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!I->allocr) {
        std::fprintf(stderr, "[sp-oracle] failed to create gallocr\n");
        return nullptr;
    }

    // Scratch arena for graph ggml_contexts
    I->ctx_mem.resize(kCtxSize);

    std::fprintf(stderr,
        "[sp-oracle] loaded: hidden=%d heads=%d/%d head_dim=%d block=%d "
        "n_target_layers=%d mask_tok=%d\n",
        I->hidden_size, I->n_heads, I->n_heads_kv, I->head_dim,
        I->block_size, I->n_target_layers, I->mask_tok_id);

    return oracle;
}

void SpFlashOracle::bind_capture(SpFlashCaptureCtx* ctx) {
    impl_->capture = ctx;
}

float SpFlashOracle::acceptance_rate() const {
    return impl_->ema_accept;
}

int SpFlashOracle::current_block_size() const {
    return impl_->cur_block_size;
}

const std::vector<int>& SpFlashOracle::target_layer_ids() const {
    return impl_->weights->target_layer_ids;
}

// ---------------------------------------------------------------------------
// step() — core Phase 1 forward
// ---------------------------------------------------------------------------

std::vector<int32_t> SpFlashOracle::step(
    const std::function<std::vector<float>(const std::vector<int32_t>&)>& target_logits_fn,
    int32_t /*last_token*/)
{
    Impl* I = impl_.get();
    auto* W = I->weights.get();

    if (!I->capture || !I->capture->active) {
        std::fprintf(stderr, "[sp-oracle] step(): capture context not bound or not active\n");
        return {};
    }
    SpFlashCaptureCtx* cap = I->capture;

    const int  hidden     = I->hidden_size;
    const int  n_tl       = I->n_target_layers;
    const int  bs         = I->cur_block_size;
    const int  seq_len    = 1 + bs;       // context + mask positions
    const int  n_heads    = I->n_heads;
    const int  n_heads_kv = I->n_heads_kv;
    const int  head_dim   = I->head_dim;
    const int  n_layers   = W->n_layers;
    const float rms_eps   = 1e-6f;
    const float attn_scale = 1.0f / sqrtf((float)head_dim);

    // -----------------------------------------------------------------------
    // 1. Collect last-token hidden states from captured layers → feature [n_tl * hidden]
    // -----------------------------------------------------------------------
    // Each captured[ti] has shape [n_embd * n_tokens] row-major (host floats).
    // We want the last token's hidden state from each.
    const int n_embd_cap = cap->n_embd > 0 ? cap->n_embd : hidden;
    std::vector<float> feature(static_cast<size_t>(n_tl) * hidden, 0.0f);

    if ((int)cap->captured.size() < n_tl) {
        std::fprintf(stderr, "[sp-oracle] step(): only %d captured layers, expected %d\n",
                     (int)cap->captured.size(), n_tl);
        return {};
    }
    for (int ti = 0; ti < n_tl; ++ti) {
        const auto& cv = cap->captured[ti];
        if (cv.empty()) {
            std::fprintf(stderr, "[sp-oracle] step(): captured[%d] is empty\n", ti);
            return {};
        }
        // Last token: cv[(n_tokens-1) * n_embd_cap .. n_tokens * n_embd_cap)
        const int n_tokens_cap = (int)(cv.size() / (size_t)n_embd_cap);
        const int last_offset  = (n_tokens_cap - 1) * n_embd_cap;
        const int copy_dim     = std::min(hidden, n_embd_cap);
        const float* src       = cv.data() + last_offset;
        float*       dst       = feature.data() + (size_t)ti * hidden;
        std::memcpy(dst, src, (size_t)copy_dim * sizeof(float));
    }

    // -----------------------------------------------------------------------
    // Build ggml compute graph
    // -----------------------------------------------------------------------
    ggml_init_params gip = {};
    gip.mem_size   = kCtxSize;
    gip.mem_buffer = I->ctx_mem.data();
    gip.no_alloc   = true;
    ggml_context* gctx = ggml_init(gip);
    if (!gctx) {
        std::fprintf(stderr, "[sp-oracle] step(): ggml_init failed\n");
        return {};
    }

    // -----------------------------------------------------------------------
    // 2+3. FC projection: feature [n_tl*hidden] -> [hidden]
    //      feature_tensor [n_tl*hidden, 1] × fc_weight [n_tl*hidden, hidden]^T = [hidden, 1]
    //      ggml_mul_mat(A, B) computes A^T × B, so:
    //      ggml_mul_mat(fc_weight, feature_2d) where fc_weight is [n_tl*hidden, hidden]
    //      and feature is [n_tl*hidden, 1] gives [hidden, 1].
    // -----------------------------------------------------------------------
    const int fc_in = n_tl * hidden;

    ggml_tensor* feat_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, fc_in, 1);
    ggml_set_name(feat_in, "feat_in");
    ggml_set_input(feat_in);

    // FC projection: [fc_in, hidden] weight, input [fc_in, 1] -> [hidden, 1]
    ggml_tensor* ctx_vec = ggml_mul_mat(gctx, W->fc_weight, feat_in);
    // RMSNorm + scale
    ctx_vec = ggml_mul(gctx, ggml_rms_norm(gctx, ctx_vec, rms_eps), W->hidden_norm);

    // -----------------------------------------------------------------------
    // 4. Embed mask tokens: ids [bs] -> [hidden, bs]
    // -----------------------------------------------------------------------
    ggml_tensor* mask_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, bs);
    ggml_set_name(mask_ids, "mask_ids");
    ggml_set_input(mask_ids);

    ggml_tensor* mask_emb = ggml_get_rows(gctx, I->target_tok_embd, mask_ids);
    // mask_emb shape: [hidden, bs]

    // -----------------------------------------------------------------------
    // 5. Build draft sequence: [hidden, seq_len] = concat(ctx_vec [hidden,1], mask_emb [hidden,bs])
    //    We build x as a fresh [hidden, seq_len] input tensor and fill it on the host.
    // -----------------------------------------------------------------------
    // We'll build the sequence as a plain input tensor [hidden, seq_len].
    ggml_tensor* x_input = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(x_input, "x_input");
    ggml_set_input(x_input);

    // x starts as a view of x_input so the transformer blocks operate on it.
    // We assign x_input data after allocation (it's built from ctx_vec + mask_emb below).
    ggml_tensor* x = x_input;

    // -----------------------------------------------------------------------
    // 6. 8 transformer blocks (bidirectional / non-causal, no mask, no RoPE)
    // -----------------------------------------------------------------------
    for (int il = 0; il < n_layers && il < (int)W->layers.size(); ++il) {
        const DFlashLayer& L = W->layers[il];

        // Pre-attention RMSNorm
        ggml_tensor* x_norm = ggml_mul(gctx, ggml_rms_norm(gctx, x, rms_eps), L.attn_norm);

        // QKV projections
        // Q: [hidden, seq_len] x attn_q [hidden, n_heads*head_dim] -> [n_heads*head_dim, seq_len]
        // ggml_mul_mat(attn_q, x_norm): attn_q^T × x_norm = [n_heads*head_dim, seq_len]
        ggml_tensor* Q = ggml_mul_mat(gctx, L.attn_q, x_norm);  // [n_heads*head_dim, seq_len]
        ggml_tensor* K = ggml_mul_mat(gctx, L.attn_k, x_norm);  // [n_heads_kv*head_dim, seq_len]
        ggml_tensor* V = ggml_mul_mat(gctx, L.attn_v, x_norm);  // [n_heads_kv*head_dim, seq_len]

        // Reshape: [head_dim, n_heads, seq_len]
        Q = ggml_reshape_3d(gctx, Q, head_dim, n_heads,    seq_len);
        K = ggml_reshape_3d(gctx, K, head_dim, n_heads_kv, seq_len);
        V = ggml_reshape_3d(gctx, V, head_dim, n_heads_kv, seq_len);

        // Per-head Q/K norm (QK-norm as in Qwen3)
        Q = ggml_mul(gctx, ggml_rms_norm(gctx, Q, rms_eps), L.attn_q_norm);
        K = ggml_mul(gctx, ggml_rms_norm(gctx, K, rms_eps), L.attn_k_norm);

        // GQA broadcast: repeat K and V from n_heads_kv to n_heads
        // ggml_repeat target shape [head_dim, n_heads, seq_len]
        ggml_tensor* K_rep = ggml_repeat(gctx, K, Q);
        ggml_tensor* V_rep = ggml_repeat(gctx, V, Q);

        // Attention scores: kq = Q^T × K per head
        // Q is [head_dim, n_heads, seq_len]; we need [seq_len, seq_len, n_heads] scores.
        // Use ggml's batched mul_mat: transpose Q and K along head_dim axis.
        // ggml_mul_mat(K_rep, Q): K_rep^T × Q per head batch
        //   K_rep: [head_dim, n_heads, seq_len] -> treated as [head_dim, seq_len] per head
        //   Q:     [head_dim, n_heads, seq_len]
        // Result: [seq_len, seq_len, n_heads]
        // Permute Q to [head_dim, seq_len, n_heads] for correct batched mul
        ggml_tensor* Q_perm = ggml_cont(gctx, ggml_permute(gctx, Q, 0, 2, 1, 3));
        // [head_dim, seq_len, n_heads]
        ggml_tensor* K_perm = ggml_cont(gctx, ggml_permute(gctx, K_rep, 0, 2, 1, 3));
        // [head_dim, seq_len, n_heads]

        // kq: [seq_len, seq_len, n_heads]
        ggml_tensor* kq = ggml_mul_mat(gctx, K_perm, Q_perm);
        // Apply softmax with scale, no mask (non-causal)
        ggml_tensor* kq_soft = ggml_soft_max_ext(gctx, kq, nullptr, attn_scale, 0.0f);

        // Weighted values: kqv = kq_soft × V
        // V_perm: [head_dim, seq_len, n_heads]
        ggml_tensor* V_perm = ggml_cont(gctx, ggml_permute(gctx, V_rep, 0, 2, 1, 3));
        // kqv = V_perm × kq_soft -> [head_dim, seq_len, n_heads]
        // ggml_mul_mat(V_perm, kq_soft): V_perm^T × kq_soft
        // V_perm is [head_dim, seq_len, n_heads], kq_soft is [seq_len, seq_len, n_heads]
        // Result: [head_dim, seq_len, n_heads]
        ggml_tensor* kqv = ggml_mul_mat(gctx, V_perm, kq_soft);

        // Reshape kqv: [head_dim, seq_len, n_heads] -> permute to [head_dim, n_heads, seq_len]
        // then flatten to [n_heads*head_dim, seq_len]
        ggml_tensor* kqv_perm = ggml_cont(gctx, ggml_permute(gctx, kqv, 0, 2, 1, 3));
        // [head_dim, n_heads, seq_len]
        ggml_tensor* kqv_flat = ggml_reshape_2d(gctx, kqv_perm, (int64_t)n_heads * head_dim, seq_len);

        // Output projection: [n_heads*head_dim, seq_len] -> [hidden, seq_len]
        ggml_tensor* attn_out = ggml_mul_mat(gctx, L.attn_o, kqv_flat);

        // Residual add
        x = ggml_add(gctx, x, attn_out);

        // Post-attention RMSNorm (pre-FFN)
        ggml_tensor* x_norm2 = ggml_mul(gctx, ggml_rms_norm(gctx, x, rms_eps), L.post_attn_norm);

        // SwiGLU FFN (no separate ffn_norm — ffn_norm may be null per arch spec)
        ggml_tensor* gate = ggml_silu(gctx, ggml_mul_mat(gctx, L.ffn_gate, x_norm2));
        ggml_tensor* up   = ggml_mul_mat(gctx, L.ffn_up, x_norm2);
        ggml_tensor* ffn  = ggml_mul_mat(gctx, L.ffn_down, ggml_mul(gctx, gate, up));

        // Residual add
        x = ggml_add(gctx, x, ffn);
    }

    // -----------------------------------------------------------------------
    // 7+8. Slice positions 1..bs, apply output_norm, project to vocab logits.
    //      Positions 1..bs (0-indexed in x [hidden, seq_len]).
    //      x is [hidden, seq_len]. We view rows 1..seq_len-1 = [hidden, bs].
    // -----------------------------------------------------------------------
    // ggml_view_2d: a [hidden, seq_len], extract [hidden, bs] starting at row 1
    size_t row_bytes = (size_t)hidden * sizeof(float);
    ggml_tensor* draft_x = ggml_view_2d(gctx, x, hidden, bs, row_bytes,
                                         /*offset=*/row_bytes);

    // Apply output_norm
    ggml_tensor* normed = ggml_mul(gctx, ggml_rms_norm(gctx, draft_x, rms_eps), W->output_norm);

    // Project to vocab: [hidden, bs] -> [vocab, bs]
    // ggml_mul_mat(target_output, normed): target_output^T × normed
    // target_output is [hidden, vocab] (ggml col-major) -> result [vocab, bs]
    ggml_tensor* logits = ggml_mul_mat(gctx, I->target_output, normed);
    ggml_set_name(logits, "draft_logits");
    ggml_set_output(logits);

    // -----------------------------------------------------------------------
    // Build and compute the graph
    // -----------------------------------------------------------------------
    // Use a large enough graph node count: 8 layers × ~20 ops + overhead
    const size_t graph_nodes = 512;
    ggml_cgraph* graph = ggml_new_graph_custom(gctx, graph_nodes, /*grads=*/false);
    // Also expand mask_emb so it gets allocated (even though we fill x_input manually)
    ggml_build_forward_expand(graph, mask_emb);
    ggml_build_forward_expand(graph, ctx_vec);
    ggml_build_forward_expand(graph, logits);

    if (!ggml_gallocr_alloc_graph(I->allocr, graph)) {
        std::fprintf(stderr, "[sp-oracle] step(): gallocr_alloc_graph failed\n");
        ggml_free(gctx);
        return {};
    }

    // -----------------------------------------------------------------------
    // Upload inputs
    // -----------------------------------------------------------------------
    // feat_in: raw feature vector [n_tl * hidden]
    ggml_backend_tensor_set(feat_in, feature.data(), 0, (size_t)fc_in * sizeof(float));

    // mask_ids: fill with mask_tok_id
    {
        std::vector<int32_t> mids(bs, I->mask_tok_id);
        ggml_backend_tensor_set(mask_ids, mids.data(), 0, (size_t)bs * sizeof(int32_t));
    }

    // Two-pass strategy: ctx_vec and mask_emb are graph nodes that don't
    // depend on x_input. We run the full graph once (pass A) so ggml
    // computes them and we can read them back. We then assemble x_input
    // on the host and re-upload it before pass B, which produces correct
    // logits. Buffers allocated by gallocr stay valid across both passes.

    // Pass A: compute ctx_vec and mask_emb (logits result discarded)
    if (ggml_backend_graph_compute(I->backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[sp-oracle] step(): mini-graph compute failed\n");
        ggml_free(gctx);
        return {};
    }

    // Read ctx_vec [hidden, 1] and mask_emb [hidden, bs]
    std::vector<float> ctx_vec_data(hidden);
    std::vector<float> mask_emb_data((size_t)hidden * bs);
    ggml_backend_tensor_get(ctx_vec, ctx_vec_data.data(), 0, (size_t)hidden * sizeof(float));
    ggml_backend_tensor_get(mask_emb, mask_emb_data.data(), 0,
                            (size_t)hidden * bs * sizeof(float));

    // Assemble x_input: row 0 = ctx_vec, rows 1..bs = mask_emb rows
    // x_input is [hidden, seq_len] stored column-major (ggml) = row-major [seq_len, hidden]
    // ggml stores ne[0]=hidden as the fast dimension, ne[1]=seq_len as slow.
    // So element [row r, col c] = data[r * hidden + c].
    std::vector<float> x_data((size_t)hidden * seq_len);
    std::memcpy(x_data.data(), ctx_vec_data.data(), (size_t)hidden * sizeof(float));
    std::memcpy(x_data.data() + hidden, mask_emb_data.data(),
                (size_t)hidden * bs * sizeof(float));
    ggml_backend_tensor_set(x_input, x_data.data(), 0, (size_t)hidden * seq_len * sizeof(float));

    // Step B: recompute the full graph (now x_input is populated)
    if (ggml_backend_graph_compute(I->backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[sp-oracle] step(): main graph compute failed\n");
        ggml_free(gctx);
        return {};
    }

    // -----------------------------------------------------------------------
    // 9. Read draft logits and greedy-sample draft tokens
    // -----------------------------------------------------------------------
    const int n_vocab = (int)I->target_output->ne[1];
    // Sanity: ne[1] must be vocab_size (>> hidden_size) — guards against a
    // transposed target_output tensor which would silently produce wrong logits.
    assert(n_vocab > I->hidden_size);
    std::vector<float> draft_logits((size_t)bs * n_vocab);
    ggml_backend_tensor_get(logits, draft_logits.data(), 0,
                            draft_logits.size() * sizeof(float));

    std::vector<int32_t> draft_tokens(bs);
    for (int i = 0; i < bs; ++i) {
        draft_tokens[i] = argmax_f32(draft_logits.data() + (size_t)i * n_vocab, n_vocab);
    }

    ggml_free(gctx);

    // -----------------------------------------------------------------------
    // 10. Verify with target model
    // -----------------------------------------------------------------------
    std::vector<float> target_logits = target_logits_fn(draft_tokens);
    if ((int)target_logits.size() != bs * n_vocab) {
        std::fprintf(stderr,
            "[sp-oracle] step(): target_logits_fn returned %zu elements, expected %d\n",
            target_logits.size(), bs * n_vocab);
        return {};
    }

    std::vector<int32_t> accepted = accept_exact(draft_tokens, target_logits, n_vocab);

    // -----------------------------------------------------------------------
    // 11. Adaptive EMA tracking (Phase 1: no Ricci sentinel, just EMA update)
    // -----------------------------------------------------------------------
    // accept_exact returns bs tokens on all-match, or i+1 tokens (last is
    // target's resampled choice) on mismatch at position i.
    const bool all_matched = ((int)accepted.size() == bs);
    const int accept_len   = all_matched ? bs : (int)accepted.size() - 1;
    const float ratio      = (float)accept_len / (float)bs;
    ++I->step_count;
    const float alpha = (I->step_count == 1) ? 1.0f : (2.0f / (I->adaptive_steps + 1));
    I->ema_accept = I->ema_accept * (1.0f - alpha) + ratio * alpha;

    // Phase 1: cur_block_size stays fixed at block_size
    // (adaptive steering via Ricci sentinel added in Phase 3)

    return accepted;
}

} // namespace sp::engine
