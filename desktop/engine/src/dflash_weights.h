// Shannon-Prime Engine — DFlash-draft-3.6 weight binding
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Loads and binds weights for the dflash-draft-3.6 GGUF architecture.
// The dflash-draft model is a compact 8-layer bidirectional draft model
// used for speculative decoding. It uses standard Qwen3 tensor naming
// for its transformer blocks plus three extra tensors:
//   dflash_fc.weight        — feature projection from target hidden states
//   dflash_hidden_norm.weight — norm applied to the FC output
//   output_norm.weight      — final norm before the logit projection
//
// This loader is self-contained and does not depend on LlamaWeights.

#pragma once
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace sp::engine {

struct DFlashLayer {
    struct ggml_tensor* attn_q        = nullptr;  // "blk.N.attn_q.weight"
    struct ggml_tensor* attn_k        = nullptr;  // "blk.N.attn_k.weight"
    struct ggml_tensor* attn_v        = nullptr;  // "blk.N.attn_v.weight"
    struct ggml_tensor* attn_o        = nullptr;  // "blk.N.attn_output.weight"
    struct ggml_tensor* attn_q_norm   = nullptr;  // "blk.N.attn_q_norm.weight"   [head_dim]
    struct ggml_tensor* attn_k_norm   = nullptr;  // "blk.N.attn_k_norm.weight"   [head_dim]
    struct ggml_tensor* attn_norm     = nullptr;  // "blk.N.attn_norm.weight"      pre-attention RMSNorm
    struct ggml_tensor* post_attn_norm = nullptr; // "blk.N.post_attention_norm.weight"
    struct ggml_tensor* ffn_norm      = nullptr;  // "blk.N.ffn_norm.weight"       pre-FFN RMSNorm (optional)
    struct ggml_tensor* ffn_gate      = nullptr;  // "blk.N.ffn_gate.weight"
    struct ggml_tensor* ffn_up        = nullptr;  // "blk.N.ffn_up.weight"
    struct ggml_tensor* ffn_down      = nullptr;  // "blk.N.ffn_down.weight"
};

struct DFlashWeights {
    // Draft model config (from GGUF metadata)
    int                  n_layers           = 0;
    int                  hidden_size        = 0;
    int                  n_heads            = 0;
    int                  n_heads_kv         = 0;
    int                  head_dim           = 128;
    float                rope_freq_base     = 1e7f;
    int                  block_size         = 8;   // our design default (model says 16)
    int                  mask_token_id      = 0;
    int                  n_target_features  = 0;   // n_target_layers * hidden_size
    std::vector<int>     target_layer_ids;          // [1, 10, 19, 28, 37]

    // Extra tensors
    struct ggml_tensor*  fc_weight          = nullptr;  // "dflash_fc.weight"          [10240, 2048]
    struct ggml_tensor*  hidden_norm        = nullptr;  // "dflash_hidden_norm.weight"  [2048]
    struct ggml_tensor*  output_norm        = nullptr;  // "output_norm.weight"         [2048]

    // Per-layer tensors (n_layers entries)
    std::vector<DFlashLayer> layers;

    // ggml context that owns both tensor descriptors and tensor data.
    struct ggml_context* ctx               = nullptr;
    void*                ctx_data          = nullptr;  // raw malloc'd block (ctx mem_buffer)
};

// Load dflash-draft-3.6 GGUF from path.
// Returns nullptr on failure (wrong arch, missing required tensors, I/O error).
// Caller owns the returned DFlashWeights and must call unload_dflash_weights()
// when done.
std::unique_ptr<DFlashWeights> load_dflash_weights(const std::string& path);

void unload_dflash_weights(DFlashWeights* w);

} // namespace sp::engine
