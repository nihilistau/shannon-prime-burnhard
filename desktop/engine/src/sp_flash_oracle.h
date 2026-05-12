// Shannon-Prime Engine — SP-Flash speculative decoding oracle
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Phase 1 draft oracle for SP-Flash speculative decoding.
// Uses dflash-draft-3.6 (8-layer bidirectional transformer) to draft
// block_size tokens per step, then applies exact-match acceptance against
// the target model's logits.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;

namespace sp::engine {

struct SpFlashCaptureCtx;

class SpFlashOracle {
public:
    struct Config {
        std::string draft_path;
        int   block_size      = 8;
        bool  adaptive        = false;
        float adaptive_hi     = 0.75f;
        float adaptive_lo     = 0.50f;
        int   adaptive_steps  = 50;
        // These tensors MUST come from the target LlamaWeights.
        // They must outlive the oracle.
        struct ggml_tensor* target_tok_embd = nullptr;  // [n_embd, vocab_size] (ggml col-major)
        struct ggml_tensor* target_output   = nullptr;  // lm_head [n_embd, vocab_size]
    };

    static std::unique_ptr<SpFlashOracle> load(const Config& cfg);
    ~SpFlashOracle();
    SpFlashOracle(const SpFlashOracle&) = delete;
    SpFlashOracle& operator=(const SpFlashOracle&) = delete;

    // Bind target's capture context (non-owning). Set active=true before each step.
    void bind_capture(SpFlashCaptureCtx* ctx);

    // Draft one block. Returns: accepted draft tokens + one resampled token.
    // target_logits_fn: given draft_tokens[block_size], returns target_logits[block_size * n_vocab]
    //   (flat row-major: token i's logits at [i*n_vocab .. (i+1)*n_vocab)).
    // last_token: the last accepted token before this block (unused in Phase 1 except for EMA bookkeeping).
    std::vector<int32_t> step(
        const std::function<std::vector<float>(const std::vector<int32_t>&)>& target_logits_fn,
        int32_t last_token);

    float acceptance_rate()    const;
    int   current_block_size() const;
    const std::vector<int>& target_layer_ids() const;

private:
    SpFlashOracle();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sp::engine
