// Shannon-Prime Engine — SP-Flash Phase 3 native draft oracle
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Phase 3 replaces the dflash-draft GGUF cross-context FC projection with
// two SP-native mechanisms (no GGUF draft model required):
//
//   B. Möbius prime-noise schedule:
//      Squarefree positions (μ(i+1) ≠ 0) use the Vilenkin context vector as
//      their embedding; prime-squared positions (μ(i+1) = 0) use the mask
//      token embedding. This injects structured noise at arithmetically
//      constrained positions while preserving signal elsewhere.
//
//   C. Hierarchical Vilenkin 14-dim skeleton predictor:
//      Target-layer hidden states are projected to the H_2 ⊗ H_7 (14-dim)
//      Kronecker Vilenkin basis via VHT2, averaged across layers, then
//      expanded back to hidden_size via a calibrated weight matrix W. The
//      resulting context vector drives the draft logit projection directly
//      via the target model's lm_head, bypassing the draft transformer.
//
// Calibration: run scripts/calibrate_sp_flash.py once to produce
//   calibration/layer_N.bin  (float32 [hidden_size × 14], one per target layer)
//   calibration/meta.json    (hidden_size, num_target_layers, target_layer_ids)

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;

namespace sp::engine {

struct SpFlashCaptureCtx;

class SpFlashOracleNative {
public:
    struct Config {
        std::string calibration_dir;           // dir containing layer_N.bin + meta.json
        int   block_size     = 8;
        bool  adaptive       = false;
        float adaptive_hi    = 0.75f;
        float adaptive_lo    = 0.50f;
        int   adaptive_steps = 50;
        int   mask_tok_id    = 0;
        // Non-owning target model tensors — must outlive the oracle.
        struct ggml_tensor* target_tok_embd = nullptr;  // [n_embd, vocab_size]
        struct ggml_tensor* target_output   = nullptr;  // [n_embd, vocab_size] lm_head
    };

    static std::unique_ptr<SpFlashOracleNative> load(const Config& cfg);
    ~SpFlashOracleNative();
    SpFlashOracleNative(const SpFlashOracleNative&) = delete;
    SpFlashOracleNative& operator=(const SpFlashOracleNative&) = delete;

    // Bind the engine's capture context (non-owning). The engine sets active=true
    // before calling step() and false afterwards.
    void bind_capture(SpFlashCaptureCtx* ctx);

    // Draft one block via Vilenkin context + Möbius noise schedule, then
    // verify against the target model. Returns accepted tokens (1..block_size);
    // the caller should hold back the last entry as the next cur_tok.
    //
    // target_logits_fn: given draft_tokens[block_size], returns
    //   target_logits[block_size * n_vocab] (row-major).
    std::vector<int32_t> step(
        const std::function<std::vector<float>(const std::vector<int32_t>&)>& target_logits_fn,
        int32_t last_token);

    float acceptance_rate()    const;
    int   current_block_size() const;
    // Target layer IDs read from meta.json; used to configure SpFlashCaptureCtx.
    const std::vector<int>& target_layer_ids() const;

private:
    SpFlashOracleNative();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sp::engine
