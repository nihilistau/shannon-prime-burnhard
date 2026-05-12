// Shannon-Prime SP-Flash Native — Phase 3 Draft Architecture
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.
//
// Phase 3 replaces the GGUF-based dflash-draft cross-context projection with
// two SP-native mechanisms:
//
//   B. Möbius prime-noise schedule:
//      Positions where μ(i+1) ≠ 0 (squarefree) carry signal from the draft
//      proposal embedding; positions where μ(i+1) = 0 (prime-squared) are
//      replaced by the mask embedding (pure noise). This focuses the draft
//      model's attention on semantically rich positions.
//
//   C. Hierarchical Vilenkin 14-dim skeleton predictor:
//      Projects target layer hidden states into the H_2 ⊗ H_7 (14-dim)
//      Kronecker Vilenkin basis, then expands via a calibrated weight matrix W
//      back to hidden_size. The predicted hidden state substitutes for the FC
//      projection, eliminating the dflash-draft GGUF dependency.

#pragma once
#include "shannon_prime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// B. Möbius prime-noise schedule
// ============================================================================
//
// Computes noise-blended embeddings for the draft sequence.
// For position i (0-indexed within the block):
//   mu_i = Möbius function of (i+1)
//   if |mu_i| == 1 (squarefree): noise_emb[i] = proposal_emb[i]  (signal)
//   if |mu_i| == 0 (prime-squared): noise_emb[i] = mask_emb       (noise)
//
// This mimics the Möbius randomisation in the SP denoising framework:
// squarefree positions are signal-bearing; prime-squared positions act as
// masked "noise" tokens that the draft model must predict through.

// In-place computation: writes noise_emb[block_size * hidden_size].
// mask_emb        : [hidden_size]                    — mask token embedding
// proposal_emb    : [block_size * hidden_size]       — initial proposal embeds
// mu              : pre-computed Möbius values μ(1..block_size) (int8_t)
//                   mu[i] = μ(i+1) for i in [0, block_size)
void sp_flash_mobius_noise(
    float        *noise_emb,          // [block_size * hidden_size] out
    const float  *mask_emb,           // [hidden_size]
    const float  *proposal_emb,       // [block_size * hidden_size]
    const int8_t *mu,                 // [block_size] — μ(i+1)
    int           block_size,
    int           hidden_size);

// ============================================================================
// C. Hierarchical Vilenkin 14-dim skeleton predictor
// ============================================================================
//
// n_skel = 14 = 2 × 7: the H_2 ⊗ H_7 Kronecker Vilenkin level.
// The first 14 VHT2 coefficients of a 2048-dim vector capture the low-frequency
// structure under the p-adic Vilenkin hierarchy up to prime p=7.
//
// Pipeline for one target layer:
//   target_hidden [hidden_size] →
//   VHT2 →
//   skel [14] (first 14 coefficients) →
//   W [hidden_size × 14] → out_hidden [hidden_size] →
//   VHT2⁻¹ (= VHT2, self-inverse)

#define SP_FLASH_N_SKEL  14   // H_2 ⊗ H_7 Kronecker skeleton dimension

typedef struct {
    float *W;           // [hidden_size × n_skel] row-major, fp32 (calibrated)
    float *vht2_buf;    // [hidden_size] — internal scratch for VHT2 on input
    int    n_skel;      // = SP_FLASH_N_SKEL (always 14)
    int    hidden_size;
} sp_flash_vilenkin_pred_t;

// Initialise predictor for one target layer.
// W_fp32: [hidden_size × n_skel] row-major weight matrix (may be NULL to zero-init).
// Allocates W and vht2_buf internally; free with sp_flash_vilenkin_pred_free().
// Returns 0 on success, -1 on allocation failure.
int sp_flash_vilenkin_pred_init(sp_flash_vilenkin_pred_t *pred,
                                int                       hidden_size,
                                const float              *W_fp32);

void sp_flash_vilenkin_pred_free(sp_flash_vilenkin_pred_t *pred);

// Predict hidden_size context from one target layer's hidden state.
// target_hidden → 14-dim skeleton → W expansion → out_hidden
//
// skel_scratch : caller-provided [SP_FLASH_N_SKEL] float scratch buffer.
// out_hidden   : [hidden_size] output.
void sp_flash_vilenkin_predict(
    const sp_flash_vilenkin_pred_t *pred,
    const float                    *target_hidden,
    float                          *out_hidden,
    float                          *skel_scratch);   // [SP_FLASH_N_SKEL]

// Aggregate predictions from all target layers into a single context vector.
// Computes sp_flash_vilenkin_predict for each layer and averages them.
//
// preds           : [num_target_layers] array of initialised predictors.
// target_hiddens  : [num_target_layers * hidden_size] concatenated hidden states.
// context_out     : [hidden_size] averaged output.
// skel_scratch    : [SP_FLASH_N_SKEL] caller temp.
// pred_scratch    : [hidden_size] caller temp for per-layer output before avg.
void sp_flash_vilenkin_context(
    const sp_flash_vilenkin_pred_t *preds,
    const float                    *target_hiddens,
    float                          *context_out,
    float                          *skel_scratch,
    float                          *pred_scratch,
    int                             num_target_layers,
    int                             hidden_size);

#ifdef __cplusplus
}
#endif
