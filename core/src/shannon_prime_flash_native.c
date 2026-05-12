// Shannon-Prime SP-Flash Native — Phase 3 Draft Architecture
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#include "shannon_prime_flash_native.h"
#include "shannon_prime.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// B. Möbius prime-noise schedule
// ============================================================================

void sp_flash_mobius_noise(
    float        *noise_emb,
    const float  *mask_emb,
    const float  *proposal_emb,
    const int8_t *mu,
    int           block_size,
    int           hidden_size)
{
    for (int i = 0; i < block_size; i++) {
        float       *dst  = noise_emb   + (size_t)i * hidden_size;
        const float *prop = proposal_emb + (size_t)i * hidden_size;
        if (mu[i] != 0) {
            // Squarefree position: use the proposal embedding (signal)
            memcpy(dst, prop, (size_t)hidden_size * sizeof(float));
        } else {
            // Prime-squared position: use the mask embedding (noise)
            memcpy(dst, mask_emb, (size_t)hidden_size * sizeof(float));
        }
    }
}

// ============================================================================
// C. Hierarchical Vilenkin 14-dim skeleton predictor
// ============================================================================

int sp_flash_vilenkin_pred_init(sp_flash_vilenkin_pred_t *pred,
                                int                       hidden_size,
                                const float              *W_fp32)
{
    if (!pred || hidden_size <= 0) return -1;

    pred->hidden_size = hidden_size;
    pred->n_skel      = SP_FLASH_N_SKEL;

    const size_t w_elems = (size_t)hidden_size * SP_FLASH_N_SKEL;

    pred->W = (float *)malloc(w_elems * sizeof(float));
    if (!pred->W) return -1;

    pred->vht2_buf = (float *)malloc((size_t)hidden_size * sizeof(float));
    if (!pred->vht2_buf) {
        free(pred->W);
        pred->W = NULL;
        return -1;
    }

    if (W_fp32) {
        memcpy(pred->W, W_fp32, w_elems * sizeof(float));
    } else {
        memset(pred->W, 0, w_elems * sizeof(float));
    }

    return 0;
}

void sp_flash_vilenkin_pred_free(sp_flash_vilenkin_pred_t *pred)
{
    if (!pred) return;
    free(pred->W);
    free(pred->vht2_buf);
    pred->W        = NULL;
    pred->vht2_buf = NULL;
}

void sp_flash_vilenkin_predict(
    const sp_flash_vilenkin_pred_t *pred,
    const float                    *target_hidden,
    float                          *out_hidden,
    float                          *skel_scratch)
{
    const int hs = pred->hidden_size;

    // Step 1: VHT2-transform target_hidden into pred->vht2_buf
    memcpy(pred->vht2_buf, target_hidden, (size_t)hs * sizeof(float));
    sp_vht2_forward_f32(pred->vht2_buf, hs);

    // Step 2: Extract the first SP_FLASH_N_SKEL (14) coefficients as skeleton
    memcpy(skel_scratch, pred->vht2_buf, SP_FLASH_N_SKEL * sizeof(float));

    // Step 3: Expand skeleton → hidden_size via W [hidden_size × n_skel]
    // out_hidden[i] = sum_j  W[i * n_skel + j] * skel_scratch[j]
    for (int i = 0; i < hs; i++) {
        const float *row = pred->W + (size_t)i * SP_FLASH_N_SKEL;
        float v = 0.0f;
        for (int j = 0; j < SP_FLASH_N_SKEL; j++) {
            v += row[j] * skel_scratch[j];
        }
        out_hidden[i] = v;
    }

    // Step 4: VHT2⁻¹ (= VHT2, self-inverse) to return to "spatial" domain
    sp_vht2_forward_f32(out_hidden, hs);
}

void sp_flash_vilenkin_context(
    const sp_flash_vilenkin_pred_t *preds,
    const float                    *target_hiddens,
    float                          *context_out,
    float                          *skel_scratch,
    float                          *pred_scratch,
    int                             num_target_layers,
    int                             hidden_size)
{
    memset(context_out, 0, (size_t)hidden_size * sizeof(float));

    for (int ti = 0; ti < num_target_layers; ti++) {
        const float *src = target_hiddens + (size_t)ti * hidden_size;

        sp_flash_vilenkin_predict(&preds[ti], src, pred_scratch, skel_scratch);

        for (int i = 0; i < hidden_size; i++) {
            context_out[i] += pred_scratch[i];
        }
    }

    if (num_target_layers > 1) {
        const float inv_n = 1.0f / (float)num_target_layers;
        for (int i = 0; i < hidden_size; i++) {
            context_out[i] *= inv_n;
        }
    }
}
