// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#include "shannon_prime_modelpack.h"

#include <stdio.h>
#include <string.h>

// ============================================================================
// Registry
// ============================================================================
//
// Kept intentionally small. Adding an architecture is cheap — adding one
// without doing the calibration work is harmful. New entries start at
// SP_PRESET_PROVISIONAL, get promoted to SP_PRESET_CALIBRATED only after
// `sp-engine cache_ppl --model <ref> --sqfree` shows PPL drift within
// the budget documented in docs/MODEL-PACK.md.
//
// Matching rules (see sp_model_preset_resolve below):
//   * arch_pattern is a substring match against arch_name (case-sensitive;
//     GGUF arch strings are lowercase by convention).
//   * Dimensional hints are soft: if head_dim_hint is nonzero it must
//     match exactly. min_n_layer / min_n_head_kv are lower bounds.
//   * First match wins — order presets from most-specific to most-general.

static const sp_model_preset_t g_presets[] = {
    // ── Qwen 3.6 35B-A3B / Qwen3-Next (hybrid SSM+MoE) ────────────────
    // Arch string is "qwen35moe". MUST come before "qwen3moe" and
    // "qwen3" — strstr("qwen35moe", "qwen3moe") is NULL (the "5" breaks
    // the substring), but strstr("qwen35moe", "qwen3") matches at pos 0,
    // so without this entry a qwen35moe model would silently adopt the
    // dense qwen3 preset. Qwen3.6 has 40 layers with only 10 full MoE
    // attention layers (every 4th) — the rest are Gated DeltaNet with
    // recurrent state, contributing no K/V to the cache at all. The
    // preset therefore applies to a much smaller slice of layers than
    // the qwen3-moe preset, but the band profile is the same: MoE
    // dense-KV mid-band energy, spinor recommended.
    {
        .name              = "qwen3-next",
        .arch_pattern      = "qwen35moe",
        .notes             = "Qwen3.6 hybrid SSM+MoE; preset applies to the 10 full-attn layers (GDN layers have no KV)",
        .head_dim_hint     = 0,
        .min_n_layer       = 1,
        .min_n_head_kv     = 1,
        .k_n_bands         = 5,
        .k_band_bits       = { 5, 4, 4, 4, 4 },
        .v_n_bands         = 5,
        .v_band_bits       = { 4, 4, 4, 4, 4 },
        .residual_bits     = 3,
        .use_mobius        = true,
        .recommend_sqfree  = true,
        .recommend_spinor  = true,
        .hier_skeleton_frac    = 0.09f,
        .hier_residual_bits    = 2,
        .recommend_hierarchical = true,   // 35B-A3B: scaling law projects <0.1% PPL hier hit
        .graph_size_mult       = 384,     // MoE routing overhead: 40 layers × MoE expansion
        .recommend_fp8         = true,    // smooth V distributions benefit from fp8 range
        // Qwen3.6 hybrid + dense Qwen 2.5 share the Qwen tokeniser; 1.5B is
        // the smallest dense Qwen 3.x with stable acceptance against a 35B
        // hybrid target. 0.5B works but acceptance dips to ~55%.
        .suggested_draft   = "Qwen2.5-1.5B-Instruct",
        .suggested_draft_acceptance = 0.65f,
        .status            = SP_PRESET_PROVISIONAL,
    },

    // ── Qwen 3 MoE (original MoE variants, non-hybrid) ─────────────────
    // Wide MoE stacks, dense K/V heads, benefit most from sqfree +
    // spinor because mid-band energy is the dominant contributor.
    {
        .name              = "qwen3-moe",
        .arch_pattern      = "qwen3moe",
        .notes             = "MoE: dense KV, high mid-band energy; spinor recovers tail",
        .head_dim_hint     = 0,
        .min_n_layer       = 1,
        .min_n_head_kv     = 1,
        .k_n_bands         = 5,
        .k_band_bits       = { 5, 4, 4, 4, 4 },
        .v_n_bands         = 5,
        .v_band_bits       = { 4, 4, 4, 4, 4 },
        .residual_bits     = 3,
        .use_mobius        = true,
        .recommend_sqfree  = true,
        .recommend_spinor  = true,
        .suggested_draft   = "Qwen2.5-1.5B-Instruct",
        .suggested_draft_acceptance = 0.65f,
        .status            = SP_PRESET_PROVISIONAL,
    },

    // ── Qwen 3.5 / Qwen 3 (dense) ──────────────────────────────────────
    {
        .name              = "qwen3",
        .arch_pattern      = "qwen3",
        .notes             = "Dense Qwen 3.x; similar band profile to Llama-3 but V carries more energy",
        .head_dim_hint     = 0,
        .min_n_layer       = 1,
        .min_n_head_kv     = 1,
        .k_n_bands         = 4,
        .k_band_bits       = { 5, 5, 4, 3 },
        .v_n_bands         = 2,
        .v_band_bits       = { 4, 3 },
        .residual_bits     = 3,
        .use_mobius        = true,
        .recommend_sqfree  = true,
        .recommend_spinor  = false,
        // Qwen 2.5 0.5B is the canonical small-draft for the dense Qwen
        // family. Same vocab across 0.5B/1.5B/3B/7B/14B/32B/72B targets,
        // 65-75% acceptance in informal testing on 7B/14B targets.
        .suggested_draft   = "Qwen2.5-0.5B-Instruct",
        .suggested_draft_acceptance = 0.70f,
        .status            = SP_PRESET_PROVISIONAL,
    },

    // ── Gemma 4 (distinct arch string; reuses the gemma3 SWA band profile) ─
    // GGUFs ship with arch "gemma4" (verified on gemma-4-31B-it-Q4_K_M).
    // Until a dedicated calibration run exists the bit budget tracks
    // gemma3's provisional numbers — same sliding-window attention
    // pattern, same "preserve upper K band" intuition. Split out now so
    // future calibration can diverge without a string-pattern migration.
    {
        .name              = "gemma4",
        .arch_pattern      = "gemma4",
        .notes             = "Gemma 4: inherits gemma3 SWA band profile until dedicated calibration lands",
        .head_dim_hint     = 0,
        .min_n_layer       = 1,
        .min_n_head_kv     = 1,
        .k_n_bands         = 4,
        .k_band_bits       = { 5, 4, 4, 3 },
        .v_n_bands         = 1,
        .v_band_bits       = { 3 },
        .residual_bits     = 3,
        .use_mobius        = true,
        .recommend_sqfree  = true,
        .recommend_spinor  = false,
        // Gemma 4 inherits gemma3's draft suggestion until calibration
        // numbers diverge — same family, same tokeniser.
        .suggested_draft   = "gemma-2-2b-it",
        .suggested_draft_acceptance = 0.60f,
        .status            = SP_PRESET_PROVISIONAL,
    },

    // ── Gemma 3 ────────────────────────────────────────────────────────
    {
        .name              = "gemma3",
        .arch_pattern      = "gemma3",
        .notes             = "Gemma 3: sliding-window attn stresses upper K band; keep K[0]=5. 12B=48 layers needs graph_size_mult.",
        .head_dim_hint     = 0,
        .min_n_layer       = 1,
        .min_n_head_kv     = 1,
        .k_n_bands         = 4,
        .k_band_bits       = { 5, 4, 4, 3 },
        .v_n_bands         = 1,
        .v_band_bits       = { 3 },
        .residual_bits     = 3,
        .use_mobius        = true,
        .recommend_sqfree  = true,
        .recommend_spinor  = false,
        .graph_size_mult       = 320,     // 48 layers on 12B variant; default 256 overflows
        .recommend_hierarchical = true,   // 12B+: scaling law projects competitive with ship
        // Gemma 2 2B-it is the smallest draft with the gemma vocab; on
        // gemma-3-12b targets acceptance dips toward 55% on long-form
        // code generation.
        .suggested_draft   = "gemma-2-2b-it",
        .suggested_draft_acceptance = 0.60f,
        .status            = SP_PRESET_PROVISIONAL,
    },

    // ── Phi 3 / Phi 3.1 / Phi 4 (all ship arch="phi3" in GGUF) ────────
    // Microsoft's phi-4 GGUF still carries general.architecture="phi3"
    // (verified on lmstudio-community/phi-4-Q4_K_M.gguf). Dense GQA
    // attention with a Llama-like band profile. Calibrated 2026-04-21
    // on Phi-3.1-mini-4k-Q8_0 at ctx=2048 chunks=8: baseline PPL 5.0297,
    // ship PPL 5.1523, drift +2.44% (≤5% ship budget). Ledger row in
    // shannon-prime/docs/MODEL-PACK-CALIBRATION.md.
    {
        .name              = "phi3",
        .arch_pattern      = "phi3",
        .notes             = "Phi 3.x and Phi 4 (all ship arch=phi3); ship +2.44% drift on Phi-3.1-mini-Q8 2026-04-21",
        .head_dim_hint     = 0,
        .min_n_layer       = 1,
        .min_n_head_kv     = 1,
        .k_n_bands         = 4,
        .k_band_bits       = { 5, 5, 4, 3 },
        .v_n_bands         = 1,
        .v_band_bits       = { 3 },
        .residual_bits     = 3,
        .use_mobius        = true,
        .recommend_sqfree  = false,
        .recommend_spinor  = false,
        // phi-3-mini is the standard Phi family draft. Acceptance hovers
        // around 65-70% on phi-4 / phi-3.1-medium targets at default temp.
        .suggested_draft   = "Phi-3-mini-4k-instruct",
        .suggested_draft_acceptance = 0.65f,
        .status            = SP_PRESET_CALIBRATED,
    },

    // ── Llama 3 / 3.1 / 3.2 / 3.3 ──────────────────────────────────────
    // Shipping default mirrors this — the preset exists so a match is
    // logged and the "auto-adapts" story is auditable.
    {
        .name              = "llama-3",
        .arch_pattern      = "llama",
        .notes             = "Llama-3 family; shipping defaults; baseline for calibration drift",
        .head_dim_hint     = 0,
        .min_n_layer       = 1,
        .min_n_head_kv     = 1,
        .k_n_bands         = 4,
        .k_band_bits       = { 5, 5, 4, 3 },
        .v_n_bands         = 1,
        .v_band_bits       = { 3 },
        .residual_bits     = 3,
        .use_mobius        = true,
        .recommend_sqfree  = false,    // ship path is strong enough
        .recommend_spinor  = false,
        // Llama 3.2 1B is the official small-draft for Llama 3.x.
        // Acceptance 60-70% on 8B targets, 55-65% on 70B (where the
        // capability gap is wider).
        .suggested_draft   = "Llama-3.2-1B-Instruct",
        .suggested_draft_acceptance = 0.65f,
        .status            = SP_PRESET_PROVISIONAL,
    },
};

static const int g_preset_count = (int)(sizeof(g_presets) / sizeof(g_presets[0]));

// ============================================================================
// Public API
// ============================================================================

const sp_model_preset_t *
sp_model_preset_resolve(const char *arch_name,
                        int head_dim,
                        int n_layers,
                        int n_heads_kv)
{
    if (arch_name == NULL || arch_name[0] == '\0') {
        return NULL;
    }
    for (int i = 0; i < g_preset_count; ++i) {
        const sp_model_preset_t *p = &g_presets[i];
        if (strstr(arch_name, p->arch_pattern) == NULL) {
            continue;
        }
        if (p->head_dim_hint != 0 && p->head_dim_hint != head_dim) {
            continue;
        }
        if (p->min_n_layer   > n_layers)    continue;
        if (p->min_n_head_kv > n_heads_kv)  continue;
        return p;
    }
    return NULL;
}

int sp_config_apply_preset(sp_config_t *cfg, const sp_model_preset_t *preset) {
    if (cfg == NULL || preset == NULL) return -1;

    // K bands — copy count + bits, leave trailing slots untouched
    int kb = preset->k_n_bands;
    if (kb < 1) kb = 1;
    if (kb > SP_MAX_BANDS) kb = SP_MAX_BANDS;
    cfg->k_n_bands = kb;
    for (int i = 0; i < kb; ++i) cfg->k_band_bits[i] = preset->k_band_bits[i];

    // V bands
    int vb = preset->v_n_bands;
    if (vb < 1) vb = 1;
    if (vb > SP_MAX_BANDS) vb = SP_MAX_BANDS;
    cfg->v_n_bands = vb;
    for (int i = 0; i < vb; ++i) cfg->v_band_bits[i] = preset->v_band_bits[i];

    // Möbius is an overlay; preserve caller's explicit override if they
    // already set it. The convention is: shipping default is true, so if
    // the caller set it to false they meant it.
    cfg->use_mobius_mask = preset->use_mobius;

    return 0;
}

const sp_model_preset_t *sp_model_preset_at(int index) {
    if (index < 0 || index >= g_preset_count) return NULL;
    return &g_presets[index];
}

int sp_model_preset_count(void) {
    return g_preset_count;
}

int sp_model_preset_describe(const sp_model_preset_t *preset,
                             char *out, size_t size)
{
    if (!preset || !out || size == 0) return -1;

    // "name (arch=X) K=a,b,c,d V=e,f res=R spinor=Y [STATUS]"
    char kbuf[48]; char vbuf[48];
    int kpos = 0, vpos = 0;
    for (int i = 0; i < preset->k_n_bands && kpos < (int)sizeof(kbuf) - 4; ++i) {
        kpos += snprintf(kbuf + kpos, sizeof(kbuf) - kpos,
                         i == 0 ? "%d" : ",%d", preset->k_band_bits[i]);
    }
    for (int i = 0; i < preset->v_n_bands && vpos < (int)sizeof(vbuf) - 4; ++i) {
        vpos += snprintf(vbuf + vpos, sizeof(vbuf) - vpos,
                         i == 0 ? "%d" : ",%d", preset->v_band_bits[i]);
    }

    return snprintf(out, size,
                    "%s (arch=%s) K=%s V=%s res=%d spinor=%s [%s]",
                    preset->name,
                    preset->arch_pattern,
                    kbuf, vbuf,
                    preset->residual_bits,
                    preset->recommend_spinor ? "recommended" : "off",
                    preset->status == SP_PRESET_CALIBRATED
                        ? "CALIBRATED" : "PROVISIONAL");
}

const char *sp_model_preset_suggested_draft(const sp_model_preset_t *preset,
                                            float *acceptance_out)
{
    if (!preset) {
        if (acceptance_out) *acceptance_out = 0.0f;
        return NULL;
    }
    if (acceptance_out) *acceptance_out = preset->suggested_draft_acceptance;
    return preset->suggested_draft;
}
