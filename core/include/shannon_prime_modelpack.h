// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#ifndef SHANNON_PRIME_MODELPACK_H
#define SHANNON_PRIME_MODELPACK_H

#include "shannon_prime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Model Pack — architecture-aware default configuration
// ============================================================================
//
// A "model pack" is a small record that carries known-good compression
// defaults for a specific model architecture family. Presets let the
// bridges ship an honest "auto-adapts" claim: when a Qwen-3 MoE user
// runs the default path, they get a config tuned for Qwen-3 MoE, not
// the Llama-3 defaults.
//
// Layering (from weakest → strongest):
//   1. Shipping defaults  (5,5,4,3 K / 3 V) baked into sp_config_init.
//   2. Preset from this table matched by architecture name + hparams.
//   3. Environment overrides (SHANNON_PRIME_K_BITS, etc.).
//   4. Explicit CLI flags (--k-bits 4,4,4,3, etc.).
//
// Each preset is a *recommendation*, not a hard dependency. The caller
// is expected to log which preset was applied so reproducibility is
// preserved.
//
// Calibration status flag:
//   SP_PRESET_PROVISIONAL — config is a reasonable guess but has not been
//     validated against PPL/accuracy on this architecture. Treat as
//     stronger-than-default but weaker-than-explicit.
//   SP_PRESET_CALIBRATED  — numbers landed after a cache_ppl run against
//     a reference checkpoint with PPL drift within budget (documented per
//     preset in docs/MODEL-PACK.md).

typedef enum {
    SP_PRESET_PROVISIONAL = 0,
    SP_PRESET_CALIBRATED  = 1
} sp_preset_status_t;

typedef struct {
    // Identity
    const char *name;            // short human-readable (e.g. "llama-3")
    const char *arch_pattern;    // substring match against GGUF arch key
                                 // (e.g. "llama", "qwen2", "qwen3moe", "gemma3")
    const char *notes;           // one-line: what calibration proved / open risks

    // Dimensional gates — preset only applies when the model matches.
    // Zero means "don't care".
    int  head_dim_hint;          // expected head_dim (0 = any)
    int  min_n_layer;            // minimum n_layer for match (0 = any)
    int  min_n_head_kv;          // minimum n_head_kv for match (0 = any)

    // Compression defaults — laid out to overlay sp_config_t.
    int  k_n_bands;
    int  k_band_bits[SP_MAX_BANDS];
    int  v_n_bands;
    int  v_band_bits[SP_MAX_BANDS];
    int  residual_bits;          // sqfree residual bits (1..4)
    bool use_mobius;             // Möbius reorder on ship path
    bool recommend_sqfree;       // architecture benefits from sqfree over ship
    bool recommend_spinor;       // architecture benefits from SU(2) sheet

    // Hierarchical Kronecker sub-projection defaults.
    // Zero means "use global defaults" (skeleton_frac=0.09, residual_bits=2).
    float hier_skeleton_frac;    // skeleton fraction (0.0 = default 0.09)
    int   hier_residual_bits;    // hierarchical residual bits (0 = default 2)
    bool  recommend_hierarchical; // architecture benefits from hier over ship

    // Graph-size multiplier for ggml_new_graph_custom.
    // Used by the engine to scale graph capacity for deep or MoE models.
    // 0 means "use default (256)".
    int   graph_size_mult;

    // FP8 recommendation: architecture benefits from fp8 over int quantization
    // (typically: smooth V-cache distributions where fp8's dynamic range wins)
    bool  recommend_fp8;

    // Recommended draft model for speculative decoding.
    //
    // suggested_draft is a free-form hint string — typically a substring of
    // a GGUF filename or a human-friendly model designation that the user
    // should look for in their model directory. Examples:
    //   "qwen2.5-0.5b-instruct-q8_0.gguf"   exact filename hint
    //   "Qwen2.5-0.5B"                      family + size hint
    //   ""                                   empty = no recommendation
    //
    // Same family + tokeniser is the only hard requirement at the API
    // layer. This field exists so a future auto-select-draft helper can
    // suggest a draft to the user based on the resolved preset; the
    // current registry just stores the recommendation as documentation.
    //
    // suggested_draft_acceptance is a coarse expected-acceptance hint
    // (0.0..1.0) for the suggested draft on this target. 0.0 means
    // "no published number". Used purely for reporting / UX hints.
    const char *suggested_draft;
    float       suggested_draft_acceptance;

    // Status
    sp_preset_status_t status;
} sp_model_preset_t;

// Resolve the best preset for a given model.
//
// arch_name is the GGUF `general.architecture` value (or the llama.cpp
// enum name, which is equivalent for shipped architectures).
// head_dim / n_layers / n_heads_kv come from hparams.
//
// Returns a pointer into a static table (do not free). Returns NULL when
// no preset matches — caller should keep the shipping defaults.
const sp_model_preset_t *
sp_model_preset_resolve(const char *arch_name,
                        int head_dim,
                        int n_layers,
                        int n_heads_kv);

// Overlay preset tunables onto an sp_config_t. Dimensional fields
// (head_dim, n_layers, n_heads_kv) are left untouched — those must come
// from the model. Returns 0 on success, -1 if preset is NULL.
int sp_config_apply_preset(sp_config_t *cfg, const sp_model_preset_t *preset);

// Iterate the registered presets (diagnostic + docs generation).
// Returns entry at index i, or NULL when i is out of range.
const sp_model_preset_t *sp_model_preset_at(int index);

// Number of registered presets.
int sp_model_preset_count(void);

// Human-readable one-line summary: "name (arch=<pat>) K=5,5,4,3 V=3 res=3 [PROVISIONAL]"
// Writes into `out` (size bytes). Returns strlen of the result (or what
// would have been written, C-standard snprintf style).
int sp_model_preset_describe(const sp_model_preset_t *preset,
                             char *out, size_t size);

// Get the suggested draft model for a target preset. Returns the
// `suggested_draft` string (NULL or empty when the preset has no
// recommendation). `acceptance_out` is filled with the expected
// acceptance hint (0.0..1.0; 0.0 means "no published number"); pass
// NULL to ignore. Pure accessor — no allocation, returned pointer
// aliases preset's static storage.
const char *sp_model_preset_suggested_draft(const sp_model_preset_t *preset,
                                            float *acceptance_out);

#ifdef __cplusplus
}
#endif

#endif // SHANNON_PRIME_MODELPACK_H
