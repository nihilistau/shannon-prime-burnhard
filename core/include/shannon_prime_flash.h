// Shannon-Prime SP-Flash: VHT2 Hidden-State Compression for Speculative Decoding
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.
//
// SP-Flash compresses hidden states extracted from a target LLM's intermediate
// layers before passing them to a draft model (dflash-draft-3.6).  The pipeline
// is:
//   target layer activations  →  VHT2  →  Möbius reorder  →  band quantise  →  bytes
//   bytes  →  band dequantise  →  Möbius unreorder  →  VHT2 (self-inverse)  →  floats
//
// Default configuration for dflash-draft-3.6 (Qwen3.6-35B-A3B target):
//   hidden_size       = 2048
//   num_target_layers = 5  (layers [1, 10, 19, 28, 37])
//   band config       = 5/5/4/3 bits  (same as KV cache ship config)
//
// NOTE: sp_flash_compress_hidden / sp_flash_decompress_hidden are NOT
// thread-safe — both share the single ctx->scratch buffer.  Callers must
// serialise access or maintain one sp_flash_ctx_t per thread.

#pragma once
#include "shannon_prime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

typedef struct {
    sp_mobius_mask_t mobius;           // Möbius permutation for hidden_size
    sp_band_config_t k_bands;          // Band quantiser config (5/5/4/3 default)
    int              num_target_layers; // Number of target layers to compress
    int              hidden_size;       // Per-layer hidden dimension
    float           *scratch;            // [hidden_size] — VHT2/work buffer (NOT thread-safe)
    float           *scratch2;           // [hidden_size] — reorder scratch (NOT thread-safe)
} sp_flash_ctx_t;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Initialize context.  Allocates mobius.order/mu and scratch[hidden_size].
// n_bands / band_bits define the band quantiser (pass NULL band_bits for the
// 5/5/4/3 ship default).
// Returns 0 on success, -1 on allocation failure.
int  sp_flash_ctx_init(sp_flash_ctx_t *ctx,
                       int hidden_size,
                       int num_target_layers,
                       int n_bands,
                       const int *band_bits);

void sp_flash_ctx_free(sp_flash_ctx_t *ctx);

// ---------------------------------------------------------------------------
// Compress / decompress
// ---------------------------------------------------------------------------

// Compress one position's hidden states for all target layers.
//   hidden : [num_target_layers * hidden_size] float array, layer-major order
//   out    : output byte buffer of sp_flash_compressed_bytes(ctx) bytes
//
// NOT thread-safe — uses ctx->scratch.
void sp_flash_compress_hidden(sp_flash_ctx_t *ctx,
                              const float    *hidden,
                              uint8_t        *out);

// Decompress back to floats.
//   in  : byte buffer of sp_flash_compressed_bytes(ctx) bytes
//   out : [num_target_layers * hidden_size] float array, layer-major order
//
// NOT thread-safe — uses ctx->scratch.
void sp_flash_decompress_hidden(sp_flash_ctx_t *ctx,
                                const uint8_t  *in,
                                float          *out);

// ---------------------------------------------------------------------------
// Size queries
// ---------------------------------------------------------------------------

// Compressed bytes per layer (one hidden vector after band quantisation).
static inline int sp_flash_compressed_bytes_per_layer(const sp_flash_ctx_t *ctx)
{
    return ctx->k_bands.total_bytes;
}

// Total compressed bytes for one position (all layers combined).
static inline int sp_flash_compressed_bytes(const sp_flash_ctx_t *ctx)
{
    return ctx->k_bands.total_bytes * ctx->num_target_layers;
}

#ifdef __cplusplus
}
#endif
