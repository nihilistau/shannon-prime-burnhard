// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#ifndef SHANNON_PRIME_CORE_H
#define SHANNON_PRIME_CORE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// FP16 Utilities
// ============================================================================

static inline float sp_f16_to_f32(uint16_t h) {
    // IEEE 754 half-precision to single-precision
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // Denormalized
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13); // Inf/NaN
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

static inline uint16_t sp_f32_to_f16(float val) {
    uint32_t f;
    memcpy(&f, &val, sizeof(uint32_t));
    uint16_t sign = (f >> 16) & 0x8000;
    int exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) {
        return sign; // Flush to zero
    } else if (exp >= 31) {
        return sign | 0x7C00; // Inf
    }
    return sign | (exp << 10) | (mant >> 13);
}

// ============================================================================
// Constants
// ============================================================================

// Maximum supported head dimensions (typically power of 2; VHT2 also accepts
// any dim that factors into {2,3,5,7,11}, e.g. sqfree-padded 66/154/330)
#define SP_MAX_HEAD_DIM     256
#define SP_MAX_BANDS        16
#define SP_MAX_LAYERS       128
#define SP_MAX_HEADS        128

// Vilenkin basis: product of first k primes
// N=210 = 2*3*5*7 covers hd <= 128 with zero-padding
#define SP_VILENKIN_N_2P    6    // 2*3 = 6
#define SP_VILENKIN_N_3P    30   // 2*3*5 = 30
#define SP_VILENKIN_N_4P    210  // 2*3*5*7 = 210

// Ship-safe bit allocation (NaN-safe on all platforms)
#define SP_DEFAULT_K_BANDS  4
#define SP_DEFAULT_K_BITS   { 5, 5, 4, 3 }
#define SP_DEFAULT_V_BANDS  1
#define SP_DEFAULT_V_BITS   { 3 }

// ============================================================================
// Configuration
// ============================================================================

typedef struct {
    int      head_dim;           // Model head dimension (64, 128, 256)
    int      n_layers;           // Number of transformer layers
    int      n_heads_kv;         // Number of KV heads (GQA-aware)

    // VHT2 banded quantization
    int      k_n_bands;          // Number of K spectral bands (default 4)
    int      k_band_bits[SP_MAX_BANDS]; // Bits per K band (default 5,5,4,3)
    int      v_n_bands;          // Number of V bands (default 1)
    int      v_band_bits[SP_MAX_BANDS]; // Bits per V band (default 3)

    // Ternary noise-tail mask. Bit b set ⇒ band b is ternary {-1,0,+1}
    // at 2 bpp regardless of the corresponding *_band_bits[b] value. The
    // strange-attractor stack 5/5/4/1.58 preset sets bit 3 in
    // k_ternary_mask. Default 0 (no ternary bands) preserves all
    // existing callers' behaviour.
    uint32_t k_ternary_mask;
    uint32_t v_ternary_mask;

    // FP8 (E4M3FN) banded quantization toggle. When true, backends with
    // an fp8 path (the engine's CUDA backend via shannon_prime_fp8.cu,
    // gated on SP_FP8 / SP_ENGINE_FP8 compile flags) substitute fp8
    // quantisation for the int8 path on bands where it helps —
    // primarily V cache, where the smooth distribution favours fp8's
    // higher dynamic range.
    //
    // Backends without an fp8 path (CPU shadow cache, Adreno, current
    // Vulkan) are expected to log a warning and fall back to int.
    // Treated as advisory at the config layer; downstream backends
    // decide what to do with it. Default false preserves the int path.
    bool     use_fp8;

    // Möbius partition mask
    bool     use_mobius_mask;    // Reorder coefficients squarefree-first
    int      skeleton_k;         // VHT2 skeleton size for K (must == head_dim)
    int      skeleton_v;         // VHT2 skeleton size for V (must == head_dim)

    // Vilenkin successive decomposition (research path)
    bool     use_vilenkin;       // Legacy flag — VHT2 is always Vilenkin-Hartley
    int      vilenkin_primes;    // Number of primes (2=6, 3=30, 4=210)
    float    energy_threshold;   // Energy retention (0.95 = 95%)
} sp_config_t;

// Initialize config with ship-safe defaults
void sp_config_init(sp_config_t *cfg, int head_dim, int n_layers, int n_heads_kv);

// ============================================================================
// VHT2 (Vilenkin-Hartley Transform) — the single transform
// ============================================================================
//
// VHT2 is an orthonormal staged Hartley transform: for a length n that factors
// into small primes {2,3,5,7,11} it applies one p × p Hartley stage per prime
// factor, each normalised by 1/√p. The transform is self-inverse
//   VHT2(VHT2(x)) = x           (within float tolerance, no 1/N needed)
// so the same function serves as forward and inverse.
//
// At n = 2^k the stages collapse to the p=2 Hartley butterfly scaled by 1/√2
// per stage — unit-norm basis, self-inverse, no 1/N division on the inverse.
// Band quantisation and Möbius reordering act on coefficients already in a
// unit-norm basis.

// In-place VHT2 on float vector of length n. n must factor into {2,3,5,7,11};
// non-supported dimensions should be handled by sqfree_pad first.
// Self-inverse: call twice to recover the original vector.
void sp_vht2_forward_f32(float *data, int n);

// In-place VHT2 on fp16 vector (for backends with native fp16).
// Reference implementation; backends may override with SIMD.
void sp_vht2_forward_f16(uint16_t *data, int n);

// ============================================================================
// Möbius mask — squarefree-first coefficient ordering
// ============================================================================
//
// μ(n) != 0 iff n is squarefree (no repeated prime factors).
// 61.4% of indices in N=210 are squarefree.
// Prioritizing squarefree indices during coefficient extraction
// improves quality by 0.14 PPL at identical coefficient budget.
//
// The mask is a property of the VHT2 spectrum, not the model.
// Cross-platform invariant: K correlation 0.997 on both hd=128 and hd=64.

typedef struct {
    int      n;                  // Dimension (head_dim)
    int     *order;              // Permutation: order[i] = original index
    int      n_squarefree;       // Count of squarefree indices
    int8_t  *mu;                 // Möbius function values μ(0..n-1)
} sp_mobius_mask_t;

// Build the Möbius mask for dimension n.
// Caller must free with sp_mobius_mask_free().
int sp_mobius_mask_init(sp_mobius_mask_t *mask, int n);
void sp_mobius_mask_free(sp_mobius_mask_t *mask);

// Apply Möbius reordering to VHT2 coefficients (in-place).
// Squarefree coefficients move to front; non-squarefree to back.
void sp_mobius_reorder(float *vht2_coeffs, const sp_mobius_mask_t *mask);

// Inverse reorder (restore original index order after dequantization).
void sp_mobius_unreorder(float *vht2_coeffs, const sp_mobius_mask_t *mask);

// Caller-owned-scratch variants. `scratch` must be at least mask->n floats
// and is writable. Used on the hot path to avoid malloc per KV vector.
void sp_mobius_reorder_ex(float *vht2_coeffs, const sp_mobius_mask_t *mask,
                          float *scratch);
void sp_mobius_unreorder_ex(float *vht2_coeffs, const sp_mobius_mask_t *mask,
                            float *scratch);

// ============================================================================
// VHT2 banded quantization
// ============================================================================
//
// After VHT2 + optional Möbius reorder:
//   1. Split N coefficients into k equal bands
//   2. Each band gets its own fp16 scale + packed integer values
//   3. Band allocation mirrors VHT2 energy decay:
//      high-energy bands get more bits, low-energy tail gets fewer
//
// Key findings from paper:
//   - 5/5/4/3 BEATS lossless fp16 by 0.04% (spectral regularization)
//   - 4/4/4/4 is off the Pareto frontier (4/4/4/3 is strictly better)
//   - 3-bit floor: 2-bit on any band is catastrophic
//   - Flat beats banded for V vectors (no exceptions)

typedef struct {
    int      n_bands;
    int      band_bits[SP_MAX_BANDS];
    int      head_dim;           // total coefficients per vector
    int      band_size;          // typical coefficients per band (= head_dim / n_bands)
    int      total_bytes;        // compressed size per vector
    // When head_dim is not evenly divisible by n_bands, the last band absorbs
    // the remainder: bands 0..n_bands-2 have `band_size` coefficients, band
    // n_bands-1 has `head_dim - band_size * (n_bands - 1)` coefficients.
    // Callers iterate through sp_band_span(bc, b, &off, &sz) rather than
    // computing `b * band_size` directly.

    // Ternary noise-tail mask. Bit b set ⇒ band b is quantised to {-1, 0, +1}
    // (≈1.58 bits/coefficient information content) regardless of band_bits[b].
    // The strange-attractor stack predicts the noise tail (typically band 3 in
    // a 5/5/4/3 ship configuration) is statistically indistinguishable from
    // ternary, so it can be stored at ~1.58 bpp without measurable quality
    // loss. This C-level implementation packs at 2 bpp (4 trits/byte) for
    // simplicity — the tighter 5-trits-per-byte packing (true 1.58 bpp via
    // base-3 encoding) is a future optimisation that doesn't change the API.
    //
    // Ternary bands ignore band_bits[b] for both encoding (scale = amax,
    // deadband at 0.5*amax) and decoding. They share the fp16 scale header
    // with regular bands. SP_MAX_BANDS is bounded well under 32 so a
    // uint32_t mask is sufficient.
    //
    // Set via sp_band_config_init_ext(); sp_band_config_init() leaves the
    // mask at zero, preserving existing callers' behaviour.
    uint32_t ternary_band_mask;
} sp_band_config_t;

// Resolve the [offset, size) span of band `b` in the coefficient vector.
// Inline-safe: `bc->band_size` for all bands except the last, which absorbs
// any head_dim % n_bands remainder.
static inline void sp_band_span(const sp_band_config_t *bc, int b,
                                int *offset_out, int *size_out) {
    int off = b * bc->band_size;
    int sz  = (b == bc->n_bands - 1) ? (bc->head_dim - off) : bc->band_size;
    *offset_out = off;
    *size_out   = sz;
}

// Compute band configuration from config. Equivalent to
// sp_band_config_init_ext(bc, head_dim, n_bands, band_bits, 0u) — no
// ternary bands. Existing callers keep their current behaviour.
void sp_band_config_init(sp_band_config_t *bc, int head_dim,
                         int n_bands, const int *band_bits);

// Extended initialiser that accepts a ternary band mask (bit b set ⇒
// band b is quantised to {-1, 0, +1} at 2 bpp). Used by the strange-
// attractor stack ternary noise-tail (5/5/4/1.58 — band 3 ternary on top
// of the 5/5/4/3 ship preset). The mask must reference only valid bands;
// bits beyond n_bands are ignored.
void sp_band_config_init_ext(sp_band_config_t *bc, int head_dim,
                             int n_bands, const int *band_bits,
                             uint32_t ternary_band_mask);

// Quantize VHT2 coefficients into banded format.
// vht2_coeffs: input (head_dim floats, already VHT2'd and optionally reordered)
// out:        output buffer (bc->total_bytes)
void sp_band_quantize(const float *vht2_coeffs, uint8_t *out,
                      const sp_band_config_t *bc);

// Dequantize banded format back to VHT2 coefficients.
// in:         compressed buffer (bc->total_bytes)
// vht2_coeffs: output (head_dim floats)
void sp_band_dequantize(const uint8_t *in, float *vht2_coeffs,
                        const sp_band_config_t *bc);

// Progressive (partial) dequantize — read only the first `max_bands` bands.
// Coefficients in bands [0, max_bands) are reconstructed normally.
// Coefficients in bands [max_bands, n_bands) are written as 0.0f.
// max_bands must be in [0, bc->n_bands]; max_bands == bc->n_bands gives
// the same result as sp_band_dequantize. max_bands == 0 zeroes the
// whole vector.
//
// Used by the progressive read path: reconstruct K/V at low fidelity from
// just band 0 (highest-energy coefficients), add bands 1..N on demand
// when attention needs more resolution. The input buffer layout is
// unchanged — bands are still packed contiguously — so this just stops
// reading early and zeros the unread coefficient positions.
//
// The cost saving is in the dequantize math (skipping the unread bands'
// bit-unpack inner loop) and in the inverse VHT2 that follows: zero
// coefficients in the high bands collapse butterfly contributions to
// near-zero on the real-data half of the spectrum, which the compiler
// + cache hierarchy can exploit. For full IO savings on disk-paged
// caches, pair with the per-band-major disk layout (phase 2).
void sp_band_dequantize_partial(const uint8_t *in, float *vht2_coeffs,
                                const sp_band_config_t *bc,
                                int max_bands);

// ============================================================================
// FP8 / FP4 banded quantization (GPU-only, compile-time gated)
// ============================================================================
//
// FP8 (E4M3FN): 8-bit float — higher dynamic range than int8 for smooth
// distributions (V cache). Per-band scaling same as int path.
// Requires: SP_FP8=1 compile flag.
//
// FP4 (MXFP4): 4-bit float with shared exponent — Blackwell tensor cores.
// Requires: SP_FP4=1 compile flag + sm_120+ GPU + CUDA >= 12.8.
//
// GPU host API declarations are in backends/cuda/shannon_prime_fp8.cu.
// CPU fp8 fallback (software emulation) not yet implemented — use int
// quantization on CPU path.

// ============================================================================
// Vilenkin-Hartley Transform — multi-prime basis (research path)
// ============================================================================
//
// VHT2 is the Vilenkin-Hartley Transform over Z/p1Z × Z/p2Z × ... × Z/pkZ;
// at p=2 it reduces to the classical Hadamard butterfly (hence the prior
// "WHT" naming that has been retired).
// Hartley kernel: cas(x) = cos(x) + sin(x)
// Self-inverse for ALL primes: V·V = N·I
// Round-trip error = 0.0000
//
// Progressive prime expansion monotonically increases correlation:
//   Walsh (Z/2Z):              0.9490
//   Z/2Z × Z/3Z:              0.9493
//   Z/2Z × Z/3Z × Z/5Z:      0.9500
//   Z/2Z × Z/3Z × Z/5Z × Z/7Z: 0.9513

typedef struct {
    int      n;                  // Basis dimension (product of primes)
    int      n_primes;           // Number of primes in factorization
    int      primes[8];          // The primes
    float   *basis;              // n×n Vilenkin-Hartley matrix (row-major)
} sp_vilenkin_basis_t;

// Initialize Vilenkin basis for given prime count.
// n_primes=2 → N=6, n_primes=3 → N=30, n_primes=4 → N=210
int sp_vilenkin_init(sp_vilenkin_basis_t *vb, int n_primes);
void sp_vilenkin_free(sp_vilenkin_basis_t *vb);

// Forward transform: project head_dim vector into Vilenkin space.
// Input is zero-padded from head_dim to vb->n if needed.
void sp_vilenkin_forward(const sp_vilenkin_basis_t *vb,
                         const float *input, int head_dim,
                         float *output);

// Inverse transform: reconstruct from Vilenkin coefficients.
// Output is truncated from vb->n back to head_dim.
void sp_vilenkin_inverse(const sp_vilenkin_basis_t *vb,
                         const float *input,
                         float *output, int head_dim);

// Three-pass successive extraction (Z/3Z skeleton, Z/5Z detail, Z/7Z texture)
typedef struct {
    int      n_coeffs;           // Number of retained coefficients
    int     *indices;            // Which Vilenkin indices are retained
    float   *values;             // Coefficient values
} sp_vilenkin_pass_t;

// Extract pass k from (residual) signal. Modifies residual in-place.
int sp_vilenkin_extract_pass(const sp_vilenkin_basis_t *vb,
                             float *residual, int head_dim,
                             float energy_threshold,
                             sp_vilenkin_pass_t *pass);

void sp_vilenkin_pass_free(sp_vilenkin_pass_t *pass);

// ============================================================================
// Sqfree + spinor aggressive path (additive to the ship path)
// ============================================================================
//
//   1. Squarefree basis padding (pad head_dim to clean Vilenkin dimension)
//   2. Knight-ranked mask with Möbius CSR predictor
//   3. N-bit residual quantization + spinor sheet bit
//
// Implementation: core/shannon_prime_sqfree.c
// Ship path (core/shannon_prime.c) is untouched.

// ============================================================================
// Squarefree basis helpers
// ============================================================================

// Known sqfree pad dimensions for common head_dim values:
//   hd=64  → 66  = 2·3·11
//   hd=128 → 154 = 2·7·11
//   hd=256 → 330 = 2·3·5·11
#define SP_SQFREE_PAD_64    66
#define SP_SQFREE_PAD_128   154
#define SP_SQFREE_PAD_256   330

// Test if n is squarefree and factors into {2,3,5,7,11,13}.
bool sp_is_sqfree_factorable(int n);

// Find next sqfree dimension ≥ head_dim that factors into small primes.
int sp_sqfree_pad_dim(int head_dim);

// Pad vector from head_dim to pad_dim with mean-fill.
// out must be at least pad_dim floats. Writes pad_dim values.
void sp_sqfree_pad_f32(const float *in, int head_dim,
                       float *out, int pad_dim);

// Truncate: just copy first head_dim values.
void sp_sqfree_unpad_f32(const float *in, float *out, int head_dim);

// ============================================================================
// Knight-ranked mask + Möbius CSR predictor
// ============================================================================
//
// The Knight mask partitions pad_dim indices into:
//   - Skeleton: top-K squarefree by variance (stored directly)
//   - Residual: non-squarefree not in skeleton (Möbius-predicted + quantized)
//
// The CSR table encodes: for residual index r, predict as
//   pred[r] = Σ μ(d) · skel_vals[slot(n/d)]
// over divisors d of (r+1) where μ(d)≠0 and (n/d)-1 is in skeleton.

typedef struct {
    int      pad_dim;            // Padded dimension
    int      sk_k;               // Skeleton size (actual, may be < requested)
    int     *skeleton_idx;       // (sk_k,) indices into [0, pad_dim)
    int      n_res;              // Number of residual positions
    int     *residual_idx;       // (n_res,) indices into [0, pad_dim)

    // CSR representation of Möbius predictor
    int     *csr_offsets;        // (n_res + 1,) — residual[i] uses terms
                                 //   [offsets[i], offsets[i+1])
    int     *csr_skel_slot;      // (n_terms,) — skeleton slot index
    int8_t  *csr_mu_sign;        // (n_terms,) — μ(d) values (±1)
    int      n_terms;            // Total CSR entries

    // Residual config
    int      residual_bits;      // 1, 2, 3, or 4 (3 is the Pareto point)
    bool     use_spinor;         // Store 1-bit sheet per residual position
} sp_knight_mask_t;

// Build the mask for the given padded dimension.
// variance: pad_dim floats of per-index variance (from calibration), or NULL.
// If NULL, uses index order (squarefree first).
int sp_knight_mask_init(sp_knight_mask_t *mask, int pad_dim, int sk_k,
                        const float *variance);
void sp_knight_mask_free(sp_knight_mask_t *mask);

// ============================================================================
// N-bit symmetric residual quantization
// ============================================================================

// Quantize: levels[i] = round((vals[i] / step) + center), clamped to [0, 2^nbits - 1]
// where step = 2*mag/(L-1), center = (L-1)/2, L = 2^nbits.
// mag is typically mean(|vals|).
void sp_quantize_residual(const float *vals, int n, int nbits, float mag,
                          uint8_t *levels);

// Dequantize: vals[i] = (levels[i] - center) * step
void sp_dequantize_residual(const uint8_t *levels, int n, int nbits, float mag,
                            float *vals);

// ============================================================================
// Sqfree shadow cache — aggressive compression path
// ============================================================================
//
// Same interface as sp_shadow_cache_t but uses:
//   - Vilenkin-Hartley transform on sqfree-padded vectors
//   - Knight skeleton + Möbius CSR predictor
//   - N-bit residual + spinor sheet bit
//
// Validated result (Qwen3-8B Q8 hd=128):
//   K+μ+3bit+spinor 3/3/3/3/3:  PPL 7.32 @ 3.3×
//   (matches MOBIUS default 7.31 @ 2.6×, +27% compression)

typedef struct {
    sp_config_t         config;
    sp_knight_mask_t    mask;
    sp_band_config_t    k_bands;       // Operates on skeleton, not full dim
    sp_band_config_t    v_bands;
    sp_vilenkin_basis_t vilenkin;       // Transform basis for pad_dim

    int                 pad_dim;       // Sqfree padded head dimension
    int                 residual_bits; // 1-4 (default 3)
    bool                use_spinor;    // Enable sheet bit correction

    // Compressed storage per slot = (layer * n_heads + head)
    // Each position stores: banded skeleton + residual levels + mag + sheet bits
    uint8_t           **k_cache;
    uint8_t           **v_cache;

    // Scratch buffers (per-thread serialized)
    float              *pad_scratch;   // pad_dim floats
    float              *coeff_scratch; // pad_dim floats
    float              *pred_scratch;  // n_res floats

    // Optional: ship-path-style Möbius reorder of the Knight skeleton
    // before band quantisation. Opt-in via SHANNON_PRIME_SQFREE_SKEL_MOBIUS=1
    // at init. The mask is built at length sk_k and is a pure permutation
    // of the already-extracted skeleton; distinct from the Knight-CSR
    // predictor's Möbius-function-based μ values (those are always on).
    bool                use_skel_mobius;
    sp_mobius_mask_t    skel_mobius_mask;
    float              *skel_mobius_scratch; // sk_k floats

    // ── Calibration state (transient, freed after calibrate_end) ─────
    bool                calibrating;         // true between begin/end
    double             *calib_sum;           // pad_dim doubles: Σ x_i
    double             *calib_sum2;          // pad_dim doubles: Σ x_i²
    int                 calib_n;             // number of vectors fed
    int                 max_seq_len;         // stashed for re-init
} sp_sqfree_cache_t;

int  sp_sqfree_cache_init(sp_sqfree_cache_t *sc, const sp_config_t *cfg,
                          int max_seq_len, int residual_bits, bool use_spinor);
void sp_sqfree_cache_free(sp_sqfree_cache_t *sc);

// ── Adaptive calibration (L/2 phase-transition + variance-ranked skeleton) ──
//
// Calibration collects raw KV vectors from a warmup pass, transforms them
// into the Vilenkin domain, and accumulates per-coefficient variance. When
// calibration ends, it rebuilds the Knight mask with real variance data and
// the L/2 skeleton size, then re-initialises the band quantisers to match.
//
// Usage:
//   sp_sqfree_calibrate_begin(sc);        // allocate accumulators
//   for each warmup vector:
//     sp_sqfree_calibrate_feed(sc, vec);  // pad → Vilenkin → accumulate
//   sp_sqfree_calibrate_end(sc);          // rebuild mask + bands
//
// After calibrate_end, write/read use the adaptive mask. If calibrate is
// never called, the cache uses the default L/2 skeleton with index-order
// (squarefree-first) ranking — still correct, just not data-adapted.

// Begin calibration: allocates per-coefficient running accumulators.
// Returns 0 on success, -1 on alloc failure.
int  sp_sqfree_calibrate_begin(sp_sqfree_cache_t *sc);

// Feed one raw KV vector (head_dim floats, NOT padded). The vector is
// sqfree-padded and Vilenkin-transformed internally; per-coefficient
// squared values are accumulated for variance estimation.
void sp_sqfree_calibrate_feed(sp_sqfree_cache_t *sc, const float *vec);

// End calibration: computes per-coefficient variance from accumulators,
// rebuilds the Knight mask with variance ranking at sk_k = pad_dim/2,
// re-initialises band quantisers, and frees the accumulators.
// Returns 0 on success, -1 if no vectors were fed.
int  sp_sqfree_calibrate_end(sp_sqfree_cache_t *sc);

// Write/read — same signatures as sp_shadow_cache but uses the sqfree path
void sp_sqfree_write_k(sp_sqfree_cache_t *sc,
                       int layer, int head, int pos,
                       const float *k_vec);
void sp_sqfree_write_v(sp_sqfree_cache_t *sc,
                       int layer, int head, int pos,
                       const float *v_vec);
void sp_sqfree_read_k(const sp_sqfree_cache_t *sc,
                      int layer, int head, int pos,
                      float *k_out);
void sp_sqfree_read_v(const sp_sqfree_cache_t *sc,
                      int layer, int head, int pos,
                      float *v_out);

// Batch variants
void sp_sqfree_write_k_batch(sp_sqfree_cache_t *sc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *k_vecs);
void sp_sqfree_read_k_batch(const sp_sqfree_cache_t *sc,
                            int layer, int head,
                            int start_pos, int n_pos,
                            float *k_out);

// ============================================================================
// Hierarchical Vilenkin Predictor — small skeleton → full reconstruction
// ============================================================================
//
// The Vilenkin basis on sqfree-padded dims has Kronecker product structure:
//   hd=128 → pad=154 = 2·7·11 → Z/2Z × Z/7Z × Z/11Z
//
// The Kronecker sub-projection over the first k primes gives a natural
// hierarchy of nested skeletons:
//   Level 1: Z/2Z             →   2 coeffs ( 1.3%)
//   Level 2: Z/2Z × Z/7Z     →  14 coeffs ( 9.1%)
//   Level 3: full             → 154 coeffs (100%)
//
// The hierarchical predictor stores only the level-2 sub-projection
// (the "core skeleton") and uses a calibrated linear map W to predict
// all remaining coefficients:
//
//   target_coeffs ≈ W · skeleton_coeffs
//
// W is a (n_skeleton × n_target) fp16 matrix, calibrated from warmup
// KV vectors via ridge regression. It is per-(layer, head) but amortized
// across all sequence positions.
//
// Write path: pad → Vilenkin → extract core skeleton → band quantize
//             → W·skel predicts targets → quantize residual → store
// Read path:  dequant skeleton → W·skel → add residual → scatter
//             → inverse Vilenkin → unpad
//
// Storage per position:  n_skeleton × skel_bits + n_target × res_bits
// Predictor overhead:    n_skeleton × n_target × 2 bytes per (layer, head)
//
// For hd=128 (pad=154, 14-skeleton):
//   Per-position:  14×5 + 140×2 = 350 bits = 43.75 bytes
//   vs L/2 Knight: 77×4.25 + 77×3 = ~559 bits = ~70 bytes → 1.6× smaller
//   vs fp16:       154×16 = 2464 bits = 308 bytes → 7.0× compression
//
// With aggressive skeleton quantisation (14×4 = 56 bits) and 1-bit residual:
//   Per-position:  56 + 140 = 196 bits = 24.5 bytes → 12.6× compression

// Maximum hierarchy depth (number of prime factors in pad_dim)
#define SP_HIER_MAX_LEVELS  5

typedef struct {
    int     pad_dim;            // Sqfree padded dimension
    int     n_primes;           // Number of primes in pad_dim factorization
    int     primes[SP_HIER_MAX_LEVELS]; // The prime factors

    // Core skeleton: Kronecker sub-projection indices.
    // These are the coefficient indices belonging to the sub-space spanned
    // by the first `hier_level` primes. Model-independent — determined
    // entirely by pad_dim and hier_level.
    int     hier_level;         // How many primes define the skeleton (1..n_primes-1)
    int     n_skeleton;         // Number of core skeleton coefficients
    int    *skeleton_idx;       // [n_skeleton] indices into [0, pad_dim)

    // Target: everything not in the skeleton
    int     n_target;           // = pad_dim - n_skeleton
    int    *target_idx;         // [n_target] indices into [0, pad_dim)

    // Calibrated linear predictor: target ≈ W · skeleton
    // W is stored in row-major order: W[t * n_skeleton + s]
    // where t is the target index and s is the skeleton index.
    // NULL before calibration; set to fp32 after calibrate_end.
    // (Future: quantise to fp16 for storage efficiency.)
    uint16_t *W;                // [n_target × n_skeleton] or NULL
    bool    calibrated;         // true after successful calibrate_end

    // Calibration accumulators (transient, freed after calibrate_end)
    bool    calibrating;
    double *calib_XtX;          // [n_skeleton × n_skeleton] Σ skeleton·skeleton^T
    double *calib_XtY;          // [n_skeleton × n_target]   Σ skeleton·target^T
    int     calib_n;            // number of vectors fed

    // Band quantisation for the core skeleton
    sp_band_config_t skel_bands;

    // Residual config for predicted targets
    int     target_res_bits;    // 1-4 bits for target residual (default 2)
} sp_hier_predictor_t;

// Build the Kronecker sub-projection for the given pad_dim and hierarchy level.
// hier_level=0 → automatic (picks second-to-last prime grouping).
// Returns 0 on success.
int  sp_hier_predictor_init(sp_hier_predictor_t *hp, int pad_dim,
                            int hier_level, int target_res_bits,
                            int skel_n_bands, const int *skel_band_bits);
void sp_hier_predictor_free(sp_hier_predictor_t *hp);

// Calibration: collect Vilenkin-domain vectors, fit W via ridge regression.
int  sp_hier_calibrate_begin(sp_hier_predictor_t *hp);
void sp_hier_calibrate_feed(sp_hier_predictor_t *hp, const float *vilenkin_coeffs);
int  sp_hier_calibrate_end(sp_hier_predictor_t *hp);
// Sticky-EMA variant. Runs the full solve, then blends the fresh W with the
// caller's snapshot of the previous W. keep_frac in [0,1]; W_prev NULL or
// keep_frac <= 0 behave like sp_hier_calibrate_end. See implementation for
// the usage contract.
int  sp_hier_calibrate_end_blend(sp_hier_predictor_t *hp,
                                 const uint16_t *W_prev, float keep_frac);

// Predict targets from skeleton coefficients.
// skeleton_vals: [n_skeleton] input
// target_out:    [n_target] output (predicted values, before residual correction)
void sp_hier_predict(const sp_hier_predictor_t *hp,
                     const float *skeleton_vals, float *target_out);

// ============================================================================
// Hierarchical sqfree cache — maximum compression path
// ============================================================================
//
// Like sp_sqfree_cache_t but uses the hierarchical Vilenkin predictor
// instead of Knight mask + Möbius CSR. Achieves higher compression by
// storing a much smaller skeleton (~9% vs 50%) at the cost of a
// per-(layer,head) predictor matrix.
//
// The predictor matrices are calibrated during the first prefill and
// then frozen. Storage layout per position:
//   [banded skeleton (skel_bands.total_bytes)]
//   [residual magnitude (4 bytes)]
//   [packed residual levels (n_target * target_res_bits / 8)]

typedef struct {
    sp_config_t          config;
    sp_vilenkin_basis_t  vilenkin;       // Transform basis for pad_dim
    int                  pad_dim;
    int                  max_seq_len;

    // One predictor per (layer, head) slot. Each predictor has its own
    // calibrated W matrix — different attention heads learn different
    // spectral patterns, so per-head predictors beat a shared one.
    int                  n_slots;        // = n_layers × n_heads_kv
    sp_hier_predictor_t *predictors;     // [n_slots]

    // Compressed storage
    uint8_t            **k_cache;        // [n_slots][max_seq × bytes_per_pos]
    uint8_t            **v_cache;
    int                  k_bytes_per_pos;
    int                  v_bytes_per_pos;

    // Scratch buffers (allocated once, reused every write/read)
    float               *pad_scratch;    // pad_dim
    float               *coeff_scratch;  // pad_dim
    float               *skel_scratch;   // n_skeleton
    float               *target_scratch; // n_target
    float               *pred_scratch;   // n_target
    uint8_t             *levels_scratch; // n_target (residual quant levels)
} sp_hier_cache_t;

// Initialize hierarchical cache.
// hier_level: 0 = automatic, 1..n_primes-1 = explicit.
// skel_bits_csv: band bit allocation for skeleton (e.g. "5,5" for 2 bands).
// target_res_bits: 1-4 bits for target residuals.
int  sp_hier_cache_init(sp_hier_cache_t *hc, const sp_config_t *cfg,
                        int max_seq_len, int hier_level,
                        int skel_n_bands, const int *skel_band_bits,
                        int target_res_bits);
void sp_hier_cache_free(sp_hier_cache_t *hc);

// Calibration: feeds Vilenkin-domain vectors to ALL slot predictors.
// slot = layer * n_heads_kv + head. Caller decides which slot(s) to feed.
int  sp_hier_cache_calibrate_begin(sp_hier_cache_t *hc);
void sp_hier_cache_calibrate_feed(sp_hier_cache_t *hc, int slot,
                                  const float *raw_vec);
int  sp_hier_cache_calibrate_end(sp_hier_cache_t *hc);
// Sticky-EMA variant: blends each slot's fresh-solve W with the previous W.
// keep_frac in [0,1]; 0 is equivalent to sp_hier_cache_calibrate_end.
int  sp_hier_cache_calibrate_end_ema(sp_hier_cache_t *hc, float keep_frac);

// Write/read
void sp_hier_cache_write_k(sp_hier_cache_t *hc,
                           int layer, int head, int pos,
                           const float *k_vec);
void sp_hier_cache_write_v(sp_hier_cache_t *hc,
                           int layer, int head, int pos,
                           const float *v_vec);
void sp_hier_cache_read_k(const sp_hier_cache_t *hc,
                          int layer, int head, int pos,
                          float *k_out);
void sp_hier_cache_read_v(const sp_hier_cache_t *hc,
                          int layer, int head, int pos,
                          float *v_out);

// ============================================================================
// Scaling law — K-corr → PPL design rule
// ============================================================================
//
// Empirical: log(PPL/base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)
//
// Use as pre-bench filter: if predicted_ppl_ratio > 1.05, skip the config.

float sp_predicted_ppl_ratio(float k_corr, float params_b, int bits);
bool  sp_is_pareto_viable(float k_corr, float params_b, int bits,
                          float budget);
float sp_min_k_corr_for_budget(float params_b, int bits, float budget);


// ============================================================================
// Shadow cache — the integration point
// ============================================================================
//
// The shadow cache intercepts KV writes, compresses via VHT2,
// and reconstructs on read. This is the interface backends implement.
//
// Architecture:
//   Write path: raw KV → VHT2 → Möbius reorder → band quantize → store
//   Read path:  load → band dequantize → Möbius unreorder → VHT2 (self-inverse) → KV

typedef struct {
    sp_config_t         config;
    sp_band_config_t    k_bands;
    sp_band_config_t    v_bands;
    sp_mobius_mask_t     mobius_mask;

    // Compressed storage (allocated by backend)
    uint8_t           **k_cache;     // [layer * n_heads + head][seq_pos] → compressed
    uint8_t           **v_cache;     // same layout
    int                *seq_len;     // per-layer current sequence length

    // Scratch buffers (callee-owned; callers must serialize across threads).
    // vht2_scratch     primary VHT2 working buffer for write path
    // mobius_scratch   scratch for Möbius reorder tmp (hot path, avoids malloc)
    // read_scratch     VHT2 buffer for read path (avoids malloc per read)
    float              *vht2_scratch;   // head_dim floats
    float              *mobius_scratch; // head_dim floats
    float              *read_scratch;   // head_dim floats

    // ── Variance-ranked reorder (adaptive alternative to Möbius) ─────
    // When calibrated, `var_order` replaces `mobius_mask` on the write/read
    // path: highest-variance VHT2 coefficients are pushed to the front
    // (high-bit bands), lowest-variance to the back (low-bit bands).
    bool                use_var_reorder;  // false until calibrated
    int                *var_order;        // head_dim permutation (sorted by variance desc)
    int                *var_unorder;      // inverse permutation for read path
    // Calibration accumulators (transient)
    bool                calibrating;
    double             *calib_sum;        // head_dim doubles
    double             *calib_sum2;       // head_dim doubles
    int                 calib_n;
} sp_shadow_cache_t;

// Initialize shadow cache. Backend allocates compressed storage.
int sp_shadow_cache_init(sp_shadow_cache_t *sc, const sp_config_t *cfg);
void sp_shadow_cache_free(sp_shadow_cache_t *sc);

// ── Variance-ranked calibration for the ship path ───────────────────
// Same lifecycle as sqfree calibration. After calibrate_end, the write
// path reorders VHT2 coefficients by variance instead of Möbius order,
// so high-variance coefficients land in the highest-bit bands.
int  sp_shadow_calibrate_begin(sp_shadow_cache_t *sc);
void sp_shadow_calibrate_feed(sp_shadow_cache_t *sc, const float *vec);
int  sp_shadow_calibrate_end(sp_shadow_cache_t *sc);

// Compress and store a K vector.
// layer, head: identifies the cache slot
// pos: sequence position
// k_vec: raw K vector (head_dim floats, already RoPE'd)
void sp_shadow_write_k(sp_shadow_cache_t *sc,
                       int layer, int head, int pos,
                       const float *k_vec);

// Compress and store a V vector.
void sp_shadow_write_v(sp_shadow_cache_t *sc,
                       int layer, int head, int pos,
                       const float *v_vec);

// Reconstruct a K vector from compressed storage.
// k_out: output buffer (head_dim floats)
void sp_shadow_read_k(const sp_shadow_cache_t *sc,
                      int layer, int head, int pos,
                      float *k_out);

// Reconstruct a V vector from compressed storage.
void sp_shadow_read_v(const sp_shadow_cache_t *sc,
                      int layer, int head, int pos,
                      float *v_out);

// Partial reads — reconstruct using only the first `max_bands` spectral
// bands. Higher bands are treated as zero coefficients, which propagates
// cleanly through the inverse Möbius reorder + inverse VHT2 to give a
// partial-fidelity reconstruction. Used by the phase 3 attention
// short-circuit: low-fidelity band-0-only reads first, promote to more
// bands when entropy demands.
//
// max_bands clamps into [0, n_bands]. INT_MAX or any value >= n_bands
// is identical to sp_shadow_read_{k,v} (full reconstruction).
// max_bands == 0 returns an all-zero vector.
void sp_shadow_read_k_partial(const sp_shadow_cache_t *sc,
                              int layer, int head, int pos,
                              float *k_out, int max_bands);
void sp_shadow_read_v_partial(const sp_shadow_cache_t *sc,
                              int layer, int head, int pos,
                              float *v_out, int max_bands);

// Batch variants — process n_pos contiguous vectors with a single setup.
// k_vecs / v_vecs must be contiguous [n_pos × head_dim] arrays.
// These are the real batches: they run in a tight loop over the persistent
// scratch buffers (no per-vector malloc) so they amortize pipeline cost
// across the batch. start_pos + n_pos must fit within the cache's max_seq.
void sp_shadow_write_k_batch(sp_shadow_cache_t *sc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *k_vecs);
void sp_shadow_write_v_batch(sp_shadow_cache_t *sc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *v_vecs);
void sp_shadow_read_k_batch (const sp_shadow_cache_t *sc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             float *k_out);
void sp_shadow_read_v_batch (const sp_shadow_cache_t *sc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             float *v_out);

// ============================================================================
// Ricci Sentinel — reactive p=3 energy drift monitor
// ============================================================================
//
// The p=3 spectral band energy is a scalar proxy for metric structural
// integrity. If p=3 energy drifts from its calibrated baseline, the
// compression errors are becoming "timelike" (aligned with the information
// flow) rather than "spacelike" (orthogonal). The metric is developing a
// Cauchy horizon — a boundary beyond which the compressed past no longer
// determines the correct future.
//
// The sentinel tracks an exponential moving average of the ratio:
//   p3_ratio = current_p3_energy / calibrated_p3_energy
//
// When |1 - p3_ratio| exceeds the metric_criticality threshold, a Cauchy
// reset is recommended. The threshold is model-size-dependent:
//   small models (1B):  0.05  (fragile metric, narrow residual stream)
//   large models (8B+): 0.15  (robust metric, strong skip geodesics)

typedef struct {
    int      p3_band_offset;
    int      p3_band_size;
    double   p3_energy_calibrated;
    bool     calibrated;
    double   p3_ema;
    double   ema_alpha;
    int      n_samples;
    double   metric_criticality;
    bool     reset_recommended;
    double   calib_p3_sum;
    int      calib_n;
    // Warm-up gate: while n_samples < warmup_samples, the EMA is
    // still being seeded (one sample sets it to `ratio` alone, which
    // for lossy compression is nowhere near 1.0). Suppressing
    // reset_recommended during this window kills a runaway where
    // every reset zeros the EMA and the next sample immediately
    // re-triggers. Default 32 gives ~2 half-lives at alpha=0.05.
    int      warmup_samples;
} sp_ricci_sentinel_t;

int  sp_ricci_init(sp_ricci_sentinel_t *rs, const sp_band_config_t *band_cfg,
                   float params_b);
void sp_ricci_calibrate_feed(sp_ricci_sentinel_t *rs,
                             const float *vht2_coeffs, int hd);
void sp_ricci_calibrate_end(sp_ricci_sentinel_t *rs);
bool sp_ricci_check(sp_ricci_sentinel_t *rs,
                    const float *vht2_coeffs, int hd);
void sp_ricci_reset(sp_ricci_sentinel_t *rs);
double sp_ricci_drift(const sp_ricci_sentinel_t *rs);

// ============================================================================
// Mertens Oracle — zeta-guided proactive reset scheduling
// ============================================================================
//
// M(n) = Σ_{k=1}^{n} μ(k) tracks the squarefree/non-squarefree balance.
// Its spectral decomposition via zeta zeros creates oscillations whose
// half-period at typical context scales (n ~ 256-2048) is 200-500 tokens,
// matching empirically-observed optimal reset windows.

#define SP_MERTENS_MAX_ZEROS    50
#define SP_MERTENS_MAX_SCHEDULE 1024

typedef struct {
    int      n_zeros;
    double   gamma[SP_MERTENS_MAX_ZEROS];
    int      n_schedule;
    int      schedule[SP_MERTENS_MAX_SCHEDULE];
    float    risk[SP_MERTENS_MAX_SCHEDULE];
    float   *risk_cache;          // O(1) direct lookup map [max_ctx]
    int      max_ctx;
    int      schedule_idx;
} sp_mertens_oracle_t;

int    sp_mertens_init(sp_mertens_oracle_t *mo, int max_ctx);
void   sp_mertens_free(sp_mertens_oracle_t *mo);   // releases risk_cache
float  sp_mertens_risk(const sp_mertens_oracle_t *mo, int pos);
int    sp_mertens_next_risk(const sp_mertens_oracle_t *mo,
                            int current_pos, int lookahead);
void   sp_mertens_advance(sp_mertens_oracle_t *mo, int pos);
double sp_mertens_eval(const sp_mertens_oracle_t *mo, int n);

// ============================================================================
// Cauchy Reset — manifold re-anchoring
// ============================================================================
//
// sp_cauchy_check returns: 0 = no reset, 1 = full reset recommended,
// 2 = partial reset OK (hierarchical only; for shadow/sqfree treat as 1).

typedef struct {
    int      mode;              // 0=off, 1=fixed-N, 2=dynamic (Ricci+Mertens)
    int      fixed_n;
    int      partial_window;
    int      last_reset_pos;
    int      total_resets;
    sp_ricci_sentinel_t  *ricci;
    sp_mertens_oracle_t  *mertens;
} sp_cauchy_ctrl_t;

void sp_cauchy_init(sp_cauchy_ctrl_t *cc, int mode, int fixed_n,
                    sp_ricci_sentinel_t *ricci,
                    sp_mertens_oracle_t *mertens);
int  sp_cauchy_check(sp_cauchy_ctrl_t *cc, int pos);
void sp_cauchy_record_reset(sp_cauchy_ctrl_t *cc, int pos);
void sp_cauchy_print_stats(const sp_cauchy_ctrl_t *cc);

// ============================================================================
// PrimePE — Lattice-Aligned RoPE Frequency Injection
// ============================================================================
//
// Computes per-dimension frequency factors that blend lattice-drawn integer
// frequencies with the standard geometric RoPE schedule. Proven −0.6% to
// −0.8% PPL improvement across architectures at zero runtime cost.
//
// The output is a float array of length n_rot/2 containing multipliers
// on the geometric base frequencies. Pass as freq_factors to ggml_rope_ext
// (or write into the model's rope_freqs tensor slot).
//
// alpha controls the blend ratio:
//   0.0 = pure geometric (identity factors, all 1.0)
//   0.17 = recommended default (−0.6% PPL on Q6_K, deployment-robust)
//   0.22 = aggressive (−0.8% PPL on Q8, slightly less robust)
//   1.0 = pure lattice (NOT recommended — destroys per-head diversity)
//
// The function allocates the output array via malloc(). Caller must free().
// Returns NULL if n_freqs <= 0 or malloc fails.

float *sp_prime_pe_freq_factors(int n_freqs, float freq_base, float alpha);

// Number of frequency pairs for a given head_dim (= head_dim / 2).
static inline int sp_prime_pe_n_freqs(int head_dim) { return head_dim / 2; }

// ============================================================================
// Diagnostics
// ============================================================================

// Compute correlation between original and reconstructed vector.
float sp_correlation_f32(const float *a, const float *b, int n);

// Compute compression ratio for current config.
float sp_compression_ratio(const sp_config_t *cfg);

// Print config summary to stderr.
void sp_config_print(const sp_config_t *cfg);

// ============================================================================
// Disk Serialization — save/load compressed cache to/from files
// ============================================================================
//
// Binary format per layer file (compatible with Archimedes VHT2 v2):
//   64-byte header: uint32_t[16]
//     [0] magic 0x56485432 ("VHT2")  [1] version = 2
//     [2] packed_stride (bytes/head)  [3] n_positions
//     [4] n_heads                     [5] cache_type (0=shadow, 1=sqfree, 2=hier)
//     [6] model_hash_lo              [7] model_hash_hi
//     [8..15] reserved
//   Followed by n_positions * n_heads * packed_stride bytes of compressed data.
//
// File naming: {prefix}.l{layer}.k.vht2, {prefix}.l{layer}.v.vht2
//
// Hot/cold store modes:
//   SP_STORE_GPU_ONLY  — compressed data lives only in VRAM
//   SP_STORE_DUAL      — GPU hot + CPU cold (default)
//   SP_STORE_COLD_ONLY — CPU pinned only (no GPU copy)

#define SP_CACHE_MAGIC    0x56485432u  // "VHT2"

// Cache file version. v2 was per-(head,pos) interleaved bands. v3 is
// band-major (band 0 across all heads/positions, then band 1, etc.) so
// partial loaders can fseek directly to a band's region without reading
// irrelevant high-band bytes. The shadow-cache writer uses v3; sqfree
// and hier writers stay on v2 until their loaders are migrated.
#define SP_CACHE_VERSION       2  // legacy / sqfree / hier
#define SP_CACHE_VERSION_BAND_MAJOR 3  // shadow cache band-major layout

enum {
    SP_STORE_GPU_ONLY  = 0,
    SP_STORE_DUAL      = 1,
    SP_STORE_COLD_ONLY = 2,
};

// Save compressed shadow cache to disk. One file per layer per K/V.
// prefix: path prefix (e.g. "/tmp/model_cache" → /tmp/model_cache.l0.k.vht2)
// n_pos: number of positions to save (0 = all written so far)
// model_hash: optional FNV-1a of model path (0 = skip validation on load)
// Returns 0 on success, -1 on error.
int sp_shadow_cache_save(const sp_shadow_cache_t *sc,
                         const char *prefix, int n_pos,
                         uint64_t model_hash);

// Load compressed shadow cache from disk. Overwrites current cache contents.
// The cache must already be initialized with matching config (head_dim, bands, etc).
// Returns number of positions loaded, or -1 on error.
int sp_shadow_cache_load(sp_shadow_cache_t *sc,
                         const char *prefix,
                         uint64_t expected_hash);

// Progressive (partial) load — read only the first `max_bands` bands from
// disk, leaving the higher bands' regions of the in-memory cache zeroed.
// Requires v3 (band-major) format; v2 files are rejected with -1 because
// per-(head,pos) interleaved bands can't be partially read without
// touching every byte. max_bands is clamped into [0, n_bands].
//
// The IO win: with v3 band-major layout, reading bands [0, max_bands)
// only touches the first sum(band_b region size) bytes of the file. On
// NVMe, that's a single contiguous read; on Optane / shared-memory
// tiers, it's a sub-millisecond pop. Compose with the System 1 fast path
// for attention: read band 0 from RAM, short-circuit if the dot product
// is determined; otherwise call this with max_bands=2 etc. as needed.
//
// Returns number of positions loaded (the n_pos stored in the header)
// or -1 on error.
int sp_shadow_cache_load_partial(sp_shadow_cache_t *sc,
                                 const char *prefix,
                                 uint64_t expected_hash,
                                 int max_bands);

// Save/load for sqfree cache (same format, different packed_stride).
int sp_sqfree_cache_save(const sp_sqfree_cache_t *sc,
                         const char *prefix, int n_pos,
                         uint64_t model_hash);
int sp_sqfree_cache_load(sp_sqfree_cache_t *sc,
                         const char *prefix,
                         uint64_t expected_hash);

// Save/load for hierarchical cache.
// Hierarchical saves both skeleton bands AND the W predictor matrices.
int sp_hier_cache_save(const sp_hier_cache_t *sc,
                       const char *prefix, int n_pos,
                       uint64_t model_hash);
int sp_hier_cache_load(sp_hier_cache_t *sc,
                       const char *prefix,
                       uint64_t expected_hash);

// Utility: compute FNV-1a hash of a string (for model_hash).
uint64_t sp_fnv1a_hash(const char *str, size_t len);

#ifdef __cplusplus
}
#endif

#endif // SHANNON_PRIME_CORE_H
