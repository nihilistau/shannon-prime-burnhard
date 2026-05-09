// Shannon-Prime BurnHard — CRT residue math (in-flight Shredder + Garner).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// The two co-prime moduli that braid the heterogeneous Furnace:
//
//   M1 = 2^31 - 1                    (Mersenne prime, RTX 2060 / "M1 chamber")
//   M2 = 2147483629                  (next prime below M1, UHD 750 / "M2 chamber")
//   M1^(-1) mod M2 = 2028178983      (Garner reconstruction constant)
//
// Verified by extended Euclidean:
//   M1 = 1 * M2 + 18
//   M2 = 119304646 * 18 + 1
//   ⇒ 1 = 119304647*M2 - 119304646*M1
//   ⇒ M1^(-1) ≡ -119304646 ≡ M2 - 119304646 ≡ 2028178983  (mod M2)
//
// USAGE:
//   • Per-element shred: sp_burnhard_shred_one(q) → (r1, r2)
//   • Per-element join : sp_burnhard_garner_one(r1, r2) → q
//   • Bulk shred       : sp_burnhard_shred_fp32(...) — see shredder_q4k_inflight.c
//   • Bulk join        : sp_burnhard_garner_fp32(...)
//
// SCALING:
//   Float weights (fp32 from Q4_K dequant) are converted to int32 via a
//   global scale S. Recommended S = 2^28 (default in shredder_q4k_inflight.c)
//   which keeps |q| ≤ 2^28 ≪ M1/2, leaving 3 bits of headroom for any
//   intermediate accumulation done before mod-reduction.
//
// THREAD SAFETY:
//   All functions are pure (no globals, no locks). Safe to call from any
//   thread, vectorised, etc.

#ifndef BURNHARD_CRT_H
#define BURNHARD_CRT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Mathematical constants — DO NOT EDIT WITHOUT REWORKING shred_verify -----
#define SP_M1                 2147483647LL   // 2^31 - 1, Mersenne prime
#define SP_M2                 2147483629LL   // largest prime below M1
#define SP_M1_INV_MOD_M2      2028178983LL   // M1^-1 mod M2, Garner constant
#define SP_M_PRODUCT          (SP_M1 * SP_M2) // ~4.6e18, fits in int64

// --- Defaults ---------------------------------------------------------------
// 2^28 keeps quantisation error at ~3.7e-9 per weight (well below the 1e-6
// gate) while leaving room for a few thousand matmul accumulations before
// any mod-reduction is required.
#define SP_BURNHARD_DEFAULT_SCALE   ((float)(1 << 28))

// --- Types ------------------------------------------------------------------
// One residue pair. r1 ∈ [0, M1), r2 ∈ [0, M2). int32 is sufficient since
// both moduli fit in 31 bits. We use signed for ergonomics with the matmul
// path; the caller normalises to non-negative before mod-reducing.
typedef struct {
    int32_t r1;
    int32_t r2;
} burnhard_residue_t;

// --- Per-element kernels (inline, branch-free in the hot path) --------------
//
// Shred: take a discretised int32 weight q and produce the residue pair.
// q may be negative; the result is normalised to [0, M).
static inline burnhard_residue_t
sp_burnhard_shred_one(int32_t q) {
    burnhard_residue_t out;
    int64_t r1 = (int64_t)q % SP_M1;
    int64_t r2 = (int64_t)q % SP_M2;
    if (r1 < 0) r1 += SP_M1;
    if (r2 < 0) r2 += SP_M2;
    out.r1 = (int32_t)r1;
    out.r2 = (int32_t)r2;
    return out;
}

// Garner: recover q from its two residues. Result is in (-M1*M2/2, M1*M2/2].
//   q = r1 + M1 * ((r2 - r1) * M1^-1  mod  M2)
// then sign-extend across the M1*M2/2 boundary so we get back negative
// weights as negative int64 (rather than huge positives mod M).
static inline int64_t
sp_burnhard_garner_one(int32_t r1, int32_t r2) {
    int64_t diff = ((int64_t)r2 - (int64_t)r1) % SP_M2;
    if (diff < 0) diff += SP_M2;
    int64_t t = (diff * SP_M1_INV_MOD_M2) % SP_M2;
    int64_t q = (int64_t)r1 + SP_M1 * t;
    // q ∈ [0, M1*M2). Lift the upper half to negatives so signed weights
    // round-trip cleanly. The threshold is M_PRODUCT/2.
    if (q > (SP_M_PRODUCT >> 1)) q -= SP_M_PRODUCT;
    return q;
}

// --- Bulk kernels (defined in shredder_q4k_inflight.c) ----------------------
//
// Forward: discretise fp32 → int32 (using `scale`) → split into M1/M2.
// All output buffers must hold at least `n` elements; out_r1/out_r2 may be
// pinned (e.g. cudaHostAlloc) when the caller needs zero-copy DMA.
void sp_burnhard_shred_fp32(const float *in_fp32,
                             int32_t *out_r1,
                             int32_t *out_r2,
                             float scale,
                             size_t n);

// Inverse: Garner-join → fp32 (divide by `scale` inverse). For the Session-1
// verifier; matmul reconstruction lives in the dispatcher in Session 2.
void sp_burnhard_garner_fp32(const int32_t *in_r1,
                              const int32_t *in_r2,
                              float *out_fp32,
                              float scale,
                              size_t n);

// --- Diagnostics ------------------------------------------------------------
// Returns the maximum |q| (post-discretisation) safely representable without
// risking accumulator overflow over `inner_dim` matmul terms. Useful for the
// matmul dispatcher to pick a scale that fits.
static inline int32_t
sp_burnhard_safe_q_max(size_t inner_dim) {
    // Each product is up to q_max^2; we want q_max^2 * inner_dim ≤ 2^62
    // (one bit shy of int64 max so the sign bit stays free).
    // q_max ≤ sqrt(2^62 / inner_dim).
    if (inner_dim == 0) return (int32_t)((1LL << 30) - 1);
    int64_t bound = 1LL << 62;
    int64_t q2 = bound / (int64_t)inner_dim;
    int64_t q  = 1;
    while ((int64_t)(q + 1) * (int64_t)(q + 1) <= q2) q++;
    if (q > (1LL << 30) - 1) q = (1LL << 30) - 1;
    return (int32_t)q;
}

#ifdef __cplusplus
}
#endif

#endif // BURNHARD_CRT_H
