// Shannon-Prime BurnHard — Residue matmul (host reference).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// The mathematical foundation under GGML_OP_BURN_MOE. Given two fp32
// matrices A [M,K] and B [K,N], computes C = A·B by:
//
//   1. Discretise A,B → int32 with scale S = sp_burnhard_safe_q_max(K)
//   2. Split into M1 and M2 residue forms (one per chamber)
//   3. Accumulate K×K products mod M1 (RTX chamber) and mod M2 (UHD chamber)
//   4. Garner-join the two int64 partial sums per (m,n) cell
//   5. Unscale int64 → fp32 by S²
//
// In Session 2 (this file) the M1 and M2 chambers run on the same host —
// it's the A/B test that proves the residue path matches stock fp32 matmul.
// Session 3 will swap chamber 1 for a CUDA kernel, chamber 2 for a Level
// Zero kernel; the host code becomes the Garner glue + the unscale tail.
//
// Convention: row-major. C[m,n] = sum_k A[m,k] * B[k,n].
// All buffers caller-allocated. M, N, K need not be powers of 2.

#ifndef BURNHARD_RESIDUE_MATMUL_H
#define BURNHARD_RESIDUE_MATMUL_H

#include "burnhard_crt.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Reference path: scalar host implementation. Used by the verifier and
// (for now) by the BURN_MOE custom op when no GPU dispatch is bound.
//
// Picks scale internally via sp_burnhard_safe_q_max(K) — the matmul
// dispatcher never has to think about overflow safety.
//
// Returns 0 on success, negative on OOM / invalid args.
int sp_burnhard_residue_matmul_fp32(const float *A,
                                     const float *B,
                                     float *C,
                                     size_t M, size_t K, size_t N);

// "Direct" reference fp32 matmul — included so the verifier doesn't have
// to depend on BLAS. C[m,n] = sum_k A[m,k] * B[k,n]. Caller-allocated C.
void sp_reference_matmul_fp32(const float *A,
                               const float *B,
                               float *C,
                               size_t M, size_t K, size_t N);

#ifdef __cplusplus
}
#endif

#endif // BURNHARD_RESIDUE_MATMUL_H
