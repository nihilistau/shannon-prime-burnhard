/*
 * sp_shredder_crt.h — Shannon-Prime Beast Canyon: Residue-Aware Shredder
 *
 * Dequantizes Q8_0 weights directly into CRT residue streams in a single
 * AVX-512 pass.  Eliminates the fp32 scratch buffer entirely.
 *
 * WHY THIS EXISTS:
 *   The naive path is: Q8_0 → dequant to fp32 scratch → CRT split → residues.
 *   That fp32 scratch buffer is expert_dim × sizeof(float) per row — for a
 *   14336-dim MoE expert, that's 56 KB/row, blowing L1 and thrashing L2.
 *
 *   The shredder fuses dequant + CRT split into one pass: each Q8_0 block
 *   (32 elements + 1 scale) is decoded, scaled to the 28-bit CRT ceiling,
 *   and split into M1/M2 residue streams in-register.  The fp32 intermediate
 *   never touches memory.
 *
 * 28-BIT CEILING (SP_SHREDDER_CRT_Q_MAX = 268328157.0f):
 *   For safe K=4096 accumulation without overflow, each CRT residue must
 *   fit in 28 bits.  The ceiling is chosen so that K dot-product terms
 *   (each ≤ Q_MAX²) sum without exceeding the int64_t accumulator range.
 *   Specifically: K × Q_MAX² < 2^63, solved for K=4096 gives Q_MAX ≈ 2^28.
 *
 * LLC-RESIDENT WRITE PATH FOR INTEL UHD SVM:
 *   The res2_buffer is allocated via SVM (Shared Virtual Memory) on Intel
 *   UHD integrated graphics.  Writes to SVM memory go through the LLC
 *   (Last Level Cache) and are coherent with the GPU without explicit
 *   flush/invalidate.  This means the shredder's output is immediately
 *   visible to UHD compute kernels — zero copy, zero sync.  The res1_buffer
 *   is pinned CPU memory for CUDA discrete GPU paths (if present), but on
 *   Beast Canyon the primary consumer is the UHD iGPU via SVM.
 *
 * Copyright (c) 2026 Ray Daniels
 * Licensed under the GNU Affero General Public License v3.0
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef SP_SHREDDER_CRT_H
#define SP_SHREDDER_CRT_H

#include <stdint.h>
#include <stddef.h>

#include "sp_crt.h"
#include "sp_avx512_shredder.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------
 * Constants
 * -------------------------------------------------------------------- */

/*
 * 28-bit ceiling for safe K=4096 accumulation.
 *
 * Each element after dequant+scale lands in [0, Q_MAX].  During the CRT
 * dot product, we accumulate K=4096 terms of (a_residue × b_residue) in
 * int64_t.  Worst case: K × Q_MAX² must not overflow int64_t (2^63).
 *
 *   4096 × (268328157)² ≈ 2.95 × 10^17 < 9.22 × 10^18 = 2^63.  ✓
 *
 * This gives us ~31× headroom before overflow — enough for K up to ~128K
 * if we ever need it.
 */
#define SP_SHREDDER_CRT_Q_MAX       268328157.0f

/* Q8_0 block size: 32 int8 elements + 1 fp16 scale. */
#define SP_SHREDDER_CRT_BLOCK_SIZE  32

/* --------------------------------------------------------------------
 * Configuration
 * -------------------------------------------------------------------- */

typedef struct {
    bool   use_avx512;       /* true if AVX-512 path is available          */
    int    expert_dim;       /* width of one expert row (e.g. 14336)       */
    size_t res_buffer_size;  /* total bytes allocated per residue buffer   */
} sp_shredder_crt_config_t;

/* --------------------------------------------------------------------
 * Shredder context — one per inference session, reused across experts.
 *
 * res1_buffer: M1 residues.  Pinned (cudaMallocHost) when a discrete
 *   NVIDIA GPU is present, otherwise plain aligned_alloc.  Used for
 *   CUDA async copy path on hybrid systems.
 *
 * res2_buffer: M2 residues.  Allocated via clSVMAlloc on Intel UHD
 *   integrated graphics.  Writes from the CPU are LLC-resident and
 *   coherent — the UHD compute kernels see updates without any
 *   explicit cache management.  This is the primary fast path on
 *   Beast Canyon (i9-12900 + UHD 770).
 * -------------------------------------------------------------------- */

typedef struct {
    sp_shredder_crt_config_t config;

    int32_t* res1_buffer;    /* M1 residues — pinned for CUDA             */
    int32_t* res2_buffer;    /* M2 residues — SVM for Intel UHD           */

    uint64_t total_shreds;   /* number of shred operations performed      */
    uint64_t total_elements; /* total elements shredded across all calls  */
    uint64_t total_us;       /* cumulative wall-clock microseconds        */
} sp_shredder_crt_t;

/* --------------------------------------------------------------------
 * sp_shredder_crt_init — allocate residue buffers and detect AVX-512.
 *
 *   ctx        — shredder context to initialise (caller-owned).
 *   expert_dim — row width (number of elements per expert row).
 *
 * Returns 0 on success, -1 on allocation failure.
 *
 * Allocates res1_buffer and res2_buffer sized for expert_dim elements.
 * Detects AVX-512 at runtime and sets config.use_avx512 accordingly.
 * On Intel UHD systems, res2_buffer is allocated via SVM for LLC-
 * resident coherent writes.
 * -------------------------------------------------------------------- */
int sp_shredder_crt_init(sp_shredder_crt_t* ctx, int expert_dim);

/* --------------------------------------------------------------------
 * sp_shredder_crt_free — release all buffers.
 * -------------------------------------------------------------------- */
void sp_shredder_crt_free(sp_shredder_crt_t* ctx);

/* --------------------------------------------------------------------
 * sp_shredder_crt_q8_row — the core shred operation.
 *
 * Dequantizes one row of Q8_0 blocks, scales each element to the 28-bit
 * CRT ceiling, and splits into M1/M2 residue streams.
 *
 *   ctx        — shredder context (for AVX-512 dispatch flag).
 *   q8_blocks  — pointer to Q8_0 encoded blocks for this row.
 *                Each block is 34 bytes: 2 bytes fp16 scale + 32 int8 quants.
 *   res1_row   — output: M1 residues for this row (row_width elements).
 *   res2_row   — output: M2 residues for this row (row_width elements).
 *   row_width  — number of elements in the row (must be multiple of 32).
 *
 * The fp32 intermediate exists only in AVX-512 registers — it never
 * touches the scratch buffer or memory.  The residue outputs stream
 * directly to res1_row/res2_row via non-temporal stores when possible.
 * -------------------------------------------------------------------- */
void sp_shredder_crt_q8_row(sp_shredder_crt_t* ctx,
                            const void* q8_blocks,
                            int32_t* res1_row,
                            int32_t* res2_row,
                            int row_width);

/* --------------------------------------------------------------------
 * sp_shredder_crt_q8_expert — shred an entire expert's weight matrix.
 *
 * Iterates over n_rows, calling sp_shredder_crt_q8_row for each.
 * Results are written into ctx->res1_buffer and ctx->res2_buffer
 * (row-major, contiguous).
 *
 *   ctx            — shredder context (buffers must be large enough).
 *   expert_weights — pointer to the start of Q8_0 data for this expert.
 *   n_rows         — number of rows in the expert weight matrix.
 *   row_width      — elements per row (must be multiple of 32).
 * -------------------------------------------------------------------- */
void sp_shredder_crt_q8_expert(sp_shredder_crt_t* ctx,
                               const void* expert_weights,
                               int n_rows,
                               int row_width);

/* --------------------------------------------------------------------
 * sp_shredder_crt_auto — auto-dispatch by ggml quant type.
 *
 * Currently supports GGML_TYPE_Q8_0.  Returns 0 on success, -1 if the
 * quant type is unsupported.
 *
 *   ctx         — shredder context.
 *   ggml_type   — ggml type enum value (e.g. GGML_TYPE_Q8_0 = 8).
 *   src         — pointer to quantised source data.
 *   res1        — output M1 residue buffer.
 *   res2        — output M2 residue buffer.
 *   n_elements  — total number of elements to shred.
 *
 * Future: Q4_0, Q4_1, Q5_K_M dispatch will be added here as the
 * shredder kernels are written.
 * -------------------------------------------------------------------- */
int sp_shredder_crt_auto(sp_shredder_crt_t* ctx,
                         uint32_t ggml_type,
                         const void* src,
                         int32_t* res1,
                         int32_t* res2,
                         size_t n_elements);

/* --------------------------------------------------------------------
 * sp_shredder_crt_scale_element — scalar reference for the core math.
 *
 * This is the non-AVX reference implementation used for:
 *   1. Correctness validation against the AVX-512 kernel.
 *   2. Fallback on systems without AVX-512 (e.g. Alder Lake E-cores
 *      when running in E-core-only power profile).
 *
 * The operation:
 *   1. Multiply the dequantised value (val × block_scale) by Q_MAX
 *      to fill the 28-bit range.
 *   2. Round to nearest integer (banker's rounding approximated by +0.5).
 *   3. Negative values are wrapped into the positive modular range
 *      by adding M1 (this preserves the CRT residue identity).
 *   4. Split into M1 and M2 residues via modular reduction.
 *
 * The AVX-512 kernel does exactly this but across 16 elements per
 * __m512i register, using _mm512_rem_epu32 intrinsics for the mod.
 * -------------------------------------------------------------------- */
static inline void sp_shredder_crt_scale_element(float val, float block_scale,
                                                  int32_t* r1, int32_t* r2) {
    float scaled = val * block_scale * SP_SHREDDER_CRT_Q_MAX;
    int64_t ival = (int64_t)(scaled + 0.5f);
    if (ival < 0) ival += (int64_t)SP_CRT_M1;
    *r1 = (int32_t)((uint64_t)ival % SP_CRT_M1);
    *r2 = (int32_t)((uint64_t)ival % SP_CRT_M2);
}

#ifdef __cplusplus
}
#endif

#endif /* SP_SHREDDER_CRT_H */
