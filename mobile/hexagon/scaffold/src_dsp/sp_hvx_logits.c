// Shannon-Prime Engine — HVX Logit Argmax kernel.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Phase 6: Move argmax from ARM CPU to Hexagon HVX.
//
// Input: one row of the Split 4 logit tensor, stored as UFIXED_POINT_16
// (dtype 1046 — raw uint16, NOT fp16). For Qwen3-4B vocab_size = 151,936.
//
// Key property: UFIXED_16 argmax == fp32 argmax.
// The quantization encoding is: fp32 = (uint16 + offset) * scale.
// With the same scale + offset for all elements in the row, the ordering
// is monotone — the largest raw uint16 is the largest fp32. No decode needed.
//
// HVX V69: 128-byte vectors = 64 × uint16 per instruction.
// vocab_size = 151,936 → 2,374 vector loads for the max scan.
// HVX max intrinsic: Q6_Vuh_vmax_VuhVuh (unsigned halfword max, 64 lanes).
//
// Algorithm:
//   Pass 1 (HVX): vectorized max scan — find the maximum uint16 value.
//                 ~2374 Q6_Vuh_vmax_VuhVuh ops. Horizontal reduce to scalar.
//   Pass 2 (scalar): scan forward to find the first index where val == max.
//                    Expected to stop early (buffer is warm in cache from pass 1).
//
// For greedy argmax this is correct and minimal. For top-k sampling we'd need
// a partial sort; that's a later extension.

#include <stdint.h>
#include <string.h>
#include "HAP_farf.h"

#ifdef __HVX__

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

// Horizontal reduce 64 uint16 lanes in an HVX register to the scalar max.
// Uses Q6_V_vror_VR to fold the vector down in log2(64) = 6 steps.
// Each fold: rotate by N lanes (N*2 bytes for uint16), then vmax.
// After 6 folds, lane 0 holds the max across all 64 original lanes.
//
// The final result is extracted via aligned store + read of lane 0.
static uint16_t sp_hvx_hreduce_max_u16(HVX_Vector v) {
    // Step down: 32 → 16 → 8 → 4 → 2 → 1 lane pairs
    v = Q6_Vuh_vmax_VuhVuh(v, Q6_V_vror_VR(v, 32 * 2));  // fold top-32 into bottom-32
    v = Q6_Vuh_vmax_VuhVuh(v, Q6_V_vror_VR(v, 16 * 2));  // fold top-16 into bottom-16
    v = Q6_Vuh_vmax_VuhVuh(v, Q6_V_vror_VR(v, 8 * 2));
    v = Q6_Vuh_vmax_VuhVuh(v, Q6_V_vror_VR(v, 4 * 2));
    v = Q6_Vuh_vmax_VuhVuh(v, Q6_V_vror_VR(v, 2 * 2));
    v = Q6_Vuh_vmax_VuhVuh(v, Q6_V_vror_VR(v, 1 * 2));
    // Lane 0 now holds the maximum. Extract via aligned scratch.
    uint16_t lanes[64] __attribute__((aligned(128)));
    *((HVX_Vector *)lanes) = v;
    return lanes[0];
}

// HVX uint16 argmax.
//
// buf        : pointer to the uint16 logit row. Must be valid for vocab_size
//              elements. DOES NOT need to be 128-byte aligned (unaligned loads
//              are handled by the scalar prologue below).
// vocab_size : number of uint16 logit values (151,936 for Qwen3-4B).
//
// Returns the index of the maximum value (ties: first occurrence).
int sp_hvx_logit_argmax_u16(const uint16_t *buf, int vocab_size) {
    if (!buf || vocab_size <= 0) return -1;

    // Scalar prologue to reach 128-byte alignment for the HVX loop.
    // At most 63 scalar iterations to align to 64*uint16 = 128 bytes.
    int start = 0;
    const uintptr_t buf_addr = (uintptr_t)buf;
    const uintptr_t align = 128;
    const uintptr_t misalign = buf_addr & (align - 1);

    uint16_t max_val  = buf[0];
    int      max_idx  = 0;

    if (misalign != 0) {
        int skip = (int)((align - misalign) / sizeof(uint16_t));
        if (skip > vocab_size) skip = vocab_size;
        for (int i = 0; i < skip; ++i) {
            if (buf[i] > max_val) { max_val = buf[i]; max_idx = i; }
        }
        start = skip;
    }

    // HVX main loop: 64 uint16 per vector.
    // We track the current HVX max independently and reconcile at the end
    // to keep the inner loop minimal (no index bookkeeping in HVX).
    const int n_vec   = (vocab_size - start) / 64;
    const uint16_t *p = buf + start;

    if (n_vec > 0) {
        HVX_Vector vmax = *((const HVX_Vector *)p);  // first aligned chunk
        const HVX_Vector *vp = (const HVX_Vector *)(p + 64);
        for (int i = 1; i < n_vec; ++i, ++vp) {
            vmax = Q6_Vuh_vmax_VuhVuh(vmax, *vp);
        }
        uint16_t hvx_max = sp_hvx_hreduce_max_u16(vmax);
        if (hvx_max > max_val) {
            max_val = hvx_max;
            // max_idx will be updated in the epilogue scalar scan below
            // (we scan the entire HVX range to find the first occurrence).
            max_idx = -1;  // sentinel: force rescan
        }
    }

    // Scalar epilogue: tail elements beyond the last full vector.
    int tail_start = start + n_vec * 64;
    for (int i = tail_start; i < vocab_size; ++i) {
        if (buf[i] > max_val) { max_val = buf[i]; max_idx = i; }
    }

    // If the HVX loop found a new max (max_idx == -1), do a forward scan
    // over the entire buffer to find the first occurrence. The buffer is
    // warm in L1/L2 from the HVX pass so this is fast.
    if (max_idx < 0) {
        for (int i = 0; i < vocab_size; ++i) {
            if (buf[i] == max_val) { max_idx = i; break; }
        }
    }

    return max_idx;
}

#else  // !__HVX__ — scalar fallback

int sp_hvx_logit_argmax_u16(const uint16_t *buf, int vocab_size) {
    if (!buf || vocab_size <= 0) return -1;
    int best = 0;
    uint16_t vmax = buf[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (buf[i] > vmax) { vmax = buf[i]; best = i; }
    }
    return best;
}

#endif  // __HVX__
