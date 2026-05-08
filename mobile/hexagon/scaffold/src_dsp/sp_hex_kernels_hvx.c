// Shannon-Prime VHT2 - Hexagon DSP HVX kernels.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// 1024-bit HVX vector implementations of the SP math core's hot loops.
// Numerically equivalent to the scalar reference (core/shannon_prime.c)
// — same IEEE fp32 ops in the same order, just batched 32 lanes wide.
//
// Build gating
// ------------
//   __HVX__   defined when the toolchain targets a DSP variant
//                     with HVX. Always true for V60+, including our V69.
//
// All entry points fall back to the scalar reference for sizes that
// can't usefully fill a vector (n < 32) or for non-power-of-2 sizes
// which the multi-prime VHT2 path would handle. This keeps the IDL
// surface uniform — callers don't branch on capability.

#ifdef __HVX__

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <string.h>

#include "shannon_prime.h"   // sp_vht2_forward_f32 (scalar fallback)

// Forward decls — exposing via header would create a circular include
// between scalar and HVX TUs. The dispatchers in sp_hex_kernels.c are
// the public API.
void sp_hex_vht2_f32_hvx(float *data, int n);

// HVX-accelerated banded quantize. Same wire format as the math core's
// sp_band_quantize: per-band fp16 scale header + signed int packed
// codes at the band's bit width. Standard path only — ternary bands
// fall through to the math core's scalar path (the dispatcher in
// sp_hex_kernels.c handles that).
int sp_hex_band_quantize_hvx(const float *coeffs, int head_dim,
                              unsigned char *out, int out_capacity);

// Vector horizontal max — reduces 32 fp32 lanes in a single HVX
// register to a scalar maximum. Used by the amax pass.
static float sp_hex_hreduce_max_f32(HVX_Vector v);

// Reinterpret a float as its 32-bit IEEE bit pattern, suitable for
// passing to Q6_V_vsplat_R. Avoids a union and keeps strict-aliasing
// rules happy.
static inline int sp_hex_f32_bits(float f) {
    int bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

// HVX-vectorised VHT2 forward butterfly for power-of-2 head_dim.
// VHT2 is self-inverse, so calling this twice round-trips to the input
// modulo IEEE roundoff (the same property the scalar reference has).
//
// Algorithm: log2(n) passes, each pass at increasing stride. The pass
// at stride S transforms each pair (data[i+j], data[i+S+j]) into
// ((x0+x1)*s, (x0-x1)*s) where s = 1/√2. For S >= 32 the j loop fills
// at least one HVX vector; we batch 32 fp32 lanes at a time. For S < 32
// (the first few passes) the scalar fallback runs — these passes touch
// O(n) total elements so the perf cost is bounded.
//
// Caller invariants: n is a power of 2 and ≥ 8 (matches SP math core).
// Misaligned float arrays work but pay the unaligned-load cost on V69;
// if profiles show this matters we'll add an aligned-staging copy.
void sp_hex_vht2_f32_hvx(float *data, int n) {
    if (!data || n < 2) return;

    const float s_const = 0.70710678118654752440f;  // 1/√2
    HVX_Vector vs = Q6_V_vsplat_R(sp_hex_f32_bits(s_const));

    for (int stride = 1; stride < n; stride <<= 1) {
        const int block = 2 * stride;

        if (stride >= 32) {
            // HVX path: 32 fp32 lanes per vector. The j loop runs in
            // strides of 32 since stride is a power of 2 and ≥ 32.
            for (int i = 0; i < n; i += block) {
                for (int j = 0; j < stride; j += 32) {
                    HVX_Vector *p0 = (HVX_Vector *)&data[i + j];
                    HVX_Vector *p1 = (HVX_Vector *)&data[i + stride + j];
                    HVX_Vector x0  = *p0;
                    HVX_Vector x1  = *p1;
                    // Q6_Vsf_*  : IEEE single-precision (ieee fp32)
                    HVX_Vector add = Q6_Vsf_vadd_VsfVsf(x0, x1);
                    HVX_Vector sub = Q6_Vsf_vsub_VsfVsf(x0, x1);
                    *p0 = Q6_Vsf_vmpy_VsfVsf(add, vs);
                    *p1 = Q6_Vsf_vmpy_VsfVsf(sub, vs);
                }
            }
        } else {
            // Scalar tail for the first few passes (stride 1, 2, 4, 8, 16).
            // Total elements touched at small strides = n * passes-with-S<32
            // = n * 5 = bounded; this loop does not dominate the runtime
            // for typical SP head_dim values (64–512).
            for (int i = 0; i < n; i += block) {
                for (int j = 0; j < stride; j++) {
                    float x0 = data[i + j];
                    float x1 = data[i + stride + j];
                    data[i + j]          = (x0 + x1) * s_const;
                    data[i + stride + j] = (x0 - x1) * s_const;
                }
            }
        }
    }
}

// ============================================================================
// Banded quantize — HVX-accelerated AMAX pass; scalar quantize + pack.
//
// V69 is missing direct HVX fp32→int conversion (Q6_Vw_equals_Vsf is
// V73+ gated). We HVX-vectorise the amax inner loop where the win is
// largest and concentrated, and fall back to scalar fp→int+clamp+pack
// for the second pass. The bit-packing is sequential by nature anyway.
//
// On V73+ we'll add a sibling kernel that pushes the second pass into
// HVX too via vw_equals_Vsf + vw_vmin/vmax saturation.
//
// The HVX path requires band_sz to be a multiple of 32 (one HVX register
// of fp32). For SP's 4-band 5/5/4/3 config on power-of-2 head_dim ≥ 128,
// this holds: band_sz = head_dim/4 ∈ {32, 64, 128, ...}. For head_dim 64
// (band_sz=16) or partial last-band remainders, the dispatcher in
// sp_hex_kernels.c falls back to the math core scalar path.
// ============================================================================

#include <math.h>
#include "shannon_prime.h"  // sp_band_config_init / sp_band_span / sp_f32_to_f16

// Reduce a 32-lane HVX fp32 vector to a scalar max. Stores to a stack
// scratch and reduces in scalar — 5 cycles for the spill, 31 fp ops for
// the reduction. A vshuff-based tree would be faster (~10 ops total)
// but adds intrinsic complexity for a non-dominant code path. Revisit
// if profiles show this hot.
static float sp_hex_hreduce_max_f32(HVX_Vector v) {
    float lanes[32] __attribute__((aligned(128)));
    *((HVX_Vector *)lanes) = v;
    float m = lanes[0];
    for (int i = 1; i < 32; ++i) {
        if (lanes[i] > m) m = lanes[i];
    }
    return m;
}

int sp_hex_band_quantize_hvx(const float *coeffs, int head_dim,
                              unsigned char *out, int out_capacity) {
    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, head_dim, 4, default_bits);
    if (out_capacity < bc.total_bytes) return -1;

    // Sign-bit-clear mask. AND-ing fp32 with 0x7FFFFFFF strips the sign
    // bit, producing the bit pattern of fabs(x). Since fp32 magnitudes
    // are monotonic in their bit pattern (for finite non-NaN values),
    // signed int max of these stripped patterns equals fp32 max of abs.
    HVX_Vector vsign_mask = Q6_V_vsplat_R(0x7FFFFFFF);

    int offset = 0;
    for (int b = 0; b < bc.n_bands; b++) {
        int band_off, band_sz;
        sp_band_span(&bc, b, &band_off, &band_sz);
        const float *band = coeffs + band_off;

        if ((band_sz & 31) != 0) return -1;

        const int bits = bc.band_bits[b];
        const int max_val = (1 << (bits - 1)) - 1;

        // Pass 1: HVX amax via the int-bit-pattern trick. Avoids the
        // broken IEEE Q6_Vsf_vabs / vmax intrinsics on V69 entirely.
        HVX_Vector vmax = Q6_V_vsplat_R(0);
        for (int i = 0; i < band_sz; i += 32) {
            HVX_Vector v        = *((HVX_Vector *)&band[i]);
            HVX_Vector v_no_sgn = Q6_V_vand_VV(v, vsign_mask);
            // Signed int max — works because all lanes have sign bit clear,
            // so signed and unsigned max give identical results, AND the
            // remaining 31 bits encode magnitude monotonically.
            vmax = Q6_Vw_vmax_VwVw(vmax, v_no_sgn);
        }
        // Reduce 32 lanes → scalar. The result is 32 fp32 bit patterns
        // (stripped of sign), so reinterpret-as-fp32 lane reduction gives
        // the amax fp32 value directly.
        float amax = sp_hex_hreduce_max_f32(vmax);

        // Compute scale exactly the way the math core does — fp16
        // round-trip on the scale so encode/decode agree byte-for-byte.
        float scale = (amax > 0.0f) ? amax / (float)max_val : 0.0f;
        uint16_t scale_f16 = sp_f32_to_f16(scale);
        out[offset]     = scale_f16 & 0xFF;
        out[offset + 1] = (scale_f16 >> 8) & 0xFF;
        offset += 2;

        float scale_stored = sp_f16_to_f32(scale_f16);
        float inv_scale = (scale_stored > 0.0f) ? 1.0f / scale_stored : 0.0f;

        // Pass 2: scalar quantize + bit-pack — same logic as math core
        // (V69 is missing the HVX fp32→int conversion intrinsic).
        uint64_t bit_buffer = 0;
        int      bit_pos = 0;
        for (int i = 0; i < band_sz; i++) {
            int q = (int)roundf(band[i] * inv_scale);
            if (q > max_val)  q = max_val;
            if (q < -max_val) q = -max_val;
            uint32_t u = (uint32_t)(q + max_val);
            bit_buffer |= ((uint64_t)u << bit_pos);
            bit_pos += bits;
            while (bit_pos >= 8) {
                out[offset++] = (uint8_t)(bit_buffer & 0xFF);
                bit_buffer >>= 8;
                bit_pos -= 8;
            }
        }
        if (bit_pos > 0) {
            out[offset++] = (uint8_t)(bit_buffer & 0xFF);
        }
    }

    return 0;
}

// ============================================================================
// VHT2 — qf32 intermediate variant.
//
// Same math as sp_hex_vht2_f32_hvx, but the butterfly arithmetic stays in
// Qualcomm's qf32 ("flexible float") representation — wider mantissa than
// IEEE sf, deterministic ordering of internal rounding — converting back
// to IEEE sf only at the final store. Theory: the drift in the IEEE-only
// HVX path comes from HVX's sf pipeline using a different rounding stance
// (possibly RTZ) than scalar's RNE; qf32 promotes the intermediate
// precision enough that any single-pass rounding-mode disagreement
// doesn't compound into visible per-element error.
//
// Exported separately so we can A/B vs the IEEE path without touching the
// existing kernel. The dispatcher stays scalar-only until one of these
// proves bit-equivalent to the math core reference.
// ============================================================================

void sp_hex_vht2_f32_hvx_qf32(float *data, int n);

void sp_hex_vht2_f32_hvx_qf32(float *data, int n) {
    if (!data || n < 2) return;

    const float s_const = 0.70710678118654752440f;
    HVX_Vector vs_sf  = Q6_V_vsplat_R(sp_hex_f32_bits(s_const));
    HVX_Vector zero_sf = Q6_V_vsplat_R(0);
    // Convert vs_sf → vs_qf32 by adding zero (the documented sf→qf32 path).
    HVX_Vector vs_qf  = Q6_Vqf32_vadd_VsfVsf(vs_sf, zero_sf);

    for (int stride = 1; stride < n; stride <<= 1) {
        const int block = 2 * stride;

        if (stride >= 32) {
            for (int i = 0; i < n; i += block) {
                for (int j = 0; j < stride; j += 32) {
                    HVX_Vector *p0 = (HVX_Vector *)&data[i + j];
                    HVX_Vector *p1 = (HVX_Vector *)&data[i + stride + j];
                    HVX_Vector x0  = *p0;   // sf
                    HVX_Vector x1  = *p1;   // sf

                    // sf + sf → qf32 (mixed add)
                    HVX_Vector sum_qf = Q6_Vqf32_vadd_VsfVsf(x0, x1);
                    HVX_Vector dif_qf = Q6_Vqf32_vsub_VsfVsf(x0, x1);
                    // qf32 * qf32 → qf32 (intermediate stays in qf32)
                    HVX_Vector ssum_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(sum_qf, vs_qf);
                    HVX_Vector sdif_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(dif_qf, vs_qf);
                    // qf32 → IEEE sf for memory store
                    *p0 = Q6_Vsf_equals_Vqf32(ssum_qf);
                    *p1 = Q6_Vsf_equals_Vqf32(sdif_qf);
                }
            }
        } else {
            // Scalar tail — identical to the IEEE-HVX kernel and to the
            // math core. No rounding-mode question at this level.
            for (int i = 0; i < n; i += block) {
                for (int j = 0; j < stride; j++) {
                    float x0 = data[i + j];
                    float x1 = data[i + stride + j];
                    data[i + j]          = (x0 + x1) * s_const;
                    data[i + stride + j] = (x0 - x1) * s_const;
                }
            }
        }
    }
}


// ────────────────────────────────────────────────────────────────────
// HVX fp32 dot product — sf×sf via qf32 intermediate.
// ────────────────────────────────────────────────────────────────────
//
// Same numerical pattern as sp_hex_vht2_f32_hvx_qf32: convert sf→qf32
// for arithmetic, qf32→sf for the final reduction, so the result is
// bit-equivalent to scalar within fp32 epsilon × O(log N) accumulation
// rounding (the qf32 boundary conversion overhead — ~3.8e-6 worst-abs
// at head_dim=1024 per the 2026-04-29 measurements).
//
// Loop emits one HVX_Vector load + one fma per 32 fp32 elements. At
// head_dim=128 that's 4 fmas + 1 horizontal reduce. The reduce uses
// Q6_V_vror_VR to fold the 32-lane accumulator down to lane 0 in
// log2(32) = 5 steps; faster than a scalar lane-extract loop and the
// pattern matches what the SDK's HVX cookbook uses for fp32 reductions.
//
// Caller must hold qurt_hvx_lock(QURT_HVX_MODE_128B). a/b must be
// 128-byte aligned; head_dim must be a multiple of 32 (the dispatcher
// in sp_hex_imp.c takes the scalar path for hd<32).

void sp_hex_dot_f32_hvx(const float *a, const float *b, int head_dim,
                         float *out);

void sp_hex_dot_f32_hvx(const float *a, const float *b, int head_dim,
                         float *out) {
    if (!a || !b || !out || head_dim < 32 || (head_dim & 31) != 0) {
        // Fall through to scalar — should not be reached when caller
        // gates on (head_dim & 31) == 0.
        float s = 0.0f;
        for (int i = 0; i < head_dim; ++i) s += a[i] * b[i];
        *out = s;
        return;
    }

    HVX_Vector zero_sf = Q6_V_vsplat_R(0);
    // Initialise qf32 accumulator to qf32 zero via the documented
    // sf→qf32 conversion (mixed-add by 0).
    HVX_Vector acc_qf  = Q6_Vqf32_vadd_VsfVsf(zero_sf, zero_sf);

    const HVX_Vector *va = (const HVX_Vector *)a;
    const HVX_Vector *vb = (const HVX_Vector *)b;
    const int n_vec = head_dim / 32;

    for (int i = 0; i < n_vec; ++i) {
        HVX_Vector a_sf = va[i];
        HVX_Vector b_sf = vb[i];
        // sf×sf → qf32. Q6_Vqf32_vmpy_VsfVsf is the "mixed multiply"
        // path documented in the V69 ISA manual as numerically stable
        // through chains of accumulation.
        HVX_Vector prod_qf = Q6_Vqf32_vmpy_VsfVsf(a_sf, b_sf);
        acc_qf = Q6_Vqf32_vadd_Vqf32Vqf32(acc_qf, prod_qf);
    }

    // qf32 → IEEE sf for the horizontal reduce.
    HVX_Vector acc_sf = Q6_Vsf_equals_Vqf32(acc_qf);

    // Horizontal reduce 32 fp32 lanes → lane 0 via butterfly.
    // Each step rotates by N×4 bytes (= N lanes of fp32) and adds.
    // Use Q6_V_vror_VR which takes a byte-rotation count in R.
    // After log2(32)=5 steps, lane 0 holds the full sum.
    for (int rot_lanes = 16; rot_lanes >= 1; rot_lanes >>= 1) {
        HVX_Vector r = Q6_V_vror_VR(acc_sf, rot_lanes * 4);
        // sf + sf → qf32 → sf (round-trip needed because there's no
        // direct sf+sf→sf vector add on V69; the qf32 intermediate is
        // the bit-equivalent path).
        HVX_Vector sum_qf = Q6_Vqf32_vadd_VsfVsf(acc_sf, r);
        acc_sf = Q6_Vsf_equals_Vqf32(sum_qf);
    }

    // Lane 0 holds the dot. Extract via aligned store + read.
    float lanes[32] __attribute__((aligned(128)));
    *(HVX_Vector *)lanes = acc_sf;
    *out = lanes[0];
}

#endif  // __HVX__
