// Shannon-Prime VHT2 - Hexagon DSP scaffold (scalar reference kernels).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// This file is the scalar-reference DSP-side math. It compiles with
// hexagon-clang (toolv87) as plain C99 — no HVX intrinsics, no platform-
// specific headers — so we can prove the FastRPC pipeline end-to-end before
// the perf work begins.
//
// HVX REPLACEMENT PLAN:
//   sp_hex_vht2_f32           -> add sp_hex_vht2_f32_hvx using hvx::vector
//                                fp32 ops + the HVX shuffle network.
//   sp_hex_band_quantize      -> int8 SIMD: HVX mul-add + saturate-cast.
//   sp_hex_band_dequantize    -> int8 → fp32 widen/scale, vector at a time.
//
// The HVX kernels go in a separate .c (sp_hex_kernels_hvx.c) gated on
// #ifdef __HVX__ so the scalar fallback always works for unit tests.

#include "sp_hex_kernels.h"

// All four kernels delegate to the SP math core (cross-compiled into
// libsp_hex_skel.so via CMakeLists.txt). The DSP side gets bit-identical
// numerical behaviour to CPU for free, and we keep one source of truth for
// the math.
//
// HVX kernels will be added later as a sibling sp_hex_kernels_hvx.c
// gated on #ifdef __HVX__, with the imp file calling the HVX path
// when the macro is defined and falling back to the scalar core otherwise.

#include "shannon_prime.h"

// Default ship band config: 4 bands at 5/5/4/3 bits. This matches the SP
// math core's sp_config_t defaults for K (see shannon_prime.h:99). When
// the IDL grows a band-config parameter we'll pass it through; for now
// the scaffold hard-codes the K defaults.
static void sp_hex_default_band_config(sp_band_config_t *bc, int head_dim) {
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(bc, head_dim, 4, default_bits);
}

// Forward declarations — defined in sp_hex_kernels_hvx.c when HVX is
// available. Power-of-2 sizes only; the dispatcher falls back to the
// scalar reference for the multi-prime VHT2 sizes that the math core
// supports for non-pow2 head_dims.
#ifdef __HVX__
#include "qurt_hvx.h"   // qurt_hvx_lock / unlock — required around HVX kernels
void sp_hex_vht2_f32_hvx(float *data, int n);
void sp_hex_vht2_f32_hvx_qf32(float *data, int n);  // numerically correct
int  sp_hex_band_quantize_hvx(const float *coeffs, int head_dim,
                               unsigned char *out, int out_capacity);
#endif

static int sp_hex_is_pow2(int n) {
    return n > 0 && ((n & (n - 1)) == 0);
}

void sp_hex_vht2_f32(float *data, int n) {
#ifdef __HVX__
    // qf32 path: IEEE-HVX (Q6_Vsf_*) produces structurally wrong output
    // on V69 (off by 4–20 absolute, not just rounding noise). The qf32
    // intermediate path is bit-equivalent to scalar within fp32 epsilon
    // (~3.8e-06 worst at head_dim=1024 — pure rounding accumulation
    // through the qf32→sf conversion at the end of each butterfly).
    if (sp_hex_is_pow2(n) && n >= 64) {
        if (qurt_hvx_lock(QURT_HVX_MODE_128B) == 0) {
            sp_hex_vht2_f32_hvx_qf32(data, n);
            qurt_hvx_unlock();
            return;
        }
    }
#endif
    sp_vht2_forward_f32(data, n);
}

int sp_hex_band_quantize_scalar(const float *coeffs, int head_dim,
                                 unsigned char *out, int out_capacity,
                                 int *out_len) {
    sp_band_config_t bc;
    sp_hex_default_band_config(&bc, head_dim);
    if (out_capacity < bc.total_bytes) return -1;

#ifdef __HVX__
    // HVX path conditions: head_dim ≥ 128 with head_dim%128==0 (so each
    // band_sz is a multiple of 32 fp32 = one HVX vector), no ternary
    // bands. Lock HVX context per-call (thread-local; FastRPC may
    // dispatch on a different worker thread than the one that opened
    // the session). Amax now uses the int-bit-pattern trick — bypasses
    // the broken IEEE Q6_Vsf_* family, gives bit-equivalent results
    // to scalar amax.
    if (bc.ternary_band_mask == 0 &&
        head_dim >= 128 && (head_dim & 127) == 0) {
        if (qurt_hvx_lock(QURT_HVX_MODE_128B) == 0) {
            int rc = sp_hex_band_quantize_hvx(coeffs, head_dim,
                                              out, out_capacity);
            qurt_hvx_unlock();
            if (rc == 0) {
                if (out_len) *out_len = bc.total_bytes;
                return 0;
            }
        }
    }
#endif

    sp_band_quantize(coeffs, out, &bc);
    if (out_len) *out_len = bc.total_bytes;
    return 0;
}

int sp_hex_band_dequantize_scalar(const unsigned char *in, int in_len,
                                   int head_dim, int max_bands,
                                   float *out_coeffs) {
    sp_band_config_t bc;
    sp_hex_default_band_config(&bc, head_dim);
    if (in_len < bc.total_bytes) return -1;
    if (max_bands < 0 || max_bands >= bc.n_bands) {
        sp_band_dequantize(in, out_coeffs, &bc);
    } else {
        sp_band_dequantize_partial(in, out_coeffs, &bc, max_bands);
    }
    return 0;
}
