// Shannon-Prime FP8/FP4 quantization kernels for KV cache compression.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
// Licensed under AGPLv3. Commercial license available.
//
// FP8 (E4M3FN) — IEEE-like 8-bit float: 1 sign + 4 exponent + 3 mantissa.
// Range: [-448, 448], resolution: ~0.001 at smallest scale.
// Matches torch.float8_e4m3fn / __nv_fp8_e4m3 (Ada+ tensor cores).
//
// FP4 (MXFP4) — Blackwell micro-scaling: 4-bit float with shared exponent.
// Requires sm_120+ (CUDA >= 12.8). Stub gated on SP_FP4 compile flag.
//
// These kernels slot into the banded quantization framework alongside the
// existing signed-int packing. The Möbius reorder is format-agnostic —
// only the inner quantizer changes.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

#ifdef SP_FP8
// ===== FP8 E4M3FN implementation =====
//
// E4M3FN format (no infinities, NaN = 0x7F):
//   sign(1) | exponent(4) | mantissa(3)
//   bias = 7, range: [-448, +448], subnormal min: 2^-9 = 0.001953125
//
// We use a simple scale + cast approach:
//   1. Compute per-band abs-max scale (same as int quantization)
//   2. Scale values to [-1, 1] range
//   3. Cast to fp8 via software emulation (for Turing compatibility)
//      or __nv_fp8_e4m3 intrinsic on Ada+

// Software fp8 E4M3FN encode: float -> uint8_t
__device__ __forceinline__
uint8_t fp32_to_fp8_e4m3(float x) {
    // Clamp to E4M3FN range
    const float MAX_VAL = 448.0f;
    x = fminf(fmaxf(x, -MAX_VAL), MAX_VAL);

    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));

    uint8_t sign = (bits >> 31) & 1;
    int32_t exp  = ((bits >> 23) & 0xFF) - 127;  // fp32 unbiased exponent
    uint32_t man = bits & 0x7FFFFF;               // fp32 mantissa (23 bits)

    // Special case: zero or denorm below fp8 min
    if (exp < -9) {
        return sign << 7;  // signed zero
    }

    // Subnormal fp8 range: exp in [-9, -7]
    int32_t fp8_exp = exp + 7;  // E4M3FN bias = 7
    if (fp8_exp <= 0) {
        // Subnormal: mantissa includes implicit 1
        uint32_t shift = 1 - fp8_exp;
        uint32_t fp8_man = ((1 << 23) | man) >> (20 + shift);  // round down
        return (sign << 7) | (fp8_man & 0x07);
    }

    // Normal: clamp exponent to [1, 15]
    if (fp8_exp > 15) fp8_exp = 15;

    // Round mantissa from 23 bits to 3 bits (round to nearest even)
    uint32_t fp8_man = (man + (1 << 19)) >> 20;  // round
    if (fp8_man >= 8) {
        fp8_man = 0;
        fp8_exp += 1;
        if (fp8_exp > 15) {
            fp8_exp = 15;
            fp8_man = 7;  // max normal
        }
    }

    return (sign << 7) | ((fp8_exp & 0xF) << 3) | (fp8_man & 0x07);
}

// Software fp8 E4M3FN decode: uint8_t -> float
__device__ __forceinline__
float fp8_e4m3_to_fp32(uint8_t x) {
    uint8_t sign = (x >> 7) & 1;
    uint8_t exp  = (x >> 3) & 0xF;
    uint8_t man  = x & 0x07;

    float result;
    if (exp == 0) {
        if (man == 0) {
            result = 0.0f;
        } else {
            // Subnormal
            result = ldexpf((float)man, -9);  // 2^(-6-3) = 2^-9
        }
    } else if (exp == 15 && man == 7) {
        result = 448.0f;  // NaN in E4M3FN is 0x7F, treat as max
    } else {
        // Normal
        result = ldexpf(1.0f + (float)man / 8.0f, (int)exp - 7);
    }

    return sign ? -result : result;
}

// ===== Kernels =====

// Quantize: fp32 -> fp8 with per-band scaling
// input:  [n_vecs, head_dim] fp32 (VHT2'd, reordered)
// output: [n_vecs, head_dim] uint8_t (fp8 E4M3)
// scales: [n_vecs, n_bands] fp16 (per-band abs-max, same as int path)
__global__ void kernel_fp8_quantize(
    const float* __restrict__ input,
    uint8_t*     __restrict__ output,
    half*        __restrict__ scales,
    int head_dim,
    int n_bands,
    const int* __restrict__ band_starts,   // [n_bands] band start indices
    const int* __restrict__ band_widths    // [n_bands] band widths
) {
    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;

    const float* vec_in  = input  + vec_idx * head_dim;
    uint8_t*     vec_out = output + vec_idx * head_dim;
    half*        vec_sc  = scales + vec_idx * n_bands;

    // Each thread handles one band
    if (tid < n_bands) {
        int start = band_starts[tid];
        int width = band_widths[tid];

        // Find abs-max for this band
        float amax = 0.0f;
        for (int i = start; i < start + width; i++) {
            float v = fabsf(vec_in[i]);
            if (v > amax) amax = v;
        }

        // Store scale as fp16
        float scale = (amax > 1e-12f) ? (448.0f / amax) : 1.0f;
        vec_sc[tid] = __float2half(amax / 448.0f);  // inverse scale for dequant

        // Quantize each element in the band
        for (int i = start; i < start + width; i++) {
            vec_out[i] = fp32_to_fp8_e4m3(vec_in[i] * scale);
        }
    }
}

// Dequantize: fp8 -> fp32 with per-band scaling
__global__ void kernel_fp8_dequantize(
    const uint8_t* __restrict__ input,
    float*         __restrict__ output,
    const half*    __restrict__ scales,
    int head_dim,
    int n_bands,
    const int* __restrict__ band_starts,
    const int* __restrict__ band_widths
) {
    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;

    const uint8_t* vec_in  = input  + vec_idx * head_dim;
    float*         vec_out = output + vec_idx * head_dim;
    const half*    vec_sc  = scales + vec_idx * n_bands;

    if (tid < n_bands) {
        int start = band_starts[tid];
        int width = band_widths[tid];
        float inv_scale = __half2float(vec_sc[tid]) * 448.0f;

        for (int i = start; i < start + width; i++) {
            float raw = fp8_e4m3_to_fp32(vec_in[i]);
            vec_out[i] = raw / ((inv_scale > 1e-12f) ? (448.0f / inv_scale) : 1.0f);
        }
    }
}

// ===== Host API =====

extern "C" {

void sp_fp8_band_quantize_gpu(
    const float* d_input,    // [n_vecs, head_dim] on device
    uint8_t*     d_output,   // [n_vecs, head_dim] on device
    half*        d_scales,   // [n_vecs, n_bands]  on device
    int n_vecs,
    int head_dim,
    int n_bands,
    const int* d_band_starts,
    const int* d_band_widths,
    cudaStream_t stream
) {
    if (n_vecs == 0) return;
    int threads = (n_bands < 256) ? 256 : n_bands;
    kernel_fp8_quantize<<<n_vecs, threads, 0, stream>>>(
        d_input, d_output, d_scales,
        head_dim, n_bands, d_band_starts, d_band_widths);
}

void sp_fp8_band_dequantize_gpu(
    const uint8_t* d_input,
    float*         d_output,
    const half*    d_scales,
    int n_vecs,
    int head_dim,
    int n_bands,
    const int* d_band_starts,
    const int* d_band_widths,
    cudaStream_t stream
) {
    if (n_vecs == 0) return;
    int threads = (n_bands < 256) ? 256 : n_bands;
    kernel_fp8_dequantize<<<n_vecs, threads, 0, stream>>>(
        d_input, d_output, d_scales,
        head_dim, n_bands, d_band_starts, d_band_widths);
}

}  // extern "C"

#endif  // SP_FP8


#ifdef SP_FP4
// ===== FP4 Blackwell stubs =====
//
// MXFP4 micro-scaling format: 4-bit float with shared 8-bit exponent
// per group of 32 elements. Requires sm_120+ instructions (__nv_fp4_*).
//
// These are stub declarations. Full implementation requires:
// 1. CUDA Toolkit >= 12.8
// 2. Hardware: NVIDIA Blackwell (B100, B200, GB200)
// 3. The __nv_fp4_e2m1 intrinsic types
//
// Until then, this file compiles but the functions assert(false).

#include <cassert>

extern "C" {

void sp_fp4_band_quantize_gpu(
    const float* d_input,
    uint8_t*     d_output,    // 2 fp4 values packed per byte
    half*        d_scales,    // shared exponent per group of 32
    int n_vecs,
    int head_dim,
    int n_bands,
    const int* d_band_starts,
    const int* d_band_widths,
    cudaStream_t stream
) {
    // Stub: requires Blackwell hardware + CUDA >= 12.8
    assert(false && "FP4 quantization requires Blackwell GPU (sm_120+)");
    (void)d_input; (void)d_output; (void)d_scales;
    (void)n_vecs; (void)head_dim; (void)n_bands;
    (void)d_band_starts; (void)d_band_widths; (void)stream;
}

void sp_fp4_band_dequantize_gpu(
    const uint8_t* d_input,
    float*         d_output,
    const half*    d_scales,
    int n_vecs,
    int head_dim,
    int n_bands,
    const int* d_band_starts,
    const int* d_band_widths,
    cudaStream_t stream
) {
    assert(false && "FP4 dequantization requires Blackwell GPU (sm_120+)");
    (void)d_input; (void)d_output; (void)d_scales;
    (void)n_vecs; (void)head_dim; (void)n_bands;
    (void)d_band_starts; (void)d_band_widths; (void)stream;
}

}  // extern "C"

#endif  // SP_FP4
