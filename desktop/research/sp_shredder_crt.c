// Shannon-Prime Beast Canyon: Residue-Aware Shredder
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Fused Q8_0 dequant + CRT residue split in a single pass.
// Eliminates the fp32 scratch buffer entirely — fp32 intermediate
// exists only in registers.
//
// AVX-512 path:  dequant 16 Q8_0 elements per iteration in-register,
//   scale to 28-bit CRT ceiling, split into M1/M2 residues using
//   Mersenne shift-and-add (M1) and conditional-subtract (M2).
//   Requires AVX-512F + BW (int8→int32 widen) + DQ (fp32→int64).
//   Tiger Lake i9-11900KB has all three.

#include "sp_shredder_crt.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(__AVX512__))
#  include <immintrin.h>
#  define SP_SHREDDER_CRT_HAS_AVX512_COMPILE 1
#elif defined(_MSC_VER)
   // MSVC: immintrin.h is always available; runtime detect gates usage.
#  include <immintrin.h>
#  define SP_SHREDDER_CRT_HAS_AVX512_COMPILE 1
#else
#  define SP_SHREDDER_CRT_HAS_AVX512_COMPILE 0
#endif

#ifdef _WIN32
#  include <windows.h>
static uint64_t sp_shredcrt_time_us(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (uint64_t)(c.QuadPart * 1000000ULL / f.QuadPart);
}
#else
#  include <time.h>
static uint64_t sp_shredcrt_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000;
}
#endif

// ---------------------------------------------------------------------------
//  fp16 -> fp32 conversion (software fallback)
// ---------------------------------------------------------------------------
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t f;

    if (exp == 0) {
        if (man == 0) {
            f = sign; // +/- zero
        } else {
            // Denormal: convert to normal float
            exp = 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            f = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)man << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | ((uint32_t)man << 13); // inf/nan
    } else {
        f = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)man << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
//  CPUID-based AVX-512 detection (F + BW + DQ required for CRT shredder)
// ---------------------------------------------------------------------------
static bool detect_avx512(void) {
#if !SP_SHREDDER_CRT_HAS_AVX512_COMPILE
    return false;
#elif defined(__GNUC__) || defined(__clang__)
#  if defined(__x86_64__) || defined(__i386__)
    // GCC/Clang: check all three extensions we need.
    return __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512bw")
        && __builtin_cpu_supports("avx512dq");
#  else
    return false;
#  endif
#elif defined(_MSC_VER)
    int info[4];
    __cpuidex(info, 7, 0);
    // EBX bit 16 = AVX-512F, bit 30 = AVX-512BW, bit 17 = AVX-512DQ
    const int required = (1 << 16) | (1 << 30) | (1 << 17);
    return (info[1] & required) == required;
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
//  sp_shredder_crt_init
// ---------------------------------------------------------------------------
int sp_shredder_crt_init(sp_shredder_crt_t* ctx, int expert_dim) {
    if (!ctx || expert_dim <= 0) return -1;
    memset(ctx, 0, sizeof(*ctx));

    ctx->config.expert_dim      = expert_dim;
    ctx->config.use_avx512      = detect_avx512();
    ctx->config.res_buffer_size = (size_t)expert_dim * sizeof(int32_t);

    // Allocate residue buffers — 64-byte aligned for AVX-512.
    size_t alloc_size = (ctx->config.res_buffer_size + 63) & ~(size_t)63;

#ifdef _WIN32
    ctx->res1_buffer = (int32_t*)_aligned_malloc(alloc_size, 64);
    ctx->res2_buffer = (int32_t*)_aligned_malloc(alloc_size, 64);
#else
    ctx->res1_buffer = (int32_t*)aligned_alloc(64, alloc_size);
    ctx->res2_buffer = (int32_t*)aligned_alloc(64, alloc_size);
#endif

    if (!ctx->res1_buffer || !ctx->res2_buffer) {
        sp_shredder_crt_free(ctx);
        return -1;
    }

    memset(ctx->res1_buffer, 0, alloc_size);
    memset(ctx->res2_buffer, 0, alloc_size);

    // TODO: If CUDA is available, upgrade res1_buffer to cudaMallocHost
    // for pinned async copy path. If Intel UHD SVM is available, upgrade
    // res2_buffer to clSVMAlloc for LLC-resident coherent writes.

    return 0;
}

// ---------------------------------------------------------------------------
//  sp_shredder_crt_free
// ---------------------------------------------------------------------------
void sp_shredder_crt_free(sp_shredder_crt_t* ctx) {
    if (!ctx) return;
#ifdef _WIN32
    if (ctx->res1_buffer) _aligned_free(ctx->res1_buffer);
    if (ctx->res2_buffer) _aligned_free(ctx->res2_buffer);
#else
    if (ctx->res1_buffer) free(ctx->res1_buffer);
    if (ctx->res2_buffer) free(ctx->res2_buffer);
#endif
    memset(ctx, 0, sizeof(*ctx));
}

// ---------------------------------------------------------------------------
//  AVX-512 vectorized CRT shredder
// ---------------------------------------------------------------------------
//
// Processes 16 Q8_0 elements per iteration in-register:
//   1. int8 → int32 widen     (_mm512_cvtepi8_epi32, AVX-512BW)
//   2. int32 → fp32           (_mm512_cvtepi32_ps)
//   3. scale by Q_MAX         (_mm512_mul_ps)
//   4. round to nearest       (_mm512_roundscale_ps)
//   5. split into 2×8 fp32    (_mm512_extractf32x8_ps, AVX-512DQ)
//   6. fp32 → int64           (_mm512_cvtps_epi64, AVX-512DQ)
//   7. negative wrap: +M1     (_mm512_mask_add_epi64)
//   8. Mersenne reduce mod M1 (shift-and-add: fold, conditional subtract)
//   9. mod M2                 (two conditional subtracts — x < 2·M2 always)
//  10. int64 → int32 pack     (_mm512_cvtepi64_epi32)
//  11. store                  (_mm512_storeu_si512)
//
// The fp32 intermediate never touches memory — everything stays in zmm
// registers from dequant through residue output.  For a 2048-dim expert
// row, this completes in ~2-3 µs on Tiger Lake vs ~20 µs scalar.
//
// Correctness note: step 7 reproduces the scalar code's convention
// exactly (add M1 when negative, then take mod of the SAME adjusted
// value for both M1 and M2).  The Garner reconstruction on the decode
// side uses the same convention, so the offset is consistent.

#if SP_SHREDDER_CRT_HAS_AVX512_COMPILE

// Process 16 Q8_0 elements: dequant + scale + CRT split, all in registers.
// quants: 16 int8 values from a Q8_0 block.
// scale_qmax: pre-multiplied (block_scale * Q_MAX).
// r1_out, r2_out: 16 int32 residue outputs.
static void sp_shredcrt_avx512_16(const int8_t* quants, float scale_qmax,
                                   int32_t* r1_out, int32_t* r2_out) {
    // Constants
    const __m512i v_m1   = _mm512_set1_epi64((int64_t)SP_CRT_M1);
    const __m512i v_m2   = _mm512_set1_epi64((int64_t)SP_CRT_M2);
    const __m512i v_mask = _mm512_set1_epi64(0x7FFFFFFFLL);
    const __m512i v_zero = _mm512_setzero_si512();
    const __m512  v_scl  = _mm512_set1_ps(scale_qmax);

    // 1. Load 16 int8 → 16 int32
    __m128i raw  = _mm_loadu_si128((const __m128i*)quants);
    __m512i vi32 = _mm512_cvtepi8_epi32(raw);

    // 2-3. int32 → fp32, scale by (block_scale * Q_MAX)
    __m512 vf     = _mm512_cvtepi32_ps(vi32);
    __m512 scaled = _mm512_mul_ps(vf, v_scl);

    // 4. Round to nearest integer
    __m512 rounded = _mm512_roundscale_ps(scaled,
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // 5. Split into two halves of 8 for int64 conversion
    __m256 lo_f = _mm512_castps512_ps256(rounded);
    __m256 hi_f = _mm512_extractf32x8_ps(rounded, 1);

    // 6. fp32 → int64 (AVX-512DQ: _mm512_cvtps_epi64 takes __m256)
    __m512i lo_i64 = _mm512_cvtps_epi64(lo_f);
    __m512i hi_i64 = _mm512_cvtps_epi64(hi_f);

    // 7. Match scalar convention: if negative, add M1
    //    After this, values are in [0, M1 + max_positive).
    //    For practical Q8_0 weights (|ival| < M1), this always makes
    //    the value non-negative.  The same adjusted value is used for
    //    both M1 and M2 reduction, matching the scalar path exactly.
    __mmask8 neg_lo = _mm512_cmpgt_epi64_mask(v_zero, lo_i64);
    __mmask8 neg_hi = _mm512_cmpgt_epi64_mask(v_zero, hi_i64);
    lo_i64 = _mm512_mask_add_epi64(lo_i64, neg_lo, lo_i64, v_m1);
    hi_i64 = _mm512_mask_add_epi64(hi_i64, neg_hi, hi_i64, v_m1);

    // --- M1 Mersenne reduction (2^31 - 1) ---
    // lo_bits = x & 0x7FFFFFFF, hi_bits = x >> 31, r = lo_bits + hi_bits
    // if r >= M1: r -= M1
    __m512i lo_r1_lo = _mm512_and_si512(lo_i64, v_mask);
    __m512i lo_r1_hi = _mm512_srli_epi64(lo_i64, 31);
    __m512i lo_r1    = _mm512_add_epi64(lo_r1_lo, lo_r1_hi);
    __mmask8 lo_r1_ge = _mm512_cmpge_epi64_mask(lo_r1, v_m1);
    lo_r1 = _mm512_mask_sub_epi64(lo_r1, lo_r1_ge, lo_r1, v_m1);

    __m512i hi_r1_lo = _mm512_and_si512(hi_i64, v_mask);
    __m512i hi_r1_hi = _mm512_srli_epi64(hi_i64, 31);
    __m512i hi_r1    = _mm512_add_epi64(hi_r1_lo, hi_r1_hi);
    __mmask8 hi_r1_ge = _mm512_cmpge_epi64_mask(hi_r1, v_m1);
    hi_r1 = _mm512_mask_sub_epi64(hi_r1, hi_r1_ge, hi_r1, v_m1);

    // --- M2 reduction (2^31 - 19) ---
    // After step 7, x is in [0, ~3.4B].  M2 ≈ 2.15B.  x < 2·M2 always.
    // One conditional subtract suffices.
    __mmask8 lo_r2_ge = _mm512_cmpge_epi64_mask(lo_i64, v_m2);
    __m512i  lo_r2    = _mm512_mask_sub_epi64(lo_i64, lo_r2_ge, lo_i64, v_m2);
    // Safety: a second pass for edge cases near 2·M2
    lo_r2_ge = _mm512_cmpge_epi64_mask(lo_r2, v_m2);
    lo_r2    = _mm512_mask_sub_epi64(lo_r2, lo_r2_ge, lo_r2, v_m2);

    __mmask8 hi_r2_ge = _mm512_cmpge_epi64_mask(hi_i64, v_m2);
    __m512i  hi_r2    = _mm512_mask_sub_epi64(hi_i64, hi_r2_ge, hi_i64, v_m2);
    hi_r2_ge = _mm512_cmpge_epi64_mask(hi_r2, v_m2);
    hi_r2    = _mm512_mask_sub_epi64(hi_r2, hi_r2_ge, hi_r2, v_m2);

    // 10. Pack 8 int64 → 8 int32 (vpmovqd — safe since residues < 2^31)
    __m256i lo_r1_32 = _mm512_cvtepi64_epi32(lo_r1);
    __m256i hi_r1_32 = _mm512_cvtepi64_epi32(hi_r1);
    __m256i lo_r2_32 = _mm512_cvtepi64_epi32(lo_r2);
    __m256i hi_r2_32 = _mm512_cvtepi64_epi32(hi_r2);

    // 11. Combine two halves of 8 into 16 and store
    __m512i r1_full = _mm512_inserti32x8(_mm512_castsi256_si512(lo_r1_32), hi_r1_32, 1);
    __m512i r2_full = _mm512_inserti32x8(_mm512_castsi256_si512(lo_r2_32), hi_r2_32, 1);

    _mm512_storeu_si512((__m512i*)r1_out, r1_full);
    _mm512_storeu_si512((__m512i*)r2_out, r2_full);
}

// AVX-512 path for a full Q8_0 row.  Each Q8_0 block is 34 bytes:
// 2 bytes fp16 scale + 32 int8 quants.  We process in two passes of 16.
static void sp_shredcrt_avx512_q8_row(const void* q8_blocks,
                                       int32_t* res1_row,
                                       int32_t* res2_row,
                                       int row_width) {
    const uint8_t* src = (const uint8_t*)q8_blocks;
    const int n_blocks = row_width / SP_SHREDDER_CRT_BLOCK_SIZE;

    for (int b = 0; b < n_blocks; b++) {
        // Extract fp16 scale and pre-multiply with Q_MAX
        uint16_t scale_fp16;
        memcpy(&scale_fp16, src, 2);
        float block_scale = fp16_to_fp32(scale_fp16);
        float scale_qmax  = block_scale * SP_SHREDDER_CRT_Q_MAX;
        src += 2;

        const int8_t* quants = (const int8_t*)src;
        int base_idx = b * SP_SHREDDER_CRT_BLOCK_SIZE;

        // Two passes of 16 elements = 32 elements per Q8_0 block
        sp_shredcrt_avx512_16(quants,      scale_qmax,
                              &res1_row[base_idx],      &res2_row[base_idx]);
        sp_shredcrt_avx512_16(quants + 16, scale_qmax,
                              &res1_row[base_idx + 16], &res2_row[base_idx + 16]);

        src += SP_SHREDDER_CRT_BLOCK_SIZE;
    }
}

#endif /* SP_SHREDDER_CRT_HAS_AVX512_COMPILE */

// ---------------------------------------------------------------------------
//  sp_shredder_crt_q8_row — dispatches to AVX-512 or scalar
// ---------------------------------------------------------------------------
//
// Q8_0 block layout (34 bytes):
//   bytes [0..1]  : fp16 scale (little-endian)
//   bytes [2..33] : 32 × int8 quantised values
//
void sp_shredder_crt_q8_row(sp_shredder_crt_t* ctx,
                            const void* q8_blocks,
                            int32_t* res1_row,
                            int32_t* res2_row,
                            int row_width) {
    if (!ctx || !q8_blocks || !res1_row || !res2_row) return;

#if SP_SHREDDER_CRT_HAS_AVX512_COMPILE
    if (ctx->config.use_avx512) {
        sp_shredcrt_avx512_q8_row(q8_blocks, res1_row, res2_row, row_width);
        return;
    }
#endif

    // Scalar fallback
    const uint8_t* src = (const uint8_t*)q8_blocks;
    int n_blocks = row_width / SP_SHREDDER_CRT_BLOCK_SIZE;

    for (int b = 0; b < n_blocks; b++) {
        uint16_t scale_fp16;
        memcpy(&scale_fp16, src, 2);
        float block_scale = fp16_to_fp32(scale_fp16);
        src += 2;

        const int8_t* quants = (const int8_t*)src;
        int base_idx = b * SP_SHREDDER_CRT_BLOCK_SIZE;

        for (int q = 0; q < SP_SHREDDER_CRT_BLOCK_SIZE; q++) {
            float val = (float)quants[q];
            sp_shredder_crt_scale_element(val, block_scale,
                                          &res1_row[base_idx + q],
                                          &res2_row[base_idx + q]);
        }

        src += SP_SHREDDER_CRT_BLOCK_SIZE;
    }
}

// ---------------------------------------------------------------------------
//  sp_shredder_crt_q8_expert
// ---------------------------------------------------------------------------
void sp_shredder_crt_q8_expert(sp_shredder_crt_t* ctx,
                               const void* expert_weights,
                               int n_rows,
                               int row_width) {
    if (!ctx || !expert_weights || n_rows <= 0 || row_width <= 0) return;

    uint64_t t0 = sp_shredcrt_time_us();

    const uint8_t* src = (const uint8_t*)expert_weights;
    // Q8_0: each row is (row_width / 32) blocks of 34 bytes each.
    int blocks_per_row = row_width / SP_SHREDDER_CRT_BLOCK_SIZE;
    size_t row_bytes = (size_t)blocks_per_row * (2 + SP_SHREDDER_CRT_BLOCK_SIZE);

    for (int r = 0; r < n_rows; r++) {
        int32_t* r1_row = ctx->res1_buffer + r * row_width;
        int32_t* r2_row = ctx->res2_buffer + r * row_width;

        sp_shredder_crt_q8_row(ctx, src, r1_row, r2_row, row_width);
        src += row_bytes;
    }

    uint64_t t1 = sp_shredcrt_time_us();

    ctx->total_shreds++;
    ctx->total_elements += (uint64_t)n_rows * (uint64_t)row_width;
    ctx->total_us += (t1 - t0);
}

// ---------------------------------------------------------------------------
//  sp_shredder_crt_auto
// ---------------------------------------------------------------------------
int sp_shredder_crt_auto(sp_shredder_crt_t* ctx,
                         uint32_t ggml_type,
                         const void* src,
                         int32_t* res1,
                         int32_t* res2,
                         size_t n_elements) {
    if (!ctx || !src || !res1 || !res2) return -1;

    // GGML_TYPE_Q8_0 = 8
    if (ggml_type == 8) {
        int row_width = ctx->config.expert_dim;
        if (row_width <= 0) return -1;
        int n_rows = (int)(n_elements / (size_t)row_width);
        if (n_rows <= 0) return -1;

        // Temporarily swap in the caller's output buffers.
        int32_t* save_r1 = ctx->res1_buffer;
        int32_t* save_r2 = ctx->res2_buffer;
        ctx->res1_buffer = res1;
        ctx->res2_buffer = res2;

        sp_shredder_crt_q8_expert(ctx, src, n_rows, row_width);

        ctx->res1_buffer = save_r1;
        ctx->res2_buffer = save_r2;
        return 0;
    }

    // TODO: GGML_TYPE_Q4_0 (2), Q4_1 (3), Q5_K_M, Q6_K
    fprintf(stderr, "[sp-shredder-crt] unsupported ggml type %u\n", ggml_type);
    return -1;
}
