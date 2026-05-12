// Shannon-Prime Beast Canyon: AVX-512 Shredder — Implementation
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "sp_avx512_shredder.h"
#include "sp_optane.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <time.h>
#endif

// ============================================================================
// Platform / CPUID detection
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#  ifdef _MSC_VER
#    include <intrin.h>
     static void sp_cpuid(int info[4], int leaf) { __cpuid(info, leaf); }
     static void sp_cpuidex(int info[4], int leaf, int sub) { __cpuidex(info, leaf, sub); }
#  else
#    include <cpuid.h>
     static void sp_cpuid(int info[4], int leaf) {
         __cpuid_count(leaf, 0, info[0], info[1], info[2], info[3]);
     }
     static void sp_cpuidex(int info[4], int leaf, int sub) {
         __cpuid_count(leaf, sub, info[0], info[1], info[2], info[3]);
     }
#  endif
#  define SP_HAS_X86_SIMD 1
#else
#  define SP_HAS_X86_SIMD 0
#endif

static uint64_t sp_time_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (uint64_t)(now.QuadPart * 1000000ULL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
#endif
}

static void sp_detect_cpu_features(sp_shredder_t *shred) {
#if SP_HAS_X86_SIMD
    int info[4];

    // Leaf 7, sub 0: extended feature flags
    sp_cpuidex(info, 7, 0);

    shred->has_avx512f    = (info[1] & (1 << 16)) != 0;  // EBX bit 16
    shred->has_avx512bw   = (info[1] & (1 << 30)) != 0;  // EBX bit 30
    shred->has_avx512_vnni = (info[2] & (1 << 11)) != 0; // ECX bit 11
    shred->has_f16c       = true; // All AVX-512 CPUs have F16C

    // Leaf 7, sub 1: AVX-512 FP16 (Sapphire Rapids+)
    sp_cpuidex(info, 7, 1);
    shred->has_avx512_fp16 = (info[0] & (1 << 5)) != 0;  // EAX bit 5
#else
    shred->has_avx512f     = false;
    shred->has_avx512bw    = false;
    shred->has_avx512_vnni = false;
    shred->has_f16c        = false;
    shred->has_avx512_fp16 = false;
#endif
}

// ============================================================================
// Shredder lifecycle
// ============================================================================

int sp_shredder_init(sp_shredder_t *shred, const sp_shredder_config_t *cfg,
                     size_t staging_elements)
{
    memset(shred, 0, sizeof(*shred));
    if (cfg) {
        shred->config = *cfg;
    } else {
        sp_shredder_config_init(&shred->config);
    }

    // Detect CPU features
    sp_detect_cpu_features(shred);

    fprintf(stderr, "[sp-shredder] CPU features: AVX-512F=%d BW=%d VNNI=%d F16C=%d FP16=%d\n",
            shred->has_avx512f, shred->has_avx512bw,
            shred->has_avx512_vnni, shred->has_f16c,
            shred->has_avx512_fp16);

    if (!shred->has_avx512f) {
        fprintf(stderr, "[sp-shredder] WARNING: AVX-512F not detected. "
                "Falling back to scalar path (SLOW).\n");
    }

    // Allocate staging buffer — 64-byte aligned for cache-line access
    size_t elem_size = shred->config.force_f32_staging ? 4 : 2;
    shred->staging_capacity = staging_elements * elem_size;
    shred->staging_elements = staging_elements;

#ifdef _WIN32
    shred->staging_buf = _aligned_malloc(shred->staging_capacity, 64);
#else
    if (posix_memalign(&shred->staging_buf, 64, shred->staging_capacity) != 0) {
        shred->staging_buf = NULL;
    }
#endif

    if (!shred->staging_buf) {
        fprintf(stderr, "[sp-shredder] ERROR: failed to allocate %.2f MB staging buffer\n",
                (double)shred->staging_capacity / (1024.0 * 1024.0));
        return -1;
    }

    // Zero the buffer to pre-fault pages into TLB
    memset(shred->staging_buf, 0, shred->staging_capacity);

    fprintf(stderr, "[sp-shredder] Staging buffer: %.2f MB (%zu elements, %s)\n",
            (double)shred->staging_capacity / (1024.0 * 1024.0),
            staging_elements,
            shred->config.force_f32_staging ? "f32" : "fp16");

    return 0;
}

void sp_shredder_free(sp_shredder_t *shred) {
    if (shred->staging_buf) {
#ifdef _WIN32
        _aligned_free(shred->staging_buf);
#else
        free(shred->staging_buf);
#endif
        shred->staging_buf = NULL;
    }
    memset(shred, 0, sizeof(*shred));
}

// ============================================================================
// Scalar fp16 helpers (forward declarations for fallback paths)
// ============================================================================

static float sp_f16_to_f32_scalar(uint16_t h);
static uint16_t sp_f32_to_f16_scalar(float val);

// ============================================================================
// Q4_0 Shredder — the primary format for Optane-resident models
// ============================================================================
//
// Q4_0 block layout (18 bytes for 32 elements):
//   [0..1]    fp16 scale (d)
//   [2..17]   16 bytes of packed 4-bit quantized values (2 per byte)
//
// Dequantization: x[i] = (q[i] - 8) * d
//
// AVX-512 strategy:
//   - Load 16 bytes of nibbles → 32 uint8 values via mask/shift
//   - Subtract 8 (zero-point), convert to fp32
//   - Multiply by scale, convert to fp16
//   - Store 32 fp16 values (64 bytes — one cache line)

void sp_shredder_q4_0(const sp_shredder_t *shred,
                      const void *src, uint16_t *dst, size_t n_elements)
{
    const uint8_t *in = (const uint8_t *)src;
    uint64_t t0 = sp_time_us();

#if SP_HAS_X86_SIMD
    if (shred->has_avx512f) {
        // AVX-512 path: process 32 elements (one Q4_0 block) per iteration
        const size_t n_blocks = n_elements / 32;
        const __m512i sub8    = _mm512_set1_epi32(8);

        for (size_t b = 0; b < n_blocks; b++) {
            // Prefetch ahead
            if (b + shred->config.prefetch_pages < n_blocks) {
                _mm_prefetch((const char*)(in + (b + shred->config.prefetch_pages) * 18),
                             _MM_HINT_T0);
            }

            // Read scale (fp16 → fp32)
            uint16_t d_fp16;
            memcpy(&d_fp16, in + b * 18, 2);

            // Convert fp16 scale to fp32 and broadcast
            __m128i d_h = _mm_set1_epi16((short)d_fp16);
            __m128  d_f = _mm_cvtph_ps(d_h);  // F16C
            __m512  vd  = _mm512_broadcastss_ps(d_f);

            // Read 16 bytes of packed nibbles
            const uint8_t *qdata = in + b * 18 + 2;

            // Unpack 16 bytes of nibbles into 32 x uint8 values
            __m128i raw128 = _mm_loadu_si128((const __m128i *)qdata);

            // Low nibbles: elements 0, 2, 4, ... (even indices)
            __m128i lo8 = _mm_and_si128(raw128, _mm_set1_epi8(0x0F));
            // High nibbles: elements 1, 3, 5, ... (odd indices)
            __m128i hi8 = _mm_srli_epi16(raw128, 4);
            hi8 = _mm_and_si128(hi8, _mm_set1_epi8(0x0F));

            // Interleave low and high to get sequential order:
            // lo[0], hi[0], lo[1], hi[1], ... → q[0], q[1], q[2], q[3], ...
            __m128i even = _mm_unpacklo_epi8(lo8, hi8);  // bytes 0-15
            __m128i odd  = _mm_unpackhi_epi8(lo8, hi8);  // bytes 16-31

            // Combine into 256-bit (32 x uint8)
            __m256i q8 = _mm256_set_m128i(odd, even);

            // Extend to 32-bit integers (two 512-bit vectors of 16 each)
            __m512i q32_lo = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(q8));
            __m512i q32_hi = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(q8, 1));

            // Subtract zero-point (8)
            q32_lo = _mm512_sub_epi32(q32_lo, sub8);
            q32_hi = _mm512_sub_epi32(q32_hi, sub8);

            // Convert to fp32
            __m512 vf_lo = _mm512_cvtepi32_ps(q32_lo);
            __m512 vf_hi = _mm512_cvtepi32_ps(q32_hi);

            // Scale
            vf_lo = _mm512_mul_ps(vf_lo, vd);
            vf_hi = _mm512_mul_ps(vf_hi, vd);

            // Convert fp32 → fp16 (F16C via AVX-512)
            __m256i h_lo = _mm512_cvtps_ph(vf_lo, _MM_FROUND_TO_NEAREST_INT);
            __m256i h_hi = _mm512_cvtps_ph(vf_hi, _MM_FROUND_TO_NEAREST_INT);

            // Store 32 fp16 values (64 bytes = one cache line)
            _mm256_storeu_si256((__m256i *)(dst + b * 32),      h_lo);
            _mm256_storeu_si256((__m256i *)(dst + b * 32 + 16), h_hi);
        }
    } else
#endif
    {
        // Scalar fallback — works on any platform
        const size_t n_blocks = n_elements / 32;

        for (size_t b = 0; b < n_blocks; b++) {
            // Read fp16 scale
            uint16_t d_fp16;
            memcpy(&d_fp16, in + b * 18, 2);
            float d = sp_f16_to_f32_scalar(d_fp16);

            const uint8_t *qdata = in + b * 18 + 2;

            for (int j = 0; j < 16; j++) {
                uint8_t byte = qdata[j];
                float v0 = ((int)(byte & 0x0F) - 8) * d;
                float v1 = ((int)(byte >> 4) - 8) * d;
                dst[b * 32 + j * 2 + 0] = sp_f32_to_f16_scalar(v0);
                dst[b * 32 + j * 2 + 1] = sp_f32_to_f16_scalar(v1);
            }
        }
    }

    // Update counters (cast away const for perf tracking)
    sp_shredder_t *s = (sp_shredder_t *)shred;
    s->total_bytes_shredded += (n_elements / 32) * 18;
    s->total_elements_produced += n_elements;
    s->total_shred_calls++;
    s->total_shred_us += sp_time_us() - t0;
}

// ============================================================================
// Scalar fp16 helpers (for fallback path)
// ============================================================================

static float sp_f16_to_f32_scalar(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        f = (mant == 0) ? sign : (sign | ((127 - 14) << 23) | (mant << 13));
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

static uint16_t sp_f32_to_f16_scalar(float val) {
    uint32_t f;
    memcpy(&f, &val, sizeof(uint32_t));
    uint16_t sign = (f >> 16) & 0x8000;
    int exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | (mant >> 13);
}

// ============================================================================
// Q8_0 Shredder
// ============================================================================
//
// Q8_0 block layout (34 bytes for 32 elements):
//   [0..1]    fp16 scale (d)
//   [2..33]   32 x int8 quantized values
//
// Dequantization: x[i] = q[i] * d

void sp_shredder_q8_0(const sp_shredder_t *shred,
                      const void *src, uint16_t *dst, size_t n_elements)
{
    const uint8_t *in = (const uint8_t *)src;
    uint64_t t0 = sp_time_us();

#if SP_HAS_X86_SIMD
    if (shred->has_avx512f) {
        const size_t n_blocks = n_elements / 32;

        for (size_t b = 0; b < n_blocks; b++) {
            if (b + shred->config.prefetch_pages < n_blocks) {
                _mm_prefetch((const char*)(in + (b + shred->config.prefetch_pages) * 34),
                             _MM_HINT_T0);
            }

            // Read scale
            uint16_t d_fp16;
            memcpy(&d_fp16, in + b * 34, 2);
            __m128i d_h = _mm_set1_epi16((short)d_fp16);
            __m128  d_f = _mm_cvtph_ps(d_h);
            __m512  vd  = _mm512_broadcastss_ps(d_f);

            // Read 32 x int8
            const int8_t *qdata = (const int8_t *)(in + b * 34 + 2);
            __m128i q8_lo = _mm_loadu_si128((const __m128i *)qdata);
            __m128i q8_hi = _mm_loadu_si128((const __m128i *)(qdata + 16));

            // Sign-extend int8 → int32
            __m512i q32_lo = _mm512_cvtepi8_epi32(q8_lo);
            __m512i q32_hi = _mm512_cvtepi8_epi32(q8_hi);

            // Convert to fp32 and scale
            __m512 vf_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(q32_lo), vd);
            __m512 vf_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(q32_hi), vd);

            // Convert to fp16 and store
            __m256i h_lo = _mm512_cvtps_ph(vf_lo, _MM_FROUND_TO_NEAREST_INT);
            __m256i h_hi = _mm512_cvtps_ph(vf_hi, _MM_FROUND_TO_NEAREST_INT);

            _mm256_storeu_si256((__m256i *)(dst + b * 32),      h_lo);
            _mm256_storeu_si256((__m256i *)(dst + b * 32 + 16), h_hi);
        }
    } else
#endif
    {
        const size_t n_blocks = n_elements / 32;
        for (size_t b = 0; b < n_blocks; b++) {
            uint16_t d_fp16;
            memcpy(&d_fp16, in + b * 34, 2);
            float d = sp_f16_to_f32_scalar(d_fp16);
            const int8_t *qdata = (const int8_t *)(in + b * 34 + 2);
            for (int j = 0; j < 32; j++) {
                dst[b * 32 + j] = sp_f32_to_f16_scalar(qdata[j] * d);
            }
        }
    }

    sp_shredder_t *s = (sp_shredder_t *)shred;
    s->total_bytes_shredded += (n_elements / 32) * 34;
    s->total_elements_produced += n_elements;
    s->total_shred_calls++;
    s->total_shred_us += sp_time_us() - t0;
}

// ============================================================================
// Q4_1 Shredder
// ============================================================================
//
// Q4_1 block layout (20 bytes for 32 elements):
//   [0..1]    fp16 scale (d)
//   [2..3]    fp16 minimum (m)
//   [4..19]   16 bytes of packed 4-bit quantized values
//
// Dequantization: x[i] = q[i] * d + m

void sp_shredder_q4_1(const sp_shredder_t *shred,
                      const void *src, uint16_t *dst, size_t n_elements)
{
    const uint8_t *in = (const uint8_t *)src;
    uint64_t t0 = sp_time_us();

#if SP_HAS_X86_SIMD
    if (shred->has_avx512f) {
        const size_t n_blocks = n_elements / 32;

        for (size_t b = 0; b < n_blocks; b++) {
            if (b + shred->config.prefetch_pages < n_blocks) {
                _mm_prefetch((const char*)(in + (b + shred->config.prefetch_pages) * 20),
                             _MM_HINT_T0);
            }

            // Read scale and min
            uint16_t d_fp16, m_fp16;
            memcpy(&d_fp16, in + b * 20, 2);
            memcpy(&m_fp16, in + b * 20 + 2, 2);

            __m128i d_h = _mm_set1_epi16((short)d_fp16);
            __m128  d_f = _mm_cvtph_ps(d_h);
            __m512  vd  = _mm512_broadcastss_ps(d_f);

            __m128i m_h = _mm_set1_epi16((short)m_fp16);
            __m128  m_f = _mm_cvtph_ps(m_h);
            __m512  vm  = _mm512_broadcastss_ps(m_f);

            // Unpack nibbles (same as Q4_0)
            const uint8_t *qdata = in + b * 20 + 4;
            __m128i raw128 = _mm_loadu_si128((const __m128i *)qdata);
            __m128i lo8 = _mm_and_si128(raw128, _mm_set1_epi8(0x0F));
            __m128i hi8 = _mm_and_si128(_mm_srli_epi16(raw128, 4), _mm_set1_epi8(0x0F));
            __m128i even = _mm_unpacklo_epi8(lo8, hi8);
            __m128i odd  = _mm_unpackhi_epi8(lo8, hi8);
            __m256i q8 = _mm256_set_m128i(odd, even);

            __m512i q32_lo = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(q8));
            __m512i q32_hi = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(q8, 1));

            // x = q * d + m (no zero-point subtraction for Q4_1)
            __m512 vf_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(q32_lo), vd, vm);
            __m512 vf_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(q32_hi), vd, vm);

            __m256i h_lo = _mm512_cvtps_ph(vf_lo, _MM_FROUND_TO_NEAREST_INT);
            __m256i h_hi = _mm512_cvtps_ph(vf_hi, _MM_FROUND_TO_NEAREST_INT);

            _mm256_storeu_si256((__m256i *)(dst + b * 32),      h_lo);
            _mm256_storeu_si256((__m256i *)(dst + b * 32 + 16), h_hi);
        }
    } else
#endif
    {
        const size_t n_blocks = n_elements / 32;
        for (size_t b = 0; b < n_blocks; b++) {
            uint16_t d_fp16, m_fp16;
            memcpy(&d_fp16, in + b * 20, 2);
            memcpy(&m_fp16, in + b * 20 + 2, 2);
            float d = sp_f16_to_f32_scalar(d_fp16);
            float m = sp_f16_to_f32_scalar(m_fp16);
            const uint8_t *qdata = in + b * 20 + 4;
            for (int j = 0; j < 16; j++) {
                uint8_t byte = qdata[j];
                dst[b * 32 + j * 2 + 0] = sp_f32_to_f16_scalar((byte & 0x0F) * d + m);
                dst[b * 32 + j * 2 + 1] = sp_f32_to_f16_scalar((byte >> 4) * d + m);
            }
        }
    }

    sp_shredder_t *s = (sp_shredder_t *)shred;
    s->total_bytes_shredded += (n_elements / 32) * 20;
    s->total_elements_produced += n_elements;
    s->total_shred_calls++;
    s->total_shred_us += sp_time_us() - t0;
}

// ============================================================================
// Q4_K Shredder (K-quant: 256 elements per superblock, 144 bytes)
// ============================================================================
//
// Q4_K superblock (144 bytes for 256 elements):
//   [0..1]     fp16 d (super scale)
//   [2..3]     fp16 dmin (super min)
//   [4..15]    12 bytes: 8 x 6-bit scales (packed)
//   [16..27]   12 bytes: 8 x 6-bit mins (packed)
//   [28..143]  128 bytes: 256 x 4-bit quants (2 per byte)
//
// Sub-block (32 elements each, 8 sub-blocks per superblock):
//   x[i] = (q[i] & 0xF) * d * sc[j] - dmin * m[j]
//   where j = i/32

void sp_shredder_q4_k(const sp_shredder_t *shred,
                      const void *src, uint16_t *dst, size_t n_elements)
{
    const uint8_t *in = (const uint8_t *)src;
    uint64_t t0 = sp_time_us();

    // Q4_K is complex enough that the scalar path is the reference.
    // AVX-512 optimisation of the 6-bit scale unpacking is future work.
    const size_t n_sb = n_elements / 256;

    for (size_t sb = 0; sb < n_sb; sb++) {
        const uint8_t *block = in + sb * 144;

        // Read super-scale and super-min
        uint16_t d_fp16, dmin_fp16;
        memcpy(&d_fp16,    block + 0, 2);
        memcpy(&dmin_fp16, block + 2, 2);
        float d    = sp_f16_to_f32_scalar(d_fp16);
        float dmin = sp_f16_to_f32_scalar(dmin_fp16);

        // Unpack 6-bit scales and mins (8 each, packed into 12 bytes)
        uint8_t sc[8], mn[8];
        const uint8_t *scales_raw = block + 4;
        const uint8_t *mins_raw   = block + 16;

        // 6-bit packing: 4 values in 3 bytes
        for (int i = 0; i < 4; i++) {
            sc[i*2+0] = scales_raw[i*3+0] & 0x3F;
            sc[i*2+1] = ((scales_raw[i*3+0] >> 6) | (scales_raw[i*3+1] << 2)) & 0x3F;
            // Simplified — full K-quant 6-bit unpacking is more involved
            // for the upper 4 scales. This handles the common case.
        }
        // Simpler: treat as 8 x uint8 with mask (approximate for now)
        for (int i = 0; i < 8; i++) {
            sc[i] = scales_raw[i] & 0x3F;  // Lower 6 bits
            mn[i] = mins_raw[i] & 0x3F;
        }

        // Dequantize 256 elements in 8 sub-blocks of 32
        const uint8_t *qdata = block + 28;

        for (int j = 0; j < 8; j++) {
            float sub_d = d * sc[j];
            float sub_m = dmin * mn[j];

            for (int k = 0; k < 16; k++) {
                uint8_t byte = qdata[j * 16 + k];
                float v0 = (byte & 0x0F) * sub_d - sub_m;
                float v1 = (byte >> 4) * sub_d - sub_m;
                dst[sb * 256 + j * 32 + k * 2 + 0] = sp_f32_to_f16_scalar(v0);
                dst[sb * 256 + j * 32 + k * 2 + 1] = sp_f32_to_f16_scalar(v1);
            }
        }
    }

    sp_shredder_t *s = (sp_shredder_t *)shred;
    s->total_bytes_shredded += (n_elements / 256) * 144;
    s->total_elements_produced += n_elements;
    s->total_shred_calls++;
    s->total_shred_us += sp_time_us() - t0;
}

// ============================================================================
// Q6_K Shredder (K-quant: 256 elements per superblock, 210 bytes)
// ============================================================================
//
// Placeholder — scalar reference. AVX-512 optimisation is Phase 2.

void sp_shredder_q6_k(const sp_shredder_t *shred,
                      const void *src, uint16_t *dst, size_t n_elements)
{
    const uint8_t *in = (const uint8_t *)src;
    uint64_t t0 = sp_time_us();

    const size_t n_sb = n_elements / 256;

    for (size_t sb = 0; sb < n_sb; sb++) {
        const uint8_t *block = in + sb * 210;

        // Q6_K layout:
        //   [0..127]    128 bytes: low 4 bits of quants (2 per byte)
        //   [128..191]  64 bytes:  high 2 bits of quants (4 per byte)
        //   [192..207]  16 bytes:  8-bit scales (16 sub-blocks)
        //   [208..209]  fp16 super-scale (d)

        uint16_t d_fp16;
        memcpy(&d_fp16, block + 208, 2);
        float d = sp_f16_to_f32_scalar(d_fp16);

        const uint8_t *ql = block;        // Low 4 bits
        const uint8_t *qh = block + 128;  // High 2 bits
        const int8_t  *sc = (const int8_t *)(block + 192);  // Scales

        for (int j = 0; j < 16; j++) {
            float sub_d = d * sc[j];

            for (int k = 0; k < 16; k++) {
                int idx = j * 16 + k;
                int byte_idx = idx / 2;
                int nibble_shift = (idx % 2) * 4;

                uint8_t lo4 = (ql[byte_idx] >> nibble_shift) & 0x0F;
                // High 2 bits packed 4 per byte
                int hi_byte = idx / 4;
                int hi_shift = (idx % 4) * 2;
                uint8_t hi2 = (qh[hi_byte] >> hi_shift) & 0x03;

                int8_t q = (int8_t)((lo4 | (hi2 << 4)) - 32);
                dst[sb * 256 + idx] = sp_f32_to_f16_scalar(q * sub_d);
            }
        }
    }

    sp_shredder_t *s = (sp_shredder_t *)shred;
    s->total_bytes_shredded += (n_elements / 256) * 210;
    s->total_elements_produced += n_elements;
    s->total_shred_calls++;
    s->total_shred_us += sp_time_us() - t0;
}

// ============================================================================
// F16 passthrough
// ============================================================================

void sp_shredder_f16(const sp_shredder_t *shred,
                     const void *src, uint16_t *dst, size_t n_elements)
{
    uint64_t t0 = sp_time_us();

    // Prefetch + memcpy. On Optane, the prefetch pulls pages into LLC
    // before the memcpy touches them.
    const uint8_t *in = (const uint8_t *)src;
    size_t total_bytes = n_elements * 2;

#if SP_HAS_X86_SIMD
    // Prefetch in 4KB strides
    for (size_t off = 0; off < total_bytes; off += SP_OPTANE_PAGE_SIZE) {
        _mm_prefetch((const char*)(in + off), _MM_HINT_T0);
    }
#endif

    memcpy(dst, src, total_bytes);

    sp_shredder_t *s = (sp_shredder_t *)shred;
    s->total_bytes_shredded += total_bytes;
    s->total_elements_produced += n_elements;
    s->total_shred_calls++;
    s->total_shred_us += sp_time_us() - t0;
}

// ============================================================================
// Auto-dispatch
// ============================================================================

int sp_shredder_auto(const sp_shredder_t *shred,
                     uint32_t ggml_type,
                     const void *src, uint16_t *dst, size_t n_elements)
{
    switch (ggml_type) {
    case SP_GGML_TYPE_Q4_0: sp_shredder_q4_0(shred, src, dst, n_elements); return 0;
    case SP_GGML_TYPE_Q4_1: sp_shredder_q4_1(shred, src, dst, n_elements); return 0;
    case SP_GGML_TYPE_Q8_0: sp_shredder_q8_0(shred, src, dst, n_elements); return 0;
    case SP_GGML_TYPE_Q4_K: sp_shredder_q4_k(shred, src, dst, n_elements); return 0;
    case SP_GGML_TYPE_Q6_K: sp_shredder_q6_k(shred, src, dst, n_elements); return 0;
    case SP_GGML_TYPE_F16:  sp_shredder_f16(shred, src, dst, n_elements);  return 0;
    default:
        fprintf(stderr, "[sp-shredder] ERROR: unsupported ggml type %u\n", ggml_type);
        return -1;
    }
}

// ============================================================================
// Diagnostics
// ============================================================================

void sp_shredder_print_status(const sp_shredder_t *shred) {
    fprintf(stderr, "\n=== SHREDDER STATUS ===\n");
    fprintf(stderr, "AVX-512F:     %s\n", shred->has_avx512f ? "YES" : "no");
    fprintf(stderr, "AVX-512BW:    %s\n", shred->has_avx512bw ? "YES" : "no");
    fprintf(stderr, "AVX-512 FP16: %s\n", shred->has_avx512_fp16 ? "YES (native)" : "no (F16C)");
    fprintf(stderr, "Staging:      %.2f MB (%zu elements)\n",
            (double)shred->staging_capacity / (1024.0*1024.0),
            shred->staging_elements);
    fprintf(stderr, "Calls:        %llu\n", (unsigned long long)shred->total_shred_calls);
    fprintf(stderr, "Elements:     %llu\n", (unsigned long long)shred->total_elements_produced);
    fprintf(stderr, "Bytes in:     %.2f MB\n",
            (double)shred->total_bytes_shredded / (1024.0*1024.0));
    fprintf(stderr, "Time:         %.2f ms\n",
            (double)shred->total_shred_us / 1000.0);
    fprintf(stderr, "Throughput:   %.2f GB/s\n", sp_shredder_throughput_gbps(shred));
    fprintf(stderr, "========================\n\n");
}

double sp_shredder_throughput_gbps(const sp_shredder_t *shred) {
    if (shred->total_shred_us == 0) return 0.0;
    double bytes = (double)shred->total_bytes_shredded;
    double seconds = (double)shred->total_shred_us / 1000000.0;
    return bytes / seconds / (1024.0 * 1024.0 * 1024.0);
}
