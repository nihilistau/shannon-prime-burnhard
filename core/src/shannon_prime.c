// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#include "shannon_prime.h"
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>

// ============================================================================
// Utilities
// ============================================================================

static inline int sp_is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}


// ============================================================================
// Config
// ============================================================================

void sp_config_init(sp_config_t *cfg, int head_dim, int n_layers, int n_heads_kv) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->head_dim    = head_dim;
    cfg->n_layers    = n_layers;
    cfg->n_heads_kv  = n_heads_kv;

    // Ship-safe K band allocation: 5/5/4/3
    cfg->k_n_bands = 4;
    cfg->k_band_bits[0] = 5;
    cfg->k_band_bits[1] = 5;
    cfg->k_band_bits[2] = 4;
    cfg->k_band_bits[3] = 3;

    // V: flat int3 (no banding — flat beats banded for V, no exceptions)
    cfg->v_n_bands = 1;
    cfg->v_band_bits[0] = 3;

    // Möbius mask on by default (free quality win)
    cfg->use_mobius_mask = true;
    cfg->skeleton_k = head_dim; // Must equal head_dim
    cfg->skeleton_v = head_dim;

    // Vilenkin off by default (research path)
    cfg->use_vilenkin    = false;
    cfg->vilenkin_primes = 2;
    cfg->energy_threshold = 0.95f;
}

// ============================================================================
// VHT2 — Vilenkin-Hartley Transform (the single transform)
// ============================================================================
//
// Staged application of p × p Hartley kernels cas(2πkj/p)/√p for each prime
// factor p of the vector length n (factors in {2,3,5,7,11}). Each stage is
// orthonormal, so the whole transform is self-inverse: VHT2(VHT2(x)) = x.
//
// At n = 2^k this collapses to the Walsh-Hadamard butterfly scaled by 1/√2
// per stage — the spectral structure is that of the p=2 Hartley butterfly but
// there is no 1/N to undo on the inverse.

#define SP_VHT2_MAX_P 11

static const int _sp_vht2_primes[] = {2, 3, 5, 7, 11};
static const int _sp_vht2_n_primes = 5;

// Hartley kernels for each supported prime, lazily initialised on first use.
static float _sp_H2[2][2];
static float _sp_H3[3][3];
static float _sp_H5[5][5];
static float _sp_H7[7][7];
static float _sp_H11[11][11];
static int   _sp_vht2_initialised = 0;

static void _sp_vht2_init_kernels(void) {
    if (_sp_vht2_initialised) return;
    float *kernels[5] = {
        &_sp_H2[0][0], &_sp_H3[0][0], &_sp_H5[0][0],
        &_sp_H7[0][0], &_sp_H11[0][0]
    };
    for (int pi = 0; pi < _sp_vht2_n_primes; pi++) {
        int p = _sp_vht2_primes[pi];
        float *H = kernels[pi];
        double inv_sqrt_p = 1.0 / sqrt((double)p);
        for (int k = 0; k < p; k++) {
            for (int j = 0; j < p; j++) {
                double angle = 2.0 * M_PI * (double)k * (double)j / (double)p;
                H[k * p + j] = (float)((cos(angle) + sin(angle)) * inv_sqrt_p);
            }
        }
    }
    _sp_vht2_initialised = 1;
}

static const float *_sp_hartley_kernel(int p) {
    switch (p) {
        case 2:  return &_sp_H2[0][0];
        case 3:  return &_sp_H3[0][0];
        case 5:  return &_sp_H5[0][0];
        case 7:  return &_sp_H7[0][0];
        case 11: return &_sp_H11[0][0];
        default: return NULL;
    }
}

void sp_vht2_forward_f32(float *data, int n) {
    if (n <= 0) return;
    _sp_vht2_init_kernels();

    int stride = 1;
    int residue = n;
    for (int pi = 0; pi < _sp_vht2_n_primes; pi++) {
        int p = _sp_vht2_primes[pi];
        while (residue % p == 0) {
            const int block = p * stride;
            if (p == 2) {
                // Specialised Walsh-Hadamard butterfly. H2 is {{s,s},{s,-s}}
                // with s = 1/√2, so the generic matmul collapses to
                // (x0+x1, x0-x1)·s. Compilers vectorise the j loop cleanly
                // once the k/m reductions are elided — this is by far the
                // dominant stage for power-of-2 head_dims.
                const float s = 0.70710678118654752440f;
                for (int i = 0; i < n; i += block) {
                    for (int j = 0; j < stride; j++) {
                        const float x0 = data[i + j];
                        const float x1 = data[i + stride + j];
                        data[i + j]          = (x0 + x1) * s;
                        data[i + stride + j] = (x0 - x1) * s;
                    }
                }
            } else {
                const float *H = _sp_hartley_kernel(p);
                for (int i = 0; i < n; i += block) {
                    for (int j = 0; j < stride; j++) {
                        float in[SP_VHT2_MAX_P];
                        for (int k = 0; k < p; k++) {
                            in[k] = data[i + k * stride + j];
                        }
                        for (int k = 0; k < p; k++) {
                            float sum = 0.0f;
                            const float *row = H + k * p;
                            for (int m = 0; m < p; m++) sum += row[m] * in[m];
                            data[i + k * stride + j] = sum;
                        }
                    }
                }
            }
            residue /= p;
            stride  *= p;
        }
    }
    // If residue != 1 here, n had a prime factor > 11 — caller should have
    // padded via sqfree_pad_dim. Leave data unchanged so the failure is loud.
}

void sp_vht2_forward_f16(uint16_t *data, int n) {
    // Reference: float promote, transform, quantise back. Backends with native
    // fp16 support (CUDA, NEON fp16) should override.
    float *tmp = (float *)malloc((size_t)n * sizeof(float));
    if (!tmp) return;
    for (int i = 0; i < n; i++) tmp[i] = sp_f16_to_f32(data[i]);
    sp_vht2_forward_f32(tmp, n);
    for (int i = 0; i < n; i++) data[i] = sp_f32_to_f16(tmp[i]);
    free(tmp);
}


// ============================================================================
// Möbius mask
// ============================================================================
//
// The Möbius function μ(n):
//   μ(1) = 1
//   μ(n) = 0 if n has a squared prime factor
//   μ(n) = (-1)^k if n is product of k distinct primes
//
// Squarefree indices (μ(n) != 0) carry the independent information.
// Non-squarefree indices are partially predictable via Möbius inversion.
// Prioritizing squarefree-first gives +0.14 PPL for free.

static void compute_mobius(int8_t *mu, int n) {
    // Sieve-based computation
    mu[0] = 0; // μ(0) undefined; treat as non-squarefree
    if (n > 1) mu[1] = 1;

    // Initialize
    for (int i = 2; i < n; i++) mu[i] = 1;

    // Mark prime factors
    int *prime_count = (int *)calloc(n, sizeof(int));
    bool *has_square = (bool *)calloc(n, sizeof(bool));

    for (int p = 2; p < n; p++) {
        if (prime_count[p] != 0) continue; // not prime
        // p is prime — mark all multiples
        for (int m = p; m < n; m += p) {
            prime_count[m]++;
        }
        // Mark squared multiples
        long long p2 = (long long)p * p;
        for (long long m = p2; m < n; m += p2) {
            has_square[m] = true;
        }
    }

    for (int i = 2; i < n; i++) {
        if (has_square[i]) {
            mu[i] = 0;
        } else {
            mu[i] = (prime_count[i] % 2 == 0) ? 1 : -1;
        }
    }

    free(prime_count);
    free(has_square);
}

int sp_mobius_mask_init(sp_mobius_mask_t *mask, int n) {
    mask->n = n;
    mask->mu = (int8_t *)malloc(n * sizeof(int8_t));
    mask->order = (int *)malloc(n * sizeof(int));
    if (!mask->mu || !mask->order) return -1;

    compute_mobius(mask->mu, n);

    // Build permutation: squarefree first, then non-squarefree
    int sf = 0;

    // Count squarefree
    for (int i = 0; i < n; i++) {
        if (mask->mu[i] != 0) sf++;
    }
    mask->n_squarefree = sf;

    // Fill order: squarefree indices first
    int pos_sf = 0;
    int pos_nsf = sf;
    for (int i = 0; i < n; i++) {
        if (mask->mu[i] != 0) {
            mask->order[pos_sf++] = i;
        } else {
            mask->order[pos_nsf++] = i;
        }
    }

    return 0;
}

void sp_mobius_mask_free(sp_mobius_mask_t *mask) {
    free(mask->mu);
    free(mask->order);
    mask->mu = NULL;
    mask->order = NULL;
}

// Caller-owned-scratch variants. Used on the hot path (shadow cache,
// Adreno backend) to avoid malloc per KV vector.
void sp_mobius_reorder_ex(float *vht2_coeffs, const sp_mobius_mask_t *mask,
                          float *scratch) {
    int n = mask->n;
    for (int i = 0; i < n; i++) {
        scratch[i] = vht2_coeffs[mask->order[i]];
    }
    memcpy(vht2_coeffs, scratch, n * sizeof(float));
}

void sp_mobius_unreorder_ex(float *vht2_coeffs, const sp_mobius_mask_t *mask,
                            float *scratch) {
    int n = mask->n;
    for (int i = 0; i < n; i++) {
        scratch[mask->order[i]] = vht2_coeffs[i];
    }
    memcpy(vht2_coeffs, scratch, n * sizeof(float));
}

// Malloc-owning variants retained for existing callers (tests, research code).
void sp_mobius_reorder(float *vht2_coeffs, const sp_mobius_mask_t *mask) {
    int n = mask->n;
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) return;
    sp_mobius_reorder_ex(vht2_coeffs, mask, tmp);
    free(tmp);
}

void sp_mobius_unreorder(float *vht2_coeffs, const sp_mobius_mask_t *mask) {
    int n = mask->n;
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) return;
    sp_mobius_unreorder_ex(vht2_coeffs, mask, tmp);
    free(tmp);
}

// ============================================================================
// VHT2 banded quantization
// ============================================================================
//
// Each band stores: 1 fp16 scale + packed integer values.
// The scale is max(abs(band)) / (2^(bits-1) - 1).
// This mirrors VHT2 energy decay: band 0 (highest energy) gets most bits,
// band 3 (lowest energy / noise tail) gets fewest.
//
// Critical results from paper:
//   5/5/4/3 → PPL 11.2147 (BETTER than fp16 11.2194 — spectral regularization)
//   4/4/4/4 → off Pareto frontier (4/4/4/3 strictly dominates)
//   Any band at 2-bit → catastrophic

void sp_band_config_init(sp_band_config_t *bc, int head_dim,
                         int n_bands, const int *band_bits) {
    sp_band_config_init_ext(bc, head_dim, n_bands, band_bits, 0u);
}

void sp_band_config_init_ext(sp_band_config_t *bc, int head_dim,
                             int n_bands, const int *band_bits,
                             uint32_t ternary_band_mask) {
    bc->n_bands           = n_bands;
    bc->head_dim          = head_dim;
    bc->band_size         = head_dim / n_bands;
    bc->ternary_band_mask = ternary_band_mask;

    int total = 0;
    for (int b = 0; b < n_bands; b++) {
        bc->band_bits[b] = band_bits[b];
        int off, sz;
        sp_band_span(bc, b, &off, &sz);
        // Per band: 2 bytes (fp16 scale) + ceil(sz * eff_bits / 8) bytes
        // where eff_bits is 2 for ternary bands (deadband {-1,0,+1} packed
        // 4-per-byte) or band_bits[b] for regular signed-int bands.
        const int is_ternary = (ternary_band_mask & (1u << b)) ? 1 : 0;
        const int eff_bits = is_ternary ? 2 : band_bits[b];
        int data_bits  = sz * eff_bits;
        int data_bytes = (data_bits + 7) / 8;
        total += 2 + data_bytes; // fp16 scale + packed data
    }
    bc->total_bytes = total;
}

void sp_band_quantize(const float *vht2_coeffs, uint8_t *out,
                      const sp_band_config_t *bc) {
    int offset = 0;

    for (int b = 0; b < bc->n_bands; b++) {
        int band_off, band_sz;
        sp_band_span(bc, b, &band_off, &band_sz);
        const float *band = vht2_coeffs + band_off;

        // ── Ternary noise-tail path ──────────────────────────────────────
        // Bit b set in the ternary mask ⇒ quantise to {-1, 0, +1} via a
        // deadband at 0.5*amax. Scale = amax (no max_val division). 2 bits
        // per coefficient, mapping -1→0, 0→1, +1→2 for storage. The fp16
        // scale header is identical to the signed-int path so format
        // consumers can read the scale uniformly.
        if (bc->ternary_band_mask & (1u << b)) {
            float amax = 0.0f;
            for (int i = 0; i < band_sz; i++) {
                float a = fabsf(band[i]);
                if (a > amax) amax = a;
            }

            uint16_t scale_f16 = sp_f32_to_f16(amax);
            out[offset]     = scale_f16 & 0xFF;
            out[offset + 1] = (scale_f16 >> 8) & 0xFF;
            offset += 2;

            // Recover the fp16-rounded scale for the threshold so encode
            // and decode agree byte-for-byte (same logic as the signed
            // path's inv_scale round-trip).
            float scale_stored = sp_f16_to_f32(scale_f16);
            float threshold    = 0.5f * scale_stored;

            uint64_t bit_buffer = 0;
            int      bit_pos    = 0;
            for (int i = 0; i < band_sz; i++) {
                int q = 0;
                if      (band[i] >  threshold) q = 1;
                else if (band[i] < -threshold) q = -1;
                // Store as 2-bit unsigned, offset by 1: -1→0, 0→1, +1→2.
                uint32_t u = (uint32_t)(q + 1);
                bit_buffer |= ((uint64_t)u << bit_pos);
                bit_pos += 2;
                while (bit_pos >= 8) {
                    out[offset++] = (uint8_t)(bit_buffer & 0xFF);
                    bit_buffer >>= 8;
                    bit_pos -= 8;
                }
            }
            if (bit_pos > 0) {
                out[offset++] = (uint8_t)(bit_buffer & 0xFF);
            }
            continue;
        }

        int bits = bc->band_bits[b];
        int max_val = (1 << (bits - 1)) - 1; // e.g. 5-bit → 15

        // Find max absolute value in band
        float amax = 0.0f;
        for (int i = 0; i < band_sz; i++) {
            float a = fabsf(band[i]);
            if (a > amax) amax = a;
        }

        // Scale: maps [-amax, amax] to [-max_val, max_val]
        float scale = (amax > 0.0f) ? amax / (float)max_val : 0.0f;

        // Store scale as fp16, then recompute inv_scale from the stored
        // fp16 value so quantize and dequantize use the same scale.
        // Without this round-trip, sp_f32_to_f16 truncation creates a
        // systematic asymmetry that variance-ranking amplifies in band 0.
        uint16_t scale_f16 = sp_f32_to_f16(scale);
        out[offset]     = scale_f16 & 0xFF;
        out[offset + 1] = (scale_f16 >> 8) & 0xFF;
        offset += 2;

        float scale_stored = sp_f16_to_f32(scale_f16);
        float inv_scale = (scale_stored > 0.0f) ? 1.0f / scale_stored : 0.0f;

        // Pack quantized values
        // We pack bits-wide signed integers in little-endian bit order
        uint64_t bit_buffer = 0;
        int      bit_pos = 0;

        for (int i = 0; i < band_sz; i++) {
            // Quantize to signed integer
            int q = (int)roundf(band[i] * inv_scale);
            if (q > max_val)  q = max_val;
            if (q < -max_val) q = -max_val;

            // Convert to unsigned representation for packing
            // Offset by max_val so range is [0, 2*max_val]
            uint32_t u = (uint32_t)(q + max_val);

            bit_buffer |= ((uint64_t)u << bit_pos);
            bit_pos += bits;

            // Flush full bytes
            while (bit_pos >= 8) {
                out[offset++] = (uint8_t)(bit_buffer & 0xFF);
                bit_buffer >>= 8;
                bit_pos -= 8;
            }
        }

        // Flush remaining bits
        if (bit_pos > 0) {
            out[offset++] = (uint8_t)(bit_buffer & 0xFF);
        }
    }
}

void sp_band_dequantize(const uint8_t *in, float *vht2_coeffs,
                        const sp_band_config_t *bc) {
    int offset = 0;

    for (int b = 0; b < bc->n_bands; b++) {
        int band_off, band_sz;
        sp_band_span(bc, b, &band_off, &band_sz);
        float *band = vht2_coeffs + band_off;

        // ── Ternary noise-tail path ──────────────────────────────────────
        // Mirrors the encode-side ternary branch. Reads the fp16 scale,
        // unpacks 2 bits/coeff into {0,1,2}, maps back to {-1,0,+1}·scale.
        if (bc->ternary_band_mask & (1u << b)) {
            uint16_t scale_f16 = (uint16_t)in[offset]
                               | ((uint16_t)in[offset + 1] << 8);
            float scale = sp_f16_to_f32(scale_f16);
            if (!isfinite(scale)) scale = 0.0f;
            offset += 2;

            uint64_t bit_buffer = 0;
            int      bit_pos    = 0;
            int      byte_idx   = offset;
            for (int i = 0; i < band_sz; i++) {
                while (bit_pos < 2) {
                    bit_buffer |= ((uint64_t)in[byte_idx++] << bit_pos);
                    bit_pos += 8;
                }
                uint32_t u = (uint32_t)(bit_buffer & 0x3u);
                bit_buffer >>= 2;
                bit_pos -= 2;
                // u ∈ {0, 1, 2} → {-1, 0, +1}. Any other value (would only
                // happen on corrupted input) maps to 0.
                int q = (int)u - 1;
                if (q < -1 || q > 1) q = 0;
                band[i] = (float)q * scale;
            }
            offset = byte_idx;
            continue;
        }

        int bits = bc->band_bits[b];
        int max_val = (1 << (bits - 1)) - 1;
        uint32_t mask = (1u << bits) - 1;

        // Read scale. Sanitise: fp16 round-trip can produce +Inf (amax overflowed
        // the fp16 range on encode) or NaN (corrupted bytes). An Inf or NaN scale
        // would poison every value in this band and then cascade through the
        // inverse VHT2, so we clamp to 0 here — the band decodes as all zeros,
        // which is the same outcome the old blanket NaN guard used to produce,
        // but applied at the root cause rather than the output tail.
        uint16_t scale_f16 = (uint16_t)in[offset] | ((uint16_t)in[offset + 1] << 8);
        float scale = sp_f16_to_f32(scale_f16);
        if (!isfinite(scale)) scale = 0.0f;
        offset += 2;

        // Unpack quantized values
        uint64_t bit_buffer = 0;
        int      bit_pos = 0;
        int      byte_idx = offset;

        for (int i = 0; i < band_sz; i++) {
            // Load bytes as needed
            while (bit_pos < bits) {
                bit_buffer |= ((uint64_t)in[byte_idx++] << bit_pos);
                bit_pos += 8;
            }

            uint32_t u = (uint32_t)(bit_buffer & mask);
            bit_buffer >>= bits;
            bit_pos -= bits;

            // Convert back to signed
            int q = (int)u - max_val;
            band[i] = (float)q * scale;
        }

        // Advance offset past the packed data
        int data_bits = band_sz * bits;
        offset += (data_bits + 7) / 8;
    }
}

void sp_band_dequantize_partial(const uint8_t *in, float *vht2_coeffs,
                                const sp_band_config_t *bc,
                                int max_bands) {
    // Clamp max_bands into the valid range. Zero is valid (returns the
    // all-zero vector); n_bands is valid (equivalent to sp_band_dequantize).
    if (max_bands < 0)            max_bands = 0;
    if (max_bands > bc->n_bands)  max_bands = bc->n_bands;

    int offset = 0;

    for (int b = 0; b < bc->n_bands; b++) {
        int band_off, band_sz;
        sp_band_span(bc, b, &band_off, &band_sz);
        float *band = vht2_coeffs + band_off;

        if (b >= max_bands) {
            // Skip-path: zero this band's coefficients in place. Don't
            // touch the input bytes for this band — the offset cursor
            // doesn't matter past the last band we read, since we're
            // returning early.
            for (int i = 0; i < band_sz; i++) band[i] = 0.0f;
            continue;
        }

        // ── Ternary noise-tail path (mirrors sp_band_dequantize) ──────────
        if (bc->ternary_band_mask & (1u << b)) {
            uint16_t scale_f16 = (uint16_t)in[offset]
                               | ((uint16_t)in[offset + 1] << 8);
            float scale = sp_f16_to_f32(scale_f16);
            if (!isfinite(scale)) scale = 0.0f;
            offset += 2;

            uint64_t bit_buffer = 0;
            int      bit_pos    = 0;
            int      byte_idx   = offset;
            for (int i = 0; i < band_sz; i++) {
                while (bit_pos < 2) {
                    bit_buffer |= ((uint64_t)in[byte_idx++] << bit_pos);
                    bit_pos += 8;
                }
                uint32_t u = (uint32_t)(bit_buffer & 0x3u);
                bit_buffer >>= 2;
                bit_pos -= 2;
                int q = (int)u - 1;
                if (q < -1 || q > 1) q = 0;
                band[i] = (float)q * scale;
            }
            offset = byte_idx;
            continue;
        }

        // ── Regular signed-int band path (mirrors sp_band_dequantize) ─────
        int bits = bc->band_bits[b];
        int max_val = (1 << (bits - 1)) - 1;
        uint32_t mask = (1u << bits) - 1;

        uint16_t scale_f16 = (uint16_t)in[offset] | ((uint16_t)in[offset + 1] << 8);
        float scale = sp_f16_to_f32(scale_f16);
        if (!isfinite(scale)) scale = 0.0f;
        offset += 2;

        uint64_t bit_buffer = 0;
        int      bit_pos = 0;
        int      byte_idx = offset;

        for (int i = 0; i < band_sz; i++) {
            while (bit_pos < bits) {
                bit_buffer |= ((uint64_t)in[byte_idx++] << bit_pos);
                bit_pos += 8;
            }
            uint32_t u = (uint32_t)(bit_buffer & mask);
            bit_buffer >>= bits;
            bit_pos -= bits;
            int q = (int)u - max_val;
            band[i] = (float)q * scale;
        }

        int data_bits = band_sz * bits;
        offset += (data_bits + 7) / 8;
    }
}

// ============================================================================
// Vilenkin-Hartley Transform
// ============================================================================
//
// Hartley kernel: cas(x) = cos(x) + sin(x)
// For prime p, the p×p Hartley matrix has entries:
//   H[i][j] = cas(2π·i·j / p) / sqrt(p)
// The full basis is the Kronecker product: V = H_p1 ⊗ H_p2 ⊗ ... ⊗ H_pk
// Self-inverse: V·V = N·I (round-trip error = 0.0000)

static const int vilenkin_primes[] = { 2, 3, 5, 7, 11, 13 };

int sp_vilenkin_init(sp_vilenkin_basis_t *vb, int n_primes) {
    if (n_primes < 1 || n_primes > 6) return -1;

    vb->n_primes = n_primes;
    vb->n = 1;
    for (int i = 0; i < n_primes; i++) {
        vb->primes[i] = vilenkin_primes[i];
        vb->n *= vilenkin_primes[i];
    }

    // Allocate n×n basis matrix
    vb->basis = (float *)malloc((size_t)vb->n * vb->n * sizeof(float));
    if (!vb->basis) return -1;

    // Build via Kronecker product of per-prime Hartley matrices
    // Start with 1×1 identity, then Kronecker with each H_p

    // Current matrix (starts as [1.0])
    int cur_n = 1;
    float *cur = (float *)malloc(sizeof(float));
    cur[0] = 1.0f;

    for (int pi = 0; pi < n_primes; pi++) {
        int p = vb->primes[pi];
        int new_n = cur_n * p;
        float *next = (float *)malloc((size_t)new_n * new_n * sizeof(float));

        // H_p[i][j] = cas(2π·i·j / p) / sqrt(p)
        float norm = 1.0f / sqrtf((float)p);

        for (int ci = 0; ci < cur_n; ci++) {
            for (int cj = 0; cj < cur_n; cj++) {
                float c_val = cur[ci * cur_n + cj];
                for (int hi = 0; hi < p; hi++) {
                    for (int hj = 0; hj < p; hj++) {
                        float angle = 2.0f * (float)M_PI * (float)(hi * hj) / (float)p;
                        float h_val = (cosf(angle) + sinf(angle)) * norm;
                        int ri = ci * p + hi;
                        int rj = cj * p + hj;
                        next[ri * new_n + rj] = c_val * h_val;
                    }
                }
            }
        }

        free(cur);
        cur = next;
        cur_n = new_n;
    }

    memcpy(vb->basis, cur, (size_t)vb->n * vb->n * sizeof(float));
    free(cur);
    return 0;
}

void sp_vilenkin_free(sp_vilenkin_basis_t *vb) {
    free(vb->basis);
    vb->basis = NULL;
}

void sp_vilenkin_forward(const sp_vilenkin_basis_t *vb,
                         const float *input, int head_dim,
                         float *output) {
    int n = vb->n;

    // Zero-pad input if head_dim < n
    float *padded = (float *)calloc(n, sizeof(float));
    memcpy(padded, input, head_dim * sizeof(float));

    // Matrix multiply: output = V · padded
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += vb->basis[i * n + j] * padded[j];
        }
        output[i] = sum;
    }

    free(padded);
}

void sp_vilenkin_inverse(const sp_vilenkin_basis_t *vb,
                         const float *input,
                         float *output, int head_dim) {
    int n = vb->n;

    // V is orthonormal (V·V = I), so inverse = V (same as forward)
    float *full = (float *)calloc(n, sizeof(float));

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += vb->basis[i * n + j] * input[j];
        }
        full[i] = sum;
    }

    // Truncate back to head_dim
    memcpy(output, full, head_dim * sizeof(float));
    free(full);
}

int sp_vilenkin_extract_pass(const sp_vilenkin_basis_t *vb,
                             float *residual, int head_dim,
                             float energy_threshold,
                             sp_vilenkin_pass_t *pass) {
    int n = vb->n;

    // Forward transform residual
    float *coeffs = (float *)malloc(n * sizeof(float));
    sp_vilenkin_forward(vb, residual, head_dim, coeffs);

    // Compute total energy
    float total_energy = 0.0f;
    for (int i = 0; i < n; i++) {
        total_energy += coeffs[i] * coeffs[i];
    }

    // Sort indices by descending energy
    int *sorted_idx = (int *)malloc(n * sizeof(int));
    float *energies = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        sorted_idx[i] = i;
        energies[i] = coeffs[i] * coeffs[i];
    }

    // Simple insertion sort (n is small: 6, 30, or 210)
    for (int i = 1; i < n; i++) {
        int key_idx = sorted_idx[i];
        float key_e = energies[i];
        int j = i - 1;
        while (j >= 0 && energies[j] < key_e) {
            sorted_idx[j + 1] = sorted_idx[j];
            energies[j + 1]   = energies[j];
            j--;
        }
        sorted_idx[j + 1] = key_idx;
        energies[j + 1]   = key_e;
    }

    // Select coefficients until energy threshold reached
    float captured = 0.0f;
    float target = total_energy * energy_threshold;
    int count = 0;

    while (count < n && captured < target) {
        captured += energies[count];
        count++;
    }

    // Allocate pass
    pass->n_coeffs = count;
    pass->indices  = (int *)malloc(count * sizeof(int));
    pass->values   = (float *)malloc(count * sizeof(float));

    for (int i = 0; i < count; i++) {
        pass->indices[i] = sorted_idx[i];
        pass->values[i]  = coeffs[sorted_idx[i]];
    }

    // Subtract extracted component from residual
    // Reconstruct the extracted part and subtract
    float *extracted = (float *)calloc(n, sizeof(float));
    for (int i = 0; i < count; i++) {
        extracted[sorted_idx[i]] = coeffs[sorted_idx[i]];
    }

    float *reconstructed = (float *)calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += vb->basis[i * n + j] * extracted[j];
        }
        reconstructed[i] = sum;
    }

    for (int i = 0; i < head_dim; i++) {
        residual[i] -= reconstructed[i];
    }

    free(coeffs);
    free(sorted_idx);
    free(energies);
    free(extracted);
    free(reconstructed);
    return 0;
}

void sp_vilenkin_pass_free(sp_vilenkin_pass_t *pass) {
    free(pass->indices);
    free(pass->values);
    pass->indices = NULL;
    pass->values = NULL;
}

// ============================================================================
// Shadow cache
// ============================================================================

int sp_shadow_cache_init(sp_shadow_cache_t *sc, const sp_config_t *cfg) {
    memset(sc, 0, sizeof(*sc));
    memcpy(&sc->config, cfg, sizeof(sp_config_t));

    // Initialize band configs. Use _ext to honour ternary masks; when
    // cfg->[kv]_ternary_mask is 0 (the default for callers initialised via
    // sp_config_init), behaviour is identical to sp_band_config_init.
    sp_band_config_init_ext(&sc->k_bands, cfg->head_dim,
                            cfg->k_n_bands, cfg->k_band_bits,
                            cfg->k_ternary_mask);
    sp_band_config_init_ext(&sc->v_bands, cfg->head_dim,
                            cfg->v_n_bands, cfg->v_band_bits,
                            cfg->v_ternary_mask);

    // Initialize Möbius mask
    if (cfg->use_mobius_mask) {
        if (sp_mobius_mask_init(&sc->mobius_mask, cfg->head_dim) != 0) {
            return -1;
        }
    }

    // Allocate persistent scratch buffers so the hot path never mallocs.
    // vht2_scratch    : write-path VHT2 buffer
    // mobius_scratch  : Möbius reorder/unreorder tmp (shared between write+read)
    // read_scratch    : read-path VHT2 buffer (independent of write path)
    sc->vht2_scratch    = (float *)malloc(cfg->head_dim * sizeof(float));
    sc->mobius_scratch = (float *)malloc(cfg->head_dim * sizeof(float));
    sc->read_scratch   = (float *)malloc(cfg->head_dim * sizeof(float));
    if (!sc->vht2_scratch || !sc->mobius_scratch || !sc->read_scratch) return -1;

    // Cache storage will be allocated by backend (depends on max_seq_len)
    sc->k_cache = NULL;
    sc->v_cache = NULL;
    sc->seq_len = (int *)calloc(cfg->n_layers, sizeof(int));

    // Variance-ranked reorder: off until calibrated
    sc->use_var_reorder = false;
    sc->var_order = NULL;
    sc->var_unorder = NULL;
    sc->calibrating = false;
    sc->calib_sum = NULL;
    sc->calib_sum2 = NULL;
    sc->calib_n = 0;

    return 0;
}

void sp_shadow_cache_free(sp_shadow_cache_t *sc) {
    if (sc->config.use_mobius_mask) {
        sp_mobius_mask_free(&sc->mobius_mask);
    }
    free(sc->vht2_scratch);
    free(sc->mobius_scratch);
    free(sc->read_scratch);
    free(sc->seq_len);
    free(sc->var_order);
    free(sc->var_unorder);
    free(sc->calib_sum);
    free(sc->calib_sum2);
    // k_cache and v_cache freed by backend
    sc->vht2_scratch    = NULL;
    sc->mobius_scratch = NULL;
    sc->read_scratch   = NULL;
    sc->seq_len        = NULL;
    sc->var_order      = NULL;
    sc->var_unorder    = NULL;
    sc->calib_sum      = NULL;
    sc->calib_sum2     = NULL;
}

// Write path: raw KV → VHT2 → Möbius reorder → band quantize → store
// Hot path: uses persistent sc->vht2_scratch + sc->mobius_scratch. No malloc.
void sp_shadow_write_k(sp_shadow_cache_t *sc,
                       int layer, int head, int pos,
                       const float *k_vec) {
    int hd = sc->config.head_dim;
    float *scratch = sc->vht2_scratch;

    memcpy(scratch, k_vec, hd * sizeof(float));
    sp_vht2_forward_f32(scratch, hd);

    // Reorder: variance-ranked (if calibrated) > Möbius (if enabled) > none
    if (sc->use_var_reorder) {
        // Permute by variance: high-variance → front → high-bit bands
        float *tmp = sc->mobius_scratch;
        for (int i = 0; i < hd; i++) tmp[i] = scratch[sc->var_order[i]];
        memcpy(scratch, tmp, hd * sizeof(float));
    } else if (sc->config.use_mobius_mask) {
        sp_mobius_reorder_ex(scratch, &sc->mobius_mask, sc->mobius_scratch);
    }

    int slot = layer * sc->config.n_heads_kv + head;
    uint8_t *dest = sc->k_cache[slot] + (size_t)pos * sc->k_bands.total_bytes;
    sp_band_quantize(scratch, dest, &sc->k_bands);
}

void sp_shadow_write_v(sp_shadow_cache_t *sc,
                       int layer, int head, int pos,
                       const float *v_vec) {
    int hd = sc->config.head_dim;
    float *scratch = sc->vht2_scratch;

    memcpy(scratch, v_vec, hd * sizeof(float));
    sp_vht2_forward_f32(scratch, hd);

    // No Möbius reorder for V (uniform spectrum — no benefit)
    // V gets flat quantization (1 band), no reordering needed

    int slot = layer * sc->config.n_heads_kv + head;
    uint8_t *dest = sc->v_cache[slot] + (size_t)pos * sc->v_bands.total_bytes;
    sp_band_quantize(scratch, dest, &sc->v_bands);
}

// Read path: load → band dequantize → Möbius unreorder → VHT2 (self-inverse) → KV
// Hot path: uses persistent sc->read_scratch + sc->mobius_scratch. No malloc.
// The `const` on sc is a contract for thread-call-safety, not true immutability
// — we write into sc->read_scratch / sc->mobius_scratch. Callers must serialize.
// Internal helper used by both the full and partial read paths.
// max_bands < 0 OR >= bc->n_bands selects the full path (sp_band_dequantize);
// otherwise sp_band_dequantize_partial is called with that bound.
static void sp_shadow_read_k_impl(const sp_shadow_cache_t *sc,
                                  int layer, int head, int pos,
                                  float *k_out, int max_bands) {
    int hd = sc->config.head_dim;
    float *scratch = sc->read_scratch;

    int slot = layer * sc->config.n_heads_kv + head;
    const uint8_t *src = sc->k_cache[slot] + (size_t)pos * sc->k_bands.total_bytes;

    if (max_bands < 0 || max_bands >= sc->k_bands.n_bands) {
        sp_band_dequantize(src, scratch, &sc->k_bands);
    } else {
        sp_band_dequantize_partial(src, scratch, &sc->k_bands, max_bands);
    }

    // Inverse reorder: variance-ranked (if calibrated) > Möbius > none
    if (sc->use_var_reorder) {
        float *tmp = sc->mobius_scratch;
        for (int i = 0; i < hd; i++) tmp[sc->var_order[i]] = scratch[i];
        memcpy(scratch, tmp, hd * sizeof(float));
    } else if (sc->config.use_mobius_mask) {
        sp_mobius_unreorder_ex(scratch, &sc->mobius_mask, sc->mobius_scratch);
    }

    // Inverse transform: VHT2 is self-inverse (1/√p per stage absorbs the
    // 1/N the old unnormalised p=2 butterfly required). Zero coefficients
    // in unread bands collapse butterfly contributions to zero on those
    // axes, giving a partial-fidelity reconstruction in head_dim space.
    sp_vht2_forward_f32(scratch, hd);

    memcpy(k_out, scratch, hd * sizeof(float));
}

static void sp_shadow_read_v_impl(const sp_shadow_cache_t *sc,
                                  int layer, int head, int pos,
                                  float *v_out, int max_bands) {
    int hd = sc->config.head_dim;
    float *scratch = sc->read_scratch;

    int slot = layer * sc->config.n_heads_kv + head;
    const uint8_t *src = sc->v_cache[slot] + (size_t)pos * sc->v_bands.total_bytes;

    if (max_bands < 0 || max_bands >= sc->v_bands.n_bands) {
        sp_band_dequantize(src, scratch, &sc->v_bands);
    } else {
        sp_band_dequantize_partial(src, scratch, &sc->v_bands, max_bands);
    }

    sp_vht2_forward_f32(scratch, hd);

    memcpy(v_out, scratch, hd * sizeof(float));
}

void sp_shadow_read_k(const sp_shadow_cache_t *sc,
                      int layer, int head, int pos,
                      float *k_out) {
    sp_shadow_read_k_impl(sc, layer, head, pos, k_out, -1 /* full */);
}

void sp_shadow_read_v(const sp_shadow_cache_t *sc,
                      int layer, int head, int pos,
                      float *v_out) {
    sp_shadow_read_v_impl(sc, layer, head, pos, v_out, -1 /* full */);
}

void sp_shadow_read_k_partial(const sp_shadow_cache_t *sc,
                              int layer, int head, int pos,
                              float *k_out, int max_bands) {
    sp_shadow_read_k_impl(sc, layer, head, pos, k_out, max_bands);
}

void sp_shadow_read_v_partial(const sp_shadow_cache_t *sc,
                              int layer, int head, int pos,
                              float *v_out, int max_bands) {
    sp_shadow_read_v_impl(sc, layer, head, pos, v_out, max_bands);
}

// Batch variants. Tight loop reusing the persistent scratch. Zero mallocs
// per batch, amortizes the "copy → transform → store/load → transform
// → copy" pipeline setup across n_pos vectors.
void sp_shadow_write_k_batch(sp_shadow_cache_t *sc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *k_vecs) {
    int hd = sc->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_shadow_write_k(sc, layer, head, start_pos + i, k_vecs + (size_t)i * hd);
    }
}

void sp_shadow_write_v_batch(sp_shadow_cache_t *sc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *v_vecs) {
    int hd = sc->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_shadow_write_v(sc, layer, head, start_pos + i, v_vecs + (size_t)i * hd);
    }
}

void sp_shadow_read_k_batch(const sp_shadow_cache_t *sc,
                            int layer, int head,
                            int start_pos, int n_pos,
                            float *k_out) {
    int hd = sc->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_shadow_read_k(sc, layer, head, start_pos + i, k_out + (size_t)i * hd);
    }
}

void sp_shadow_read_v_batch(const sp_shadow_cache_t *sc,
                            int layer, int head,
                            int start_pos, int n_pos,
                            float *v_out) {
    int hd = sc->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_shadow_read_v(sc, layer, head, start_pos + i, v_out + (size_t)i * hd);
    }
}

// ============================================================================
// Ship-path variance-ranked calibration
// ============================================================================

int sp_shadow_calibrate_begin(sp_shadow_cache_t *sc) {
    if (sc->calibrating) return -1;
    int hd = sc->config.head_dim;
    sc->calib_sum  = (double *)calloc(hd, sizeof(double));
    sc->calib_sum2 = (double *)calloc(hd, sizeof(double));
    if (!sc->calib_sum || !sc->calib_sum2) {
        free(sc->calib_sum);
        free(sc->calib_sum2);
        sc->calib_sum = NULL;
        sc->calib_sum2 = NULL;
        return -1;
    }
    sc->calib_n = 0;
    sc->calibrating = true;
    return 0;
}

void sp_shadow_calibrate_feed(sp_shadow_cache_t *sc, const float *vec) {
    if (!sc->calibrating) return;
    int hd = sc->config.head_dim;

    // Transform to VHT2 domain using the persistent scratch
    memcpy(sc->vht2_scratch, vec, hd * sizeof(float));
    sp_vht2_forward_f32(sc->vht2_scratch, hd);

    for (int i = 0; i < hd; i++) {
        double v = (double)sc->vht2_scratch[i];
        sc->calib_sum[i]  += v;
        sc->calib_sum2[i] += v * v;
    }
    sc->calib_n++;
}

int sp_shadow_calibrate_end(sp_shadow_cache_t *sc) {
    if (!sc->calibrating || sc->calib_n < 1) return -1;
    sc->calibrating = false;

    int hd = sc->config.head_dim;
    double inv_n = 1.0 / (double)sc->calib_n;

    // Compute per-coefficient variance
    float *variance = (float *)malloc(hd * sizeof(float));
    for (int i = 0; i < hd; i++) {
        double mean = sc->calib_sum[i] * inv_n;
        double var  = sc->calib_sum2[i] * inv_n - mean * mean;
        variance[i] = (var > 0.0) ? (float)var : 0.0f;
    }

    free(sc->calib_sum);
    free(sc->calib_sum2);
    sc->calib_sum = NULL;
    sc->calib_sum2 = NULL;

    // Build variance-ranked permutation: indices sorted by variance descending
    // so highest-variance coefficients land in band 0 (highest bits).
    // Free any prior allocation (safe even on first call — they start NULL).
    free(sc->var_order);
    free(sc->var_unorder);
    sc->var_order   = (int *)malloc(hd * sizeof(int));
    sc->var_unorder = (int *)malloc(hd * sizeof(int));
    for (int i = 0; i < hd; i++) sc->var_order[i] = i;

    // Insertion sort (head_dim ≤ 256, not hot path)
    for (int i = 1; i < hd; i++) {
        int key = sc->var_order[i];
        float kv = variance[key];
        int j = i - 1;
        while (j >= 0 && variance[sc->var_order[j]] < kv) {
            sc->var_order[j + 1] = sc->var_order[j];
            j--;
        }
        sc->var_order[j + 1] = key;
    }

    // Build inverse permutation for the read path
    for (int i = 0; i < hd; i++) {
        sc->var_unorder[sc->var_order[i]] = i;
    }

    sc->use_var_reorder = true;
    free(variance);

    if (getenv("SHANNON_PRIME_VERBOSE")) {
        fprintf(stderr, "[Shannon-Prime SHADOW] variance-ranked reorder calibrated "
                        "(head_dim=%d, n_vectors=%d)\n", hd, sc->calib_n);
    }

    sc->calib_n = 0;
    return 0;
}

// ============================================================================
// Diagnostics
// ============================================================================

float sp_correlation_f32(const float *a, const float *b, int n) {
    double sum_a = 0, sum_b = 0, sum_ab = 0;
    double sum_a2 = 0, sum_b2 = 0;

    for (int i = 0; i < n; i++) {
        sum_a  += a[i];
        sum_b  += b[i];
        sum_ab += (double)a[i] * b[i];
        sum_a2 += (double)a[i] * a[i];
        sum_b2 += (double)b[i] * b[i];
    }

    double mean_a = sum_a / n;
    double mean_b = sum_b / n;
    double cov = sum_ab / n - mean_a * mean_b;
    double var_a = sum_a2 / n - mean_a * mean_a;
    double var_b = sum_b2 / n - mean_b * mean_b;

    if (var_a < 1e-12 || var_b < 1e-12) return 0.0f;
    return (float)(cov / sqrt(var_a * var_b));
}

float sp_compression_ratio(const sp_config_t *cfg) {
    int hd = cfg->head_dim;
    int baseline_bytes = hd * 2; // fp16 per element

    // K compressed size
    sp_band_config_t kbc;
    sp_band_config_init(&kbc, hd, cfg->k_n_bands, cfg->k_band_bits);

    // V compressed size
    sp_band_config_t vbc;
    sp_band_config_init(&vbc, hd, cfg->v_n_bands, cfg->v_band_bits);

    // Total ratio (K and V equally weighted)
    return 2.0f * (float)baseline_bytes / (float)(kbc.total_bytes + vbc.total_bytes);
}

void sp_config_print(const sp_config_t *cfg) {
    fprintf(stderr, "Shannon-Prime VHT2 Configuration:\n");
    fprintf(stderr, "  head_dim:     %d\n", cfg->head_dim);
    fprintf(stderr, "  n_layers:     %d\n", cfg->n_layers);
    fprintf(stderr, "  n_heads_kv:   %d\n", cfg->n_heads_kv);

    fprintf(stderr, "  K bands:      %d (", cfg->k_n_bands);
    for (int i = 0; i < cfg->k_n_bands; i++) {
        fprintf(stderr, "%d%s", cfg->k_band_bits[i],
                i < cfg->k_n_bands - 1 ? "/" : "");
    }
    fprintf(stderr, ")\n");

    fprintf(stderr, "  V bands:      %d (", cfg->v_n_bands);
    for (int i = 0; i < cfg->v_n_bands; i++) {
        fprintf(stderr, "%d%s", cfg->v_band_bits[i],
                i < cfg->v_n_bands - 1 ? "/" : "");
    }
    fprintf(stderr, ")\n");

    fprintf(stderr, "  Möbius mask:  %s\n", cfg->use_mobius_mask ? "on" : "off");
    fprintf(stderr, "  Vilenkin:     %s", cfg->use_vilenkin ? "on" : "off");
    if (cfg->use_vilenkin) {
        fprintf(stderr, " (%d primes, %.0f%% energy)",
                cfg->vilenkin_primes, cfg->energy_threshold * 100);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "  Compression:  %.1f×\n", sp_compression_ratio(cfg));
}

// ============================================================================
// Disk Serialization
// ============================================================================

#include <stdio.h>

uint64_t sp_fnv1a_hash(const char *str, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)(unsigned char)str[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// Write 64-byte VHT2 v2 header
static int sp_write_cache_header(FILE *f, int packed_stride, int n_pos,
                                 int n_heads, int cache_type,
                                 uint64_t model_hash) {
    uint32_t hdr[16] = {0};
    hdr[0] = SP_CACHE_MAGIC;
    hdr[1] = SP_CACHE_VERSION;
    hdr[2] = (uint32_t)packed_stride;
    hdr[3] = (uint32_t)n_pos;
    hdr[4] = (uint32_t)n_heads;
    hdr[5] = (uint32_t)cache_type;
    hdr[6] = (uint32_t)(model_hash & 0xFFFFFFFF);
    hdr[7] = (uint32_t)(model_hash >> 32);
    return (fwrite(hdr, sizeof(hdr), 1, f) == 1) ? 0 : -1;
}

// Compute per-band per-(head,pos) record byte size (2-byte fp16 scale +
// ceil(band_size * effective_bits / 8)). effective_bits is 2 for ternary
// bands, else band_bits[b]. Returns 0 if band index is out of range.
static int sp_band_record_bytes(const sp_band_config_t *bc, int b) {
    if (b < 0 || b >= bc->n_bands) return 0;
    int off, sz;
    sp_band_span(bc, b, &off, &sz);
    int eff_bits = (bc->ternary_band_mask & (1u << b)) ? 2 : bc->band_bits[b];
    int data_bytes = (sz * eff_bits + 7) / 8;
    return 2 + data_bytes;
}

// Write 64-byte v3 band-major header. Bumps version to
// SP_CACHE_VERSION_BAND_MAJOR. Stores n_bands at slot 8 and the first
// 7 band offsets (in bytes from file start) at slots 9-15. Caps disk
// persistence at 7 bands; configs with more bands fall back to writing
// v2 headers via the existing helper.
static int sp_write_cache_header_v3(FILE *f, int packed_stride, int n_pos,
                                    int n_heads, int cache_type,
                                    uint64_t model_hash,
                                    int n_bands,
                                    const uint32_t band_offsets[],
                                    int n_band_offsets) {
    uint32_t hdr[16] = {0};
    hdr[0] = SP_CACHE_MAGIC;
    hdr[1] = SP_CACHE_VERSION_BAND_MAJOR;
    hdr[2] = (uint32_t)packed_stride;
    hdr[3] = (uint32_t)n_pos;
    hdr[4] = (uint32_t)n_heads;
    hdr[5] = (uint32_t)cache_type;
    hdr[6] = (uint32_t)(model_hash & 0xFFFFFFFF);
    hdr[7] = (uint32_t)(model_hash >> 32);
    hdr[8] = (uint32_t)n_bands;
    int slots = (n_band_offsets > 7) ? 7 : n_band_offsets;
    for (int b = 0; b < slots; b++) {
        hdr[9 + b] = band_offsets[b];
    }
    return (fwrite(hdr, sizeof(hdr), 1, f) == 1) ? 0 : -1;
}

// Read v3 header. Returns 0 on v3 OK, -1 on bad magic / read failure,
// -2 on hash mismatch in strict mode, +1 on v2 (caller should fall back
// to legacy reader). Out parameters populated for v3; v2 path leaves
// n_bands_out and band_offsets unset.
static int sp_read_cache_header_v3(FILE *f, int *packed_stride, int *n_pos,
                                   int *n_heads, int *cache_type,
                                   uint64_t expected_hash,
                                   int *n_bands_out,
                                   uint32_t band_offsets_out[7]) {
    uint32_t hdr[16] = {0};
    if (fread(hdr, sizeof(hdr), 1, f) != 1)        return -1;
    if (hdr[0] != SP_CACHE_MAGIC)                    return -1;

    if (hdr[1] == SP_CACHE_VERSION) {
        // Caller asked for v3 but file is legacy v2 — signal fallback.
        return 1;
    }
    if (hdr[1] != SP_CACHE_VERSION_BAND_MAJOR)       return -1;

    *packed_stride = (int)hdr[2];
    *n_pos         = (int)hdr[3];
    *n_heads       = (int)hdr[4];
    *cache_type    = (int)hdr[5];
    if (n_bands_out) *n_bands_out = (int)hdr[8];
    if (band_offsets_out) {
        for (int b = 0; b < 7; b++) band_offsets_out[b] = hdr[9 + b];
    }

    if (expected_hash != 0) {
        uint64_t disk_hash = (uint64_t)hdr[6] | ((uint64_t)hdr[7] << 32);
        if (disk_hash != 0 && disk_hash != expected_hash) {
            fprintf(stderr, "[sp-disk] model hash mismatch: disk=%llx expected=%llx\n",
                    (unsigned long long)disk_hash,
                    (unsigned long long)expected_hash);
            const char *strict_env = getenv("SP_DISK_HASH_STRICT");
            const int  warn_only   = strict_env && strict_env[0] == '0';
            if (!warn_only) {
                fprintf(stderr, "[sp-disk] refusing to load: hash mismatch is "
                        "fatal (set SP_DISK_HASH_STRICT=0 to load anyway)\n");
                return -2;
            }
            fprintf(stderr, "[sp-disk] SP_DISK_HASH_STRICT=0 — loading anyway\n");
        }
    }
    return 0;
}

// Read and validate 64-byte VHT2 v2 header
static int sp_read_cache_header(FILE *f, int *packed_stride, int *n_pos,
                                int *n_heads, int *cache_type,
                                uint64_t expected_hash) {
    uint32_t hdr[16] = {0};
    if (fread(hdr, sizeof(hdr), 1, f) != 1)        return -1;
    if (hdr[0] != SP_CACHE_MAGIC)                    return -1;
    if (hdr[1] != SP_CACHE_VERSION)                  return -1;

    *packed_stride = (int)hdr[2];
    *n_pos         = (int)hdr[3];
    *n_heads       = (int)hdr[4];
    *cache_type    = (int)hdr[5];

    if (expected_hash != 0) {
        uint64_t disk_hash = (uint64_t)hdr[6] | ((uint64_t)hdr[7] << 32);
        if (disk_hash != 0 && disk_hash != expected_hash) {
            fprintf(stderr, "[sp-disk] model hash mismatch: disk=%llx expected=%llx\n",
                    (unsigned long long)disk_hash,
                    (unsigned long long)expected_hash);
            // Strict by default — refuse to load to protect against silent
            // garbage output from a cross-model load. Escape hatch is
            // SP_DISK_HASH_STRICT=0 (matches the prior Archimedes-style
            // warn-only behaviour for advanced users who know what they're
            // doing). Default flipped from warn-only after the disk-cache
            // smoke test exposed the API/impl divergence: the C++ wrapper
            // doc in kv_cache.h:143-147 says mismatch returns -1 from the
            // load — strict mode honours that contract.
            //
            // Returns -2 (vs -1 for "corrupt file") so the layer loop in
            // sp_*_cache_load can distinguish "abort whole load" from
            // "skip this layer". Hash mismatch is a security boundary;
            // a corrupt single-layer file is recoverable best-effort.
            const char *strict_env = getenv("SP_DISK_HASH_STRICT");
            const int  warn_only   = strict_env && strict_env[0] == '0';
            if (!warn_only) {
                fprintf(stderr, "[sp-disk] refusing to load: hash mismatch is "
                        "fatal (set SP_DISK_HASH_STRICT=0 to load anyway)\n");
                return -2;
            }
            fprintf(stderr, "[sp-disk] SP_DISK_HASH_STRICT=0 — loading anyway\n");
        }
    }
    return 0;
}

// ── Shadow cache save/load ──────────────────────────────────────────

int sp_shadow_cache_save(const sp_shadow_cache_t *sc,
                         const char *prefix, int n_pos,
                         uint64_t model_hash) {
    if (!sc || !prefix) return -1;

    const int n_layer = sc->config.n_layers;
    const int n_head  = sc->config.n_heads_kv;

    for (int il = 0; il < n_layer; il++) {
        // Determine positions written for this layer
        int layer_pos = (n_pos > 0) ? n_pos : (sc->seq_len ? sc->seq_len[il] : 0);
        if (layer_pos <= 0) continue;

        // Helper macro for K and V — same band-major write pattern, parameterised
        // on which bands struct + cache pointer + path suffix.
        #define SAVE_KV_BAND_MAJOR(bc, cache_array, suffix, type_id)                         \
            do {                                                                              \
                char path[1024];                                                              \
                snprintf(path, sizeof(path), "%s.l%d." suffix ".vht2", prefix, il);          \
                FILE *f = fopen(path, "wb");                                                  \
                if (!f) {                                                                     \
                    fprintf(stderr, "[sp-disk] cannot write %s\n", path);                    \
                    return -1;                                                                \
                }                                                                             \
                const int n_b = (bc).n_bands;                                                 \
                if (n_b > 7) {                                                                \
                    /* Too many bands for v3 header (caps at 7). Fall back to v2. */         \
                    if (sp_write_cache_header(f, (bc).total_bytes, layer_pos, n_head,        \
                                               (type_id), model_hash) != 0) {                 \
                        fclose(f); return -1;                                                 \
                    }                                                                         \
                    for (int ih = 0; ih < n_head; ih++) {                                     \
                        int slot = il * n_head + ih;                                          \
                        const uint8_t *cache_ptr = (cache_array)[slot];                       \
                        for (int p = 0; p < layer_pos; p++) {                                 \
                            if (fwrite(cache_ptr + (size_t)p * (bc).total_bytes,             \
                                       (bc).total_bytes, 1, f) != 1) {                       \
                                fclose(f); return -1;                                         \
                            }                                                                 \
                        }                                                                     \
                    }                                                                         \
                } else {                                                                      \
                    /* v3 band-major path */                                                  \
                    int rec_off[8] = {0};                                                     \
                    int rec_bytes_band[8] = {0};                                              \
                    for (int b = 0; b < n_b; b++) {                                           \
                        rec_bytes_band[b] = sp_band_record_bytes(&(bc), b);                  \
                        rec_off[b + 1] = rec_off[b] + rec_bytes_band[b];                      \
                    }                                                                         \
                    uint32_t band_offs[7] = {0};                                              \
                    uint32_t off_in_file = 64;                                                \
                    for (int b = 0; b < n_b; b++) {                                           \
                        band_offs[b] = off_in_file;                                           \
                        off_in_file += (uint32_t)((size_t)rec_bytes_band[b] *                \
                                                  (size_t)n_head * (size_t)layer_pos);       \
                    }                                                                         \
                    if (sp_write_cache_header_v3(f, (bc).total_bytes, layer_pos, n_head,     \
                                                  (type_id), model_hash, n_b,                 \
                                                  band_offs, n_b) != 0) {                     \
                        fclose(f); return -1;                                                 \
                    }                                                                         \
                    for (int b = 0; b < n_b; b++) {                                           \
                        for (int ih = 0; ih < n_head; ih++) {                                 \
                            int slot = il * n_head + ih;                                      \
                            const uint8_t *cache_ptr = (cache_array)[slot];                   \
                            for (int p = 0; p < layer_pos; p++) {                             \
                                const uint8_t *src = cache_ptr +                              \
                                    (size_t)p * (bc).total_bytes + rec_off[b];               \
                                if (fwrite(src, rec_bytes_band[b], 1, f) != 1) {              \
                                    fclose(f); return -1;                                     \
                                }                                                             \
                            }                                                                 \
                        }                                                                     \
                    }                                                                         \
                }                                                                             \
                fclose(f);                                                                    \
            } while (0)

        SAVE_KV_BAND_MAJOR(sc->k_bands, sc->k_cache, "k", 0);
        SAVE_KV_BAND_MAJOR(sc->v_bands, sc->v_cache, "v", 0);

        #undef SAVE_KV_BAND_MAJOR
    }
    return 0;
}

// Internal helper: load one (layer, K-or-V) file. Handles v2 (per-vec
// interleaved) and v3 (band-major) layouts by sniffing the version.
// max_bands clamps how many bands of v3 files are read (others are
// zeroed in the cache); v2 files ignore max_bands and load fully.
// Returns: positive n_pos on success, 0 to skip layer, -1 on error,
// -2 on strict-mode hash mismatch.
static int sp_load_layer_kv(FILE *f, sp_shadow_cache_t *sc, int il,
                            int n_head, const sp_band_config_t *bc,
                            uint8_t **cache_array, uint64_t expected_hash,
                            int max_bands, const char *kv_label) {
    const int total_bytes = bc->total_bytes;

    int pstr, n_pos, n_hd, ctype, n_bands_disk = 0;
    uint32_t band_offs[7] = {0};
    int hrc = sp_read_cache_header_v3(f, &pstr, &n_pos, &n_hd, &ctype,
                                      expected_hash, &n_bands_disk, band_offs);
    if (hrc == -2) return -2;       // strict-mode hash mismatch
    if (hrc == -1) {                // bad magic / read fail
        fprintf(stderr, "[sp-disk] %s layer %d: bad header\n", kv_label, il);
        return 0;
    }
    if (hrc == 1) {
        // v2 fallback: file has the legacy per-vec layout. fseek back to
        // start and re-read with the v2 reader. v2 ignores max_bands —
        // it has to read all bytes per vec because they're interleaved.
        fseek(f, 0, SEEK_SET);
        int v2_hrc = sp_read_cache_header(f, &pstr, &n_pos, &n_hd, &ctype, expected_hash);
        if (v2_hrc == -2) return -2;
        if (v2_hrc != 0) {
            fprintf(stderr, "[sp-disk] %s layer %d: v2 header bad\n", kv_label, il);
            return 0;
        }
        if (pstr != total_bytes || n_hd != n_head) {
            fprintf(stderr, "[sp-disk] %s layer %d: v2 config mismatch\n",
                    kv_label, il);
            return 0;
        }
        for (int ih = 0; ih < n_head; ih++) {
            int slot = il * n_head + ih;
            uint8_t *cache_ptr = cache_array[slot];
            for (int p = 0; p < n_pos; p++) {
                if (fread(cache_ptr + (size_t)p * total_bytes, total_bytes, 1, f) != 1) {
                    fprintf(stderr, "[sp-disk] %s layer %d head %d: v2 read truncated\n",
                            kv_label, il, ih);
                    break;
                }
            }
        }
        return n_pos;
    }

    // v3 band-major path
    if (pstr != total_bytes || n_hd != n_head) {
        fprintf(stderr, "[sp-disk] %s layer %d: v3 config mismatch (disk pstr=%d n_hd=%d, "
                "expected pstr=%d n_hd=%d)\n",
                kv_label, il, pstr, n_hd, total_bytes, n_head);
        return 0;
    }
    if (n_bands_disk != bc->n_bands) {
        fprintf(stderr, "[sp-disk] %s layer %d: v3 n_bands mismatch (disk=%d expected=%d)\n",
                kv_label, il, n_bands_disk, bc->n_bands);
        return 0;
    }

    // Compute per-band record sizes + offsets within a packed vec.
    int rec_off[8] = {0};
    int rec_bytes_band[8] = {0};
    for (int b = 0; b < bc->n_bands; b++) {
        rec_bytes_band[b] = sp_band_record_bytes(bc, b);
        rec_off[b + 1] = rec_off[b] + rec_bytes_band[b];
    }

    // Clamp max_bands.
    int read_n = max_bands;
    if (read_n < 0) read_n = 0;
    if (read_n > bc->n_bands) read_n = bc->n_bands;

    // Zero the cache regions for bands [read_n, n_bands) so partial reads
    // don't leave stale data from a prior load. Only zero the per-band
    // segments — the bands we WILL read overwrite themselves.
    if (read_n < bc->n_bands) {
        for (int b = read_n; b < bc->n_bands; b++) {
            for (int ih = 0; ih < n_head; ih++) {
                int slot = il * n_head + ih;
                uint8_t *cache_ptr = cache_array[slot];
                for (int p = 0; p < n_pos; p++) {
                    memset(cache_ptr + (size_t)p * total_bytes + rec_off[b],
                           0, (size_t)rec_bytes_band[b]);
                }
            }
        }
    }

    // Read bands [0, read_n) from disk into the cache. Each band's region
    // on disk is contiguous: head 0 pos 0..n-1, head 1 pos 0..n-1, etc.
    for (int b = 0; b < read_n; b++) {
        if (b < 7) {
            // Use the recorded offset (matches how the writer laid it out)
            if (fseek(f, (long)band_offs[b], SEEK_SET) != 0) {
                fprintf(stderr, "[sp-disk] %s layer %d band %d: fseek failed\n",
                        kv_label, il, b);
                return 0;
            }
        }
        for (int ih = 0; ih < n_head; ih++) {
            int slot = il * n_head + ih;
            uint8_t *cache_ptr = cache_array[slot];
            for (int p = 0; p < n_pos; p++) {
                uint8_t *dst = cache_ptr + (size_t)p * total_bytes + rec_off[b];
                if (fread(dst, rec_bytes_band[b], 1, f) != 1) {
                    fprintf(stderr, "[sp-disk] %s layer %d band %d head %d: "
                            "read truncated at pos %d\n",
                            kv_label, il, b, ih, p);
                    break;
                }
            }
        }
    }

    return n_pos;
}

int sp_shadow_cache_load(sp_shadow_cache_t *sc,
                         const char *prefix,
                         uint64_t expected_hash) {
    return sp_shadow_cache_load_partial(sc, prefix, expected_hash, INT_MAX);
}

int sp_shadow_cache_load_partial(sp_shadow_cache_t *sc,
                                 const char *prefix,
                                 uint64_t expected_hash,
                                 int max_bands) {
    if (!sc || !prefix) return -1;

    const int n_layer = sc->config.n_layers;
    const int n_head  = sc->config.n_heads_kv;
    int max_loaded = 0;

    for (int il = 0; il < n_layer; il++) {
        // Load K
        {
            char path[1024];
            snprintf(path, sizeof(path), "%s.l%d.k.vht2", prefix, il);
            FILE *f = fopen(path, "rb");
            if (!f) continue;
            int n_pos = sp_load_layer_kv(f, sc, il, n_head, &sc->k_bands,
                                          sc->k_cache, expected_hash,
                                          max_bands, "K");
            fclose(f);
            if (n_pos == -2) return -1;     // strict hash mismatch — abort
            if (n_pos > 0) {
                if (sc->seq_len) sc->seq_len[il] = n_pos;
                if (n_pos > max_loaded) max_loaded = n_pos;
            }
        }

        // Load V
        {
            char path[1024];
            snprintf(path, sizeof(path), "%s.l%d.v.vht2", prefix, il);
            FILE *f = fopen(path, "rb");
            if (!f) continue;
            int n_pos = sp_load_layer_kv(f, sc, il, n_head, &sc->v_bands,
                                          sc->v_cache, expected_hash,
                                          max_bands, "V");
            fclose(f);
            if (n_pos == -2) return -1;
        }
    }
    return max_loaded;
}

// ── Sqfree cache save/load ──────────────────────────────────────────
// Same binary format; packed_stride is per-position bytes from sqfree layout.

int sp_sqfree_cache_save(const sp_sqfree_cache_t *sc,
                         const char *prefix, int n_pos,
                         uint64_t model_hash) {
    if (!sc || !prefix) return -1;

    const int n_layer  = sc->config.head_dim > 0 ? sc->config.n_layers : 0;
    const int n_head   = sc->config.n_heads_kv;
    const int n_res    = sc->mask.n_res;
    // Per-position storage: banded skeleton + residual + magnitude + optional spinor
    const int k_bytes  = sc->k_bands.total_bytes
                       + (n_res * sc->residual_bits + 7) / 8
                       + 4  // magnitude (float)
                       + (sc->use_spinor ? (n_res + 7) / 8 : 0);
    const int v_bytes  = k_bytes;  // Same layout for V

    for (int il = 0; il < n_layer; il++) {
        int layer_pos = (n_pos > 0) ? n_pos : 0;
        if (layer_pos <= 0) continue;

        // K
        {
            char path[1024];
            snprintf(path, sizeof(path), "%s.l%d.k.vht2", prefix, il);
            FILE *f = fopen(path, "wb");
            if (!f) { fprintf(stderr, "[sp-disk] cannot write %s\n", path); return -1; }
            sp_write_cache_header(f, k_bytes, layer_pos, n_head, 1, model_hash);
            for (int ih = 0; ih < n_head; ih++) {
                int slot = il * n_head + ih;
                const uint8_t *data = sc->k_cache[slot];
                fwrite(data, (size_t)k_bytes * layer_pos, 1, f);
            }
            fclose(f);
        }
        // V
        {
            char path[1024];
            snprintf(path, sizeof(path), "%s.l%d.v.vht2", prefix, il);
            FILE *f = fopen(path, "wb");
            if (!f) { fprintf(stderr, "[sp-disk] cannot write %s\n", path); return -1; }
            sp_write_cache_header(f, v_bytes, layer_pos, n_head, 1, model_hash);
            for (int ih = 0; ih < n_head; ih++) {
                int slot = il * n_head + ih;
                const uint8_t *data = sc->v_cache[slot];
                fwrite(data, (size_t)v_bytes * layer_pos, 1, f);
            }
            fclose(f);
        }
    }
    return 0;
}

int sp_sqfree_cache_load(sp_sqfree_cache_t *sc,
                         const char *prefix,
                         uint64_t expected_hash) {
    if (!sc || !prefix) return -1;

    const int n_layer = sc->config.n_layers;
    const int n_head  = sc->config.n_heads_kv;
    const int n_res   = sc->mask.n_res;
    const int k_bytes = sc->k_bands.total_bytes
                      + (n_res * sc->residual_bits + 7) / 8
                      + 4
                      + (sc->use_spinor ? (n_res + 7) / 8 : 0);
    int max_loaded = 0;

    for (int il = 0; il < n_layer; il++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s.l%d.k.vht2", prefix, il);
        FILE *f = fopen(path, "rb");
        if (!f) continue;

        int pstr, n_pos, n_hd, ctype;
        int hrc = sp_read_cache_header(f, &pstr, &n_pos, &n_hd, &ctype, expected_hash);
        if (hrc == -2) { fclose(f); return -1; }   // strict-mode hash mismatch — abort
        if (hrc != 0 || pstr != k_bytes || n_hd != n_head) {
            fprintf(stderr, "[sp-disk] sqfree K layer %d: config mismatch\n", il);
            fclose(f); continue;
        }
        for (int ih = 0; ih < n_head; ih++) {
            int slot = il * n_head + ih;
            fread(sc->k_cache[slot], (size_t)k_bytes * n_pos, 1, f);
        }
        if (n_pos > max_loaded) max_loaded = n_pos;
        fclose(f);

        // V
        snprintf(path, sizeof(path), "%s.l%d.v.vht2", prefix, il);
        f = fopen(path, "rb");
        if (!f) continue;
        hrc = sp_read_cache_header(f, &pstr, &n_pos, &n_hd, &ctype, expected_hash);
        if (hrc == -2) { fclose(f); return -1; }
        if (hrc != 0) {
            fclose(f); continue;
        }
        for (int ih = 0; ih < n_head; ih++) {
            int slot = il * n_head + ih;
            fread(sc->v_cache[slot], (size_t)pstr * n_pos, 1, f);
        }
        fclose(f);
    }
    return max_loaded;
}

// ── Hierarchical cache save/load ────────────────────────────────────
// Saves: skeleton bands + residual per position, PLUS the W predictor matrices.
// W matrices go in a separate file: {prefix}.hier_w.bin

int sp_hier_cache_save(const sp_hier_cache_t *sc,
                       const char *prefix, int n_pos,
                       uint64_t model_hash) {
    if (!sc || !prefix) return -1;

    const int n_layer   = sc->config.n_layers;
    const int n_head    = sc->config.n_heads_kv;
    const int k_bpp     = sc->k_bytes_per_pos;
    const int v_bpp     = sc->v_bytes_per_pos;
    const int n_target  = sc->predictors[0].n_target;
    const int n_skel    = sc->predictors[0].n_skeleton;

    for (int il = 0; il < n_layer; il++) {
        int layer_pos = (n_pos > 0) ? n_pos : 0;
        if (layer_pos <= 0) continue;

        // K
        {
            char path[1024];
            snprintf(path, sizeof(path), "%s.l%d.k.vht2", prefix, il);
            FILE *f = fopen(path, "wb");
            if (!f) return -1;
            sp_write_cache_header(f, k_bpp, layer_pos, n_head, 2, model_hash);
            for (int ih = 0; ih < n_head; ih++) {
                int slot = il * n_head + ih;
                fwrite(sc->k_cache[slot], (size_t)k_bpp * layer_pos, 1, f);
            }
            fclose(f);
        }
        // V
        {
            char path[1024];
            snprintf(path, sizeof(path), "%s.l%d.v.vht2", prefix, il);
            FILE *f = fopen(path, "wb");
            if (!f) return -1;
            sp_write_cache_header(f, v_bpp, layer_pos, n_head, 2, model_hash);
            for (int ih = 0; ih < n_head; ih++) {
                int slot = il * n_head + ih;
                fwrite(sc->v_cache[slot], (size_t)v_bpp * layer_pos, 1, f);
            }
            fclose(f);
        }
    }

    // Save W predictor matrices: one file for all slots
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s.hier_w.bin", prefix);
        FILE *f = fopen(path, "wb");
        if (!f) return -1;

        const int n_slots = n_layer * n_head;
        // Header: magic + n_slots + n_skeleton + n_target
        uint32_t whdr[4] = {SP_CACHE_MAGIC, (uint32_t)n_slots,
                            (uint32_t)n_skel, (uint32_t)n_target};
        fwrite(whdr, sizeof(whdr), 1, f);

        // W matrices stored as fp16 (uint16_t), [n_target × n_skeleton] per slot
        const size_t w_size = (size_t)n_target * n_skel * sizeof(uint16_t);
        for (int s = 0; s < n_slots; s++) {
            if (sc->predictors[s].W) {
                fwrite(sc->predictors[s].W, w_size, 1, f);
            } else {
                // Uncalibrated slot — write zeros
                uint16_t *zeros = (uint16_t *)calloc(w_size, 1);
                fwrite(zeros, w_size, 1, f);
                free(zeros);
            }
        }
        fclose(f);
    }

    return 0;
}

int sp_hier_cache_load(sp_hier_cache_t *sc,
                       const char *prefix,
                       uint64_t expected_hash) {
    if (!sc || !prefix) return -1;

    const int n_layer  = sc->config.n_layers;
    const int n_head   = sc->config.n_heads_kv;
    const int k_bpp    = sc->k_bytes_per_pos;
    const int v_bpp    = sc->v_bytes_per_pos;
    const int n_target = sc->predictors[0].n_target;
    const int n_skel   = sc->predictors[0].n_skeleton;
    int max_loaded = 0;

    for (int il = 0; il < n_layer; il++) {
        // K
        {
            char path[1024];
            snprintf(path, sizeof(path), "%s.l%d.k.vht2", prefix, il);
            FILE *f = fopen(path, "rb");
            if (!f) continue;

            int pstr, n_pos, n_hd, ctype;
            int hrc = sp_read_cache_header(f, &pstr, &n_pos, &n_hd, &ctype, expected_hash);
            if (hrc == -2) { fclose(f); return -1; }   // strict-mode hash mismatch — abort
            if (hrc != 0 || pstr != k_bpp || n_hd != n_head) {
                fclose(f); continue;
            }
            for (int ih = 0; ih < n_head; ih++) {
                int slot = il * n_head + ih;
                fread(sc->k_cache[slot], (size_t)k_bpp * n_pos, 1, f);
            }
            if (n_pos > max_loaded) max_loaded = n_pos;
            fclose(f);
        }
        // V
        {
            char path[1024];
            snprintf(path, sizeof(path), "%s.l%d.v.vht2", prefix, il);
            FILE *f = fopen(path, "rb");
            if (!f) continue;

            int pstr, n_pos, n_hd, ctype;
            int hrc = sp_read_cache_header(f, &pstr, &n_pos, &n_hd, &ctype, expected_hash);
            if (hrc == -2) { fclose(f); return -1; }
            if (hrc != 0 || pstr != v_bpp || n_hd != n_head) {
                fclose(f); continue;
            }
            for (int ih = 0; ih < n_head; ih++) {
                int slot = il * n_head + ih;
                fread(sc->v_cache[slot], (size_t)v_bpp * n_pos, 1, f);
            }
            fclose(f);
        }
    }

    // Load W predictor matrices
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s.hier_w.bin", prefix);
        FILE *f = fopen(path, "rb");
        if (f) {
            uint32_t whdr[4];
            if (fread(whdr, sizeof(whdr), 1, f) == 1
                && whdr[0] == SP_CACHE_MAGIC
                && (int)whdr[2] == n_skel
                && (int)whdr[3] == n_target) {

                const int n_slots = n_layer * n_head;
                const size_t w_size = (size_t)n_target * n_skel * sizeof(uint16_t);
                for (int s = 0; s < n_slots && s < (int)whdr[1]; s++) {
                    if (sc->predictors[s].W) {
                        fread(sc->predictors[s].W, w_size, 1, f);
                        sc->predictors[s].calibrated = true;
                    } else {
                        fseek(f, (long)w_size, SEEK_CUR);  // Skip uncalibrated
                    }
                }
            }
            fclose(f);
        }
    }

    return max_loaded;
}
