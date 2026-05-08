// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#ifndef SP_SHOR_H
#define SP_SHOR_H

// ============================================================================
// Shor-Resonant Transformer — Heterogeneous Inference Engine
// ============================================================================
//
// Architecture: Beast Canyon (i9-12900 + RTX 2060 12GB + Intel UHD 770)
//
// The Shor-Resonant engine replaces standard dot-product attention with
// complex-valued Phase Resonance scoring. Two compute devices execute
// concurrently in a ping-pong pattern:
//
//   Stream A (RTX 2060, CUDA):
//     - CRT ring M1 (Mersenne 2^31-1) modular matmuls for weight projections
//     - Shor interference kernel for Expert #1 attention scoring
//     - Primary matmul engine for large weight projections
//
//   Stream B (Intel UHD 770, Level Zero — future):
//     - CRT ring M2 (2^31-19) modular matmuls
//     - Shor interference kernel for Expert #2 attention scoring
//     - Secondary matmul via Vulkan compute shaders
//
//   Host (i9 AVX-512):
//     - Garner CRT recombination (dual-ring → full-precision)
//     - Optane M10 expert paging + AVX-512 dequant "Shredder"
//     - Free-Energy homeostat (SVD spectral gap monitoring)
//
//   Sidecar (S22U Hexagon DSP, USB-C — optional):
//     - Zeta-Spectral phase generation (log-phase trig offloaded)
//     - HVX-vectorized Mersenne VHT2 index computation
//
// Data flow per layer:
//   1. [Optane/RAM] Expert weights → [i9 AVX-512] dequant → [LLC 24MB]
//   2. [i9] Calibrate CRT quantization from weight range
//   3. [RTX 2060] CRT ring M1 matmul: C₀ = (A × B) mod M1
//      [Intel UHD] CRT ring M2 matmul: C₁ = (A × B) mod M2   (concurrent)
//   4. [i9] Garner reconstruct: C = CRT(C₀, C₁)
//   5. [RTX 2060] Shor interference: Energy = |Q·conj(K)|² mod M1
//   6. [i9] Softmax on energy scores → attention weights
//   7. Repeat for next layer (pipelined with step 1 of layer N+1)

#include "sp_crt.h"
#include "sp_crt_dispatch.h"

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Zeta zeros table — first 256 non-trivial zeros of the Riemann zeta function
// ============================================================================
//
// γ_n = Im(ρ_n) where ζ(1/2 + iγ_n) = 0. These are the phase frequencies
// for the Zeta-Spectral Embedding. The table covers up to d_model = 256;
// for wider models, zeros are computed on-the-fly via the Gram's law
// approximation: γ_n ≈ 2π(n-7/8) / ln(n/(2πe)).
//
// The first few are hardcoded from LMFDB (verified to 40 digits):

#define SP_SHOR_MAX_ZETA_ZEROS 256

extern const float sp_shor_zeta_zeros[SP_SHOR_MAX_ZETA_ZEROS];

// ============================================================================
// Shor engine context
// ============================================================================

typedef struct {
    // CRT dispatch layer — handles weight matmuls across dual rings
    sp_crt_dispatch_t   crt_dispatch;

    // Interference kernel state
    float2*  d_Q_complex;       // [max_seq, max_dim] device complex Q
    float2*  d_K_complex;       // [max_seq, max_dim] device complex K
    uint32_t* d_energy_int;     // [max_seq, max_seq] Mersenne-folded energy
    float*   d_energy_f32;      // [max_seq, max_seq] fp32 energy for softmax

    // Zeta zeros on device (for real→complex embedding)
    float*   d_zeta_zeros;      // [max_dim] on device
    int*     d_positions;       // [max_seq] position ids on device

    // Dimensions
    int max_seq;
    int max_dim;

    // Device streams (owned, or shared with CRT dispatch)
    void* stream_primary;       // RTX 2060 / CUDA stream 0
    void* stream_secondary;     // Intel UHD / Level Zero queue (future)

    // Phase: which components are ready
    int crt_ready;              // CRT matmul dispatch is operational
    int interference_ready;     // Shor interference kernel is operational
    int zeta_ready;             // Zeta zeros uploaded to device

    // Stats
    int n_interference_calls;
    double interference_total_us;
} sp_shor_engine_t;

// ============================================================================
// Lifecycle
// ============================================================================

// Initialise the Shor engine. max_dim covers both head_dim and n_embd
// (the larger of the two). max_seq is the context window.
int  sp_shor_init(sp_shor_engine_t* eng, int max_seq, int max_dim);
void sp_shor_free(sp_shor_engine_t* eng);

// Upload zeta zeros to device. Call once at init. If dim > 256,
// remaining zeros are approximated via Gram's law.
int  sp_shor_upload_zeta_zeros(sp_shor_engine_t* eng, int dim);

// ============================================================================
// Operations
// ============================================================================

// Convert real Q/K to complex via Zeta-Spectral embedding.
// real_qk: [seq, dim] fp32 on device
// positions: [seq] int32 on device
// out_complex: [seq, dim] float2 on device
int sp_shor_embed_complex(sp_shor_engine_t* eng,
                           const float* d_real_qk,
                           const int* d_positions,
                           float2* d_complex_out,
                           int seq, int dim);

// Compute Shor interference attention scores.
// Q_complex, K_complex: [seq_q/seq_k, dim] float2 on device
// out_energy: [seq_q, seq_k] fp32 on device (for softmax)
// out_energy_int: [seq_q, seq_k] uint32 on device (Mersenne, optional)
int sp_shor_attention_energy(sp_shor_engine_t* eng,
                              const float2* d_Q, const float2* d_K,
                              float* d_out_energy_f32,
                              uint32_t* d_out_energy_int,
                              int seq_q, int seq_k, int dim);

// Full pipeline: real Q/K → complex embed → interference → fp32 scores.
// This is the drop-in replacement for the Q·K^T attention scoring step.
int sp_shor_attention_scores(sp_shor_engine_t* eng,
                              const float* d_Q_real,
                              const float* d_K_real,
                              const int* d_positions_q,
                              const int* d_positions_k,
                              float* d_scores_out,
                              int seq_q, int seq_k, int dim);

// Print engine statistics.
void sp_shor_stats(const sp_shor_engine_t* eng);

// ============================================================================
// CUDA kernel launchers (defined in sp_shor_kernel.cu)
// ============================================================================

void sp_shor_interference(const float2* d_Q, const float2* d_K,
                           uint32_t* d_Energy, float* d_Energy_f32,
                           int seq_q, int seq_k, int dim,
                           float energy_scale, void* stream);

void sp_shor_real_to_complex(const float* d_real, const int* d_positions,
                              const float* d_zeta_zeros, float2* d_complex,
                              int seq, int dim, void* stream);

#ifdef __cplusplus
}
#endif

#endif // SP_SHOR_H
