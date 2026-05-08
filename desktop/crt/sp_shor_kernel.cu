// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

// ============================================================================
// Shor-Resonant Interference Kernel — Phase-Aligned Attention Scoring
// ============================================================================
//
// Replaces the standard Q·K^T dot product with a complex interference pattern.
// Instead of scalar similarity, we compute the *resonant energy* between query
// and key vectors treated as complex-valued signals on the unit circle.
//
// The mechanism:
//   1. Q and K vectors are complex (float2): real + imaginary components
//   2. For each (i, j) pair, compute the complex conjugate inner product:
//        Z_ij = sum_n Q[i,n] * conj(K[j,n])
//   3. The "attention energy" is |Z_ij|^2 (interference magnitude)
//   4. Fold through the Mersenne prime for modular arithmetic downstream
//
// Constructive interference (phases align) → high energy → strong attention
// Destructive interference (phases cancel) → low energy → weak attention
//
// This kernel targets the RTX 2060 (Beast Canyon stream_0). The Intel Xe
// iGPU runs the second expert's interference via Level Zero (future).
//
// Hardware note: RTX 2060 has dedicated int32 units alongside FP32 cores,
// making the Mersenne reduction essentially free when interleaved with
// the floating-point accumulation.

#include "sp_crt.h"   // for SP_CRT_M1, d_mersenne_reduce is local
#include <cuda_runtime.h>
#include <stdint.h>

#define SHOR_TILE_DIM 16

// ============================================================================
// Device-side Mersenne reduction (same as sp_crt_cuda.cu)
// ============================================================================

__device__ __forceinline__ uint32_t shor_mersenne_reduce(uint64_t x) {
    uint64_t lo = x & 0x7FFFFFFFULL;
    uint64_t hi = x >> 31;
    uint64_t r = lo + hi;
    if (r >= 0x80000000ULL) {
        lo = r & 0x7FFFFFFFULL;
        hi = r >> 31;
        r = lo + hi;
    }
    return (uint32_t)(r >= SP_CRT_M1 ? r - SP_CRT_M1 : r);
}

// ============================================================================
// Kernel: Shor Interference Attention — complex conjugate energy scoring
// ============================================================================
//
// Q: [seq_q, dim] complex (stored as float2*, so stride = dim float2's)
// K: [seq_k, dim] complex (stored as float2*)
// Energy: [seq_q, seq_k] uint32 — Mersenne-folded interference magnitude
//
// Each thread computes one (i, j) attention score by accumulating the
// complex conjugate product across all dim elements, then folding the
// magnitude through the Mersenne prime.
//
// Tiled shared memory for coalesced access to both Q and K.

__global__ void kernel_shor_interference(
    const float2* __restrict__ Q,       // [seq_q, dim]
    const float2* __restrict__ K,       // [seq_k, dim]
    uint32_t* __restrict__ Energy,      // [seq_q, seq_k]
    float* __restrict__ Energy_f32,     // [seq_q, seq_k] fp32 magnitude (for softmax path)
    int seq_q, int seq_k, int dim,
    float energy_scale)                 // fixed-point scaling (1e6 default)
{
    __shared__ float2 sQ[SHOR_TILE_DIM][SHOR_TILE_DIM];
    __shared__ float2 sK[SHOR_TILE_DIM][SHOR_TILE_DIM];

    int i = blockIdx.y * SHOR_TILE_DIM + threadIdx.y;   // query index
    int j = blockIdx.x * SHOR_TILE_DIM + threadIdx.x;   // key index

    float sum_re = 0.0f;
    float sum_im = 0.0f;

    int n_tiles = (dim + SHOR_TILE_DIM - 1) / SHOR_TILE_DIM;

    for (int t = 0; t < n_tiles; t++) {
        int qd = t * SHOR_TILE_DIM + threadIdx.x;   // Q's dim index
        int kd = t * SHOR_TILE_DIM + threadIdx.y;   // K's dim index

        // Load Q tile
        if (i < seq_q && qd < dim)
            sQ[threadIdx.y][threadIdx.x] = Q[i * dim + qd];
        else
            sQ[threadIdx.y][threadIdx.x] = make_float2(0.0f, 0.0f);

        // Load K tile
        if (j < seq_k && kd < dim)
            sK[threadIdx.y][threadIdx.x] = K[j * dim + kd];
        else
            sK[threadIdx.y][threadIdx.x] = make_float2(0.0f, 0.0f);

        __syncthreads();

        // Accumulate complex conjugate product:
        // Z += Q[i,n] * conj(K[j,n])
        //    = (q.x + q.y*i) * (k.x - k.y*i)
        //    = (q.x*k.x + q.y*k.y) + (q.y*k.x - q.x*k.y)*i
        #pragma unroll
        for (int n = 0; n < SHOR_TILE_DIM; n++) {
            float2 q = sQ[threadIdx.y][n];
            float2 k = sK[threadIdx.x][n];

            sum_re += (q.x * k.x + q.y * k.y);   // real part
            sum_im += (q.y * k.x - q.x * k.y);   // imaginary part
        }

        __syncthreads();
    }

    if (i < seq_q && j < seq_k) {
        // Magnitude squared: |Z|^2 = re^2 + im^2
        float mag_sq = sum_re * sum_re + sum_im * sum_im;

        // Write fp32 magnitude for the softmax attention path
        if (Energy_f32)
            Energy_f32[i * seq_k + j] = mag_sq;

        // Mersenne fold for the prime-field path
        if (Energy) {
            uint64_t scaled = (uint64_t)(mag_sq * energy_scale);
            Energy[i * seq_k + j] = shor_mersenne_reduce(scaled);
        }
    }
}

// ============================================================================
// Kernel: Real → Complex embedding (Zeta-Spectral phase generation)
// ============================================================================
//
// Converts real-valued Q/K vectors to complex by applying position-dependent
// phase rotations. Each dimension n gets phase exp(i * gamma_n * ln(pos)),
// where gamma_n are the first `dim` non-trivial zeta zeros (provided as input).
//
// This is the "Zeta-Spectral Embedding" — the S22U DSP can compute these
// phases and stream them to the host, but for Beast Canyon we compute
// on the GPU directly.

__global__ void kernel_real_to_complex(
    const float* __restrict__ real_vec,     // [seq, dim] real input (Q or K)
    const int*   __restrict__ positions,    // [seq] position ids
    const float* __restrict__ zeta_zeros,   // [dim] imaginary parts of zeta zeros
    float2* __restrict__ complex_vec,       // [seq, dim] complex output
    int seq, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / dim;   // sequence index
    int n = idx % dim;   // dimension index

    if (s >= seq || n >= dim) return;

    float val = real_vec[s * dim + n];
    float ln_pos = logf((float)(positions[s] + 1));   // ln(pos+1) to avoid ln(0)
    float phase = zeta_zeros[n] * ln_pos;

    // Embed: complex = val * exp(i * phase) = val * (cos(phase) + i*sin(phase))
    float cp, sp2;
    sincosf(phase, &sp2, &cp);

    complex_vec[s * dim + n] = make_float2(val * cp, val * sp2);
}

// ============================================================================
// Host launchers
// ============================================================================

extern "C" void sp_shor_interference(
    const float2* d_Q, const float2* d_K,
    uint32_t* d_Energy, float* d_Energy_f32,
    int seq_q, int seq_k, int dim,
    float energy_scale,
    void* stream)
{
    dim3 block(SHOR_TILE_DIM, SHOR_TILE_DIM);
    dim3 grid((seq_k + SHOR_TILE_DIM - 1) / SHOR_TILE_DIM,
              (seq_q + SHOR_TILE_DIM - 1) / SHOR_TILE_DIM);

    kernel_shor_interference<<<grid, block, 0, (cudaStream_t)stream>>>(
        d_Q, d_K, d_Energy, d_Energy_f32,
        seq_q, seq_k, dim, energy_scale);
}

extern "C" void sp_shor_real_to_complex(
    const float* d_real, const int* d_positions,
    const float* d_zeta_zeros, float2* d_complex,
    int seq, int dim,
    void* stream)
{
    int total = seq * dim;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    kernel_real_to_complex<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
        d_real, d_positions, d_zeta_zeros, d_complex, seq, dim);
}
