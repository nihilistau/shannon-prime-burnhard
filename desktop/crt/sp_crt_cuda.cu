// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// CRT Multi-GPU — CUDA Modular MatMul Kernels
//
// Two kernel variants:
//   1. Mersenne (M1 = 2^31-1): bit-shift reduction in the inner loop
//   2. Generic modular (M2 = 2^31-19): uint64 accumulation + final %
//
// Both use tiled shared-memory strategy (TILE_DIM × TILE_DIM) for
// memory coalescing. The uint32 datatype means we get 2× the elements
// per cacheline vs fp32 (same bit width, but the reduction overhead
// is compensated by the CRT parallelism across two GPUs).

#include "sp_crt.h"
#include <cuda_runtime.h>
#include <stdint.h>

#define CRT_TILE_DIM 16

// ============================================================================
// Device-side Mersenne reduction
// ============================================================================

__device__ __forceinline__ uint32_t d_mersenne_reduce(uint64_t x) {
    // Two-round fold for inputs up to 2^63 (zero-centered quantization
    // produces ring values near M1, so products reach M1² ≈ 2^62).
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
// Kernel: Tiled Mersenne matmul — C = (A × B) mod (2^31 - 1)
// ============================================================================
//
// A: [M × K] uint32, B: [K × N] uint32, C: [M × N] uint32
// Uses shared memory tiles of TILE_DIM × TILE_DIM.
// Mersenne reduction every TILE_DIM accumulations to prevent overflow.

__global__ void kernel_matmul_mersenne(const uint32_t * __restrict__ A,
                                        const uint32_t * __restrict__ B,
                                        uint32_t * __restrict__ C,
                                        int M, int N, int K) {
    __shared__ uint32_t sA[CRT_TILE_DIM][CRT_TILE_DIM];
    __shared__ uint32_t sB[CRT_TILE_DIM][CRT_TILE_DIM];

    int row = blockIdx.y * CRT_TILE_DIM + threadIdx.y;
    int col = blockIdx.x * CRT_TILE_DIM + threadIdx.x;

    uint64_t acc = 0;
    int n_tiles = (K + CRT_TILE_DIM - 1) / CRT_TILE_DIM;

    for (int t = 0; t < n_tiles; t++) {
        int ak = t * CRT_TILE_DIM + threadIdx.x;
        int bk = t * CRT_TILE_DIM + threadIdx.y;

        // Load tile from A
        sA[threadIdx.y][threadIdx.x] =
            (row < M && ak < K) ? A[row * K + ak] : 0;
        // Load tile from B
        sB[threadIdx.y][threadIdx.x] =
            (bk < K && col < N) ? B[bk * N + col] : 0;

        __syncthreads();

        // Accumulate dot product for this tile, reducing every iteration.
        // With zero-centered quantization, ring values can be near M1,
        // so each product can reach M1² ≈ 2^62. Must reduce per-element.
        #pragma unroll
        for (int k = 0; k < CRT_TILE_DIM; k++) {
            acc += (uint64_t)sA[threadIdx.y][k] * sB[k][threadIdx.x];
            acc = d_mersenne_reduce(acc);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = d_mersenne_reduce(acc);
    }
}

// ============================================================================
// Kernel: Tiled generic modular matmul — C = (A × B) mod m
// ============================================================================
//
// Same tiled structure but uses standard % for non-Mersenne modulus.

__global__ void kernel_matmul_mod(const uint32_t * __restrict__ A,
                                   const uint32_t * __restrict__ B,
                                   uint32_t * __restrict__ C,
                                   int M, int N, int K,
                                   uint64_t modulus) {
    __shared__ uint32_t sA[CRT_TILE_DIM][CRT_TILE_DIM];
    __shared__ uint32_t sB[CRT_TILE_DIM][CRT_TILE_DIM];

    int row = blockIdx.y * CRT_TILE_DIM + threadIdx.y;
    int col = blockIdx.x * CRT_TILE_DIM + threadIdx.x;

    uint64_t acc = 0;
    int n_tiles = (K + CRT_TILE_DIM - 1) / CRT_TILE_DIM;

    for (int t = 0; t < n_tiles; t++) {
        int ak = t * CRT_TILE_DIM + threadIdx.x;
        int bk = t * CRT_TILE_DIM + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] =
            (row < M && ak < K) ? A[row * K + ak] : 0;
        sB[threadIdx.y][threadIdx.x] =
            (bk < K && col < N) ? B[bk * N + col] : 0;

        __syncthreads();

        // Reduce every iteration — products of ring values near M can reach M² ≈ 2^62.
        #pragma unroll
        for (int k = 0; k < CRT_TILE_DIM; k++) {
            acc += (uint64_t)sA[threadIdx.y][k] * sB[k][threadIdx.x];
            acc %= modulus;
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = (uint32_t)(acc % modulus);
    }
}

// ============================================================================
// Kernel: Float-to-residue quantization
// ============================================================================

__global__ void kernel_quantize(const float * __restrict__ input,
                                 uint32_t * __restrict__ output,
                                 int n,
                                 double scale, int64_t zero_point,
                                 uint64_t modulus) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int64_t ival = (int64_t)((double)input[idx] * scale + 0.5) + zero_point;
    if (ival < 0) ival = 0;
    output[idx] = (uint32_t)((uint64_t)ival % modulus);
}

// ============================================================================
// Host launchers
// ============================================================================

extern "C" void sp_crt_cuda_matmul_mersenne(const uint32_t *d_A,
                                             const uint32_t *d_B,
                                             uint32_t *d_C,
                                             int M, int N, int K,
                                             void *stream) {
    dim3 block(CRT_TILE_DIM, CRT_TILE_DIM);
    dim3 grid((N + CRT_TILE_DIM - 1) / CRT_TILE_DIM,
              (M + CRT_TILE_DIM - 1) / CRT_TILE_DIM);

    kernel_matmul_mersenne<<<grid, block, 0, (cudaStream_t)stream>>>(
        d_A, d_B, d_C, M, N, K);
}

extern "C" void sp_crt_cuda_matmul_mod(const uint32_t *d_A,
                                        const uint32_t *d_B,
                                        uint32_t *d_C,
                                        int M, int N, int K,
                                        uint64_t modulus,
                                        void *stream) {
    dim3 block(CRT_TILE_DIM, CRT_TILE_DIM);
    dim3 grid((N + CRT_TILE_DIM - 1) / CRT_TILE_DIM,
              (M + CRT_TILE_DIM - 1) / CRT_TILE_DIM);

    kernel_matmul_mod<<<grid, block, 0, (cudaStream_t)stream>>>(
        d_A, d_B, d_C, M, N, K, modulus);
}

// ============================================================================
// Zero-centered symmetric quantization kernel (GPU-side)
// ============================================================================
//
// Maps positive floats → small ring integers, negative → M - |q|.
// This is the GPU-side equivalent of sp_crt_quantize_symmetric in sp_crt.c.

__global__ void kernel_quantize_symmetric(const float * __restrict__ input,
                                           uint32_t * __restrict__ output,
                                           int n,
                                           double scale,
                                           uint64_t modulus) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double qd = (double)input[idx] * scale;
    int64_t q = (qd >= 0.0) ? (int64_t)(qd + 0.5) : -(int64_t)(-qd + 0.5);

    if (q >= 0) {
        output[idx] = (uint32_t)((uint64_t)q % modulus);
    } else {
        uint64_t pos = (uint64_t)(-q);
        pos %= modulus;
        output[idx] = (pos == 0) ? 0 : (uint32_t)(modulus - pos);
    }
}

extern "C" void sp_crt_cuda_quantize_symmetric(const float *d_input,
                                                 uint32_t *d_output,
                                                 int n,
                                                 double scale,
                                                 uint64_t modulus,
                                                 void *stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    kernel_quantize_symmetric<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
        d_input, d_output, n, scale, modulus);
}

// ============================================================================
// CUDA memory helpers for CRT GPU dispatch
// ============================================================================

extern "C" void* sp_crt_cuda_alloc(size_t bytes) {
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    return (err == cudaSuccess) ? ptr : NULL;
}

extern "C" void sp_crt_cuda_free(void *ptr) {
    if (ptr) cudaFree(ptr);
}

extern "C" int sp_crt_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes, void *stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes,
                                       cudaMemcpyHostToDevice, (cudaStream_t)stream);
    return (err == cudaSuccess) ? 0 : -1;
}

extern "C" int sp_crt_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes, void *stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes,
                                       cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    return (err == cudaSuccess) ? 0 : -1;
}

extern "C" int sp_crt_cuda_stream_sync(void *stream) {
    cudaError_t err = cudaStreamSynchronize((cudaStream_t)stream);
    return (err == cudaSuccess) ? 0 : -1;
}

extern "C" void* sp_crt_cuda_stream_create(void) {
    cudaStream_t s;
    cudaError_t err = cudaStreamCreate(&s);
    return (err == cudaSuccess) ? (void*)s : NULL;
}

extern "C" void sp_crt_cuda_stream_destroy(void *stream) {
    if (stream) cudaStreamDestroy((cudaStream_t)stream);
}

// ============================================================================
// Host launchers (existing)
// ============================================================================

extern "C" void sp_crt_cuda_quantize(const float *d_input,
                                      uint32_t *d_output,
                                      int n,
                                      double scale, int64_t zero_point,
                                      uint64_t modulus,
                                      void *stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    kernel_quantize<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
        d_input, d_output, n, scale, zero_point, modulus);
}
