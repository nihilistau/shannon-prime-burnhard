// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#ifndef SP_CRT_H
#define SP_CRT_H

// ============================================================================
// CRT Multi-GPU Tensor Splitting
// ============================================================================
//
// Chinese Remainder Theorem parallelism for heterogeneous multi-GPU inference.
// Instead of splitting *layers* across GPUs (traditional tensor parallelism),
// we split the *number space*: each GPU computes the full matmul in a different
// residue ring, then the host recombines via Garner's algorithm.
//
// This eliminates all inter-GPU communication during compute. The only sync
// point is the host-side recombination after both GPUs complete.
//
// Moduli:
//   M1 = 2^31 - 1  = 2,147,483,647  (Mersenne prime — shift-and-add reduction)
//   M2 = 2^31 - 19 = 2,147,483,629  (largest prime < 2^31 coprime to M1)
//
// Combined range: M1 × M2 ≈ 4.6 × 10^18 — enough for 16-bit weight
// accumulation across 4096-dim hidden layers without overflow.
//
// The Garner constant C = M1^{-1} mod M2 = 2,028,178,983 is precomputed.
//
// Integration:
//   1. Quantize fp16 weights to scaled integers in [0, M_i)
//   2. Each GPU runs: C_i = (A × B) mod M_i  (standard matmul + reduction)
//   3. Host recombines: X = a1 + M1 × ((a2 - a1) × C mod M2)
//   4. Rescale back to floating point
//
// The matmul kernels use the existing CUDA/Vulkan backends but with
// uint32 accumulation and modular reduction instead of fp16/fp32 MAC.

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define SP_CRT_M1          2147483647ULL   // 2^31 - 1   (Mersenne prime)
#define SP_CRT_M2          2147483629ULL   // 2^31 - 19  (coprime to M1)
#define SP_CRT_GARNER_C    2028178983ULL   // M1^{-1} mod M2  (verified: M1*C mod M2 == 1)
#define SP_CRT_RANGE       (SP_CRT_M1 * SP_CRT_M2)  // ~4.6e18

// ============================================================================
// Modular arithmetic primitives
// ============================================================================

// Mersenne reduction: x mod (2^31 - 1) via shift-and-add.
// Input: x < 2^63 (safe for accumulator values in modular matmul).
// Output: result in [0, M1).
//
// With zero-centered quantization, negative values wrap to near M1,
// so a single product of two ring elements can reach M1² ≈ 2^62.
// An accumulator value after adding such a product can reach ~2^62.5.
// We handle this by using uint64_t for the intermediate sum and doing
// two rounds of fold-and-subtract.
static inline uint32_t sp_crt_mersenne_reduce(uint64_t x) {
    // Round 1: fold high bits down
    uint64_t lo = x & 0x7FFFFFFFULL;
    uint64_t hi = x >> 31;
    uint64_t r = lo + hi;
    // r < 2^31 + 2^32 = 3*2^31 after round 1 (if x < 2^63)
    // Round 2: fold again if still >= 2^31
    if (r >= 0x80000000ULL) {
        lo = r & 0x7FFFFFFFULL;
        hi = r >> 31;
        r = lo + hi;
    }
    // Now r < 2*M1, at most one final subtraction
    return (uint32_t)(r >= SP_CRT_M1 ? r - SP_CRT_M1 : r);
}

// Generic modular reduction for M2 (not Mersenne — uses standard remainder).
// Input: x < 2^62.
// Output: result in [0, M2).
static inline uint32_t sp_crt_m2_reduce(uint64_t x) {
    return (uint32_t)(x % SP_CRT_M2);
}

// Modular addition: (a + b) mod m, avoiding overflow.
static inline uint32_t sp_crt_add_mod(uint32_t a, uint32_t b, uint64_t m) {
    uint64_t s = (uint64_t)a + b;
    return (uint32_t)(s >= m ? s - m : s);
}

// Modular subtraction: (a - b) mod m, handling underflow.
static inline uint32_t sp_crt_sub_mod(uint32_t a, uint32_t b, uint64_t m) {
    return (uint32_t)(a >= b ? a - b : (uint32_t)(m - b + a));
}

// ============================================================================
// Garner recombination
// ============================================================================
//
// Given residues a1 = X mod M1 and a2 = X mod M2, recover X in [0, M1*M2).
//
//   X = a1 + M1 * h   where  h = (a2 - a1) * C  mod M2
//
// The result is a uint64_t. The caller rescales to float.

static inline uint64_t sp_crt_garner_reconstruct(uint32_t a1, uint32_t a2) {
    // Compute h = ((a2 - a1 mod M2) * GARNER_C) mod M2
    uint32_t a1_mod_m2 = sp_crt_m2_reduce((uint64_t)a1);
    uint32_t diff = sp_crt_sub_mod(a2, a1_mod_m2, SP_CRT_M2);
    uint64_t h = ((uint64_t)diff * SP_CRT_GARNER_C) % SP_CRT_M2;
    return (uint64_t)a1 + h * SP_CRT_M1;
}

// ============================================================================
// Quantization: fp16/fp32 ↔ residue ring
// ============================================================================
//
// To use CRT matmul, we need to convert floating-point weights and activations
// to integers in [0, M_i). The scale factor is chosen so the full fp16 range
// maps to [0, M1) with minimal quantization error.
//
// For fp16 weights in [-1, 1] (typical after LayerNorm):
//   scale = M1 / 2 = 1,073,741,823
//   zero_point = M1 / 2
//   q = round(x * scale) + zero_point
//
// For wider ranges, the scale adapts to the observed min/max.

typedef struct {
    double scale;       // float → int scaling factor
    double inv_scale;   // int → float (1.0 / scale)
    int64_t zero_point; // offset to make all values non-negative
} sp_crt_quant_t;

// Calibrate quantization parameters from observed value range.
// After calling, use sp_crt_quantize_f32 to convert.
// sp_crt_quant_calibrate_k is K-aware: limits the quantized range so that
// K-dimensional accumulation stays within the M1×M2 CRT range.
void sp_crt_quant_calibrate(sp_crt_quant_t *q, float min_val, float max_val);
void sp_crt_quant_calibrate_k(sp_crt_quant_t *q, float min_val, float max_val, int K);

// Quantize a float to a residue-ring integer for the given modulus.
static inline uint32_t sp_crt_quantize_f32(const sp_crt_quant_t *q,
                                            float val, uint64_t modulus) {
    int64_t ival = (int64_t)(val * q->scale + 0.5) + q->zero_point;
    if (ival < 0) ival = 0;
    return (uint32_t)((uint64_t)ival % modulus);
}

// Dequantize a reconstructed uint64 back to float.
static inline float sp_crt_dequantize_f32(const sp_crt_quant_t *q,
                                           uint64_t reconstructed) {
    return (float)((double)((int64_t)reconstructed - q->zero_point) * q->inv_scale);
}

// ============================================================================
// Batch operations — the main API
// ============================================================================

// CRT tensor split context. Owns the per-GPU residue buffers and
// the host-side reconstruction scratch.
typedef struct {
    // Per-GPU residue output buffers (host-mapped for async D2H)
    uint32_t *h_residue_0;   // pinned host memory for GPU 0 output
    uint32_t *h_residue_1;   // pinned host memory for GPU 1 output
    float    *h_output;      // final reconstructed float output

    // Quantization state (calibrated per-layer or global)
    sp_crt_quant_t weight_quant;
    sp_crt_quant_t act_quant;

    // Dimensions
    int M;    // output rows
    int N;    // output cols
    int K;    // inner dimension

    // GPU streams (opaque — CUDA stream or Vulkan queue)
    void *stream_0;   // GPU 0 (primary, e.g. CUDA)
    void *stream_1;   // GPU 1 (secondary, e.g. Vulkan or second CUDA)

    // Device-side buffers for GPU dispatch (allocated by sp_crt_init_gpu).
    // NULL when running CPU-only reference path.
    uint32_t *d_a_m1;       // [M × K] residues on GPU 0
    uint32_t *d_b_m1;       // [K × N] residues on GPU 0
    uint32_t *d_c_m1;       // [M × N] output residues on GPU 0
    uint32_t *d_a_m2;       // [M × K] residues on GPU 1 (or same GPU, stream_1)
    uint32_t *d_b_m2;       // [K × N] residues on GPU 1
    uint32_t *d_c_m2;       // [M × N] output residues on GPU 1
    int       gpu_ready;    // 1 if device buffers are allocated

    // Heterogeneous dispatch: Ring M2 on Vulkan (Intel UHD) instead of CUDA
    int       vk_ring2;     // 1 if Ring M2 uses Vulkan compute
    void     *vk_ctx;       // sp_crt_vulkan_ctx_t* (opaque, for Ring M2 Vulkan dispatch)

    int initialized;
} sp_crt_context_t;

// Initialize CRT context. Allocates pinned host buffers for the given
// maximum matmul dimensions (M × K × N). Both GPU streams must already
// be created by the caller.
int  sp_crt_init(sp_crt_context_t *ctx, int max_M, int max_N, int max_K,
                 void *stream_0, void *stream_1);
void sp_crt_free(sp_crt_context_t *ctx);

// ── The main dispatch ──────────────────────────────────────────────
//
// Performs C = A × B across two GPUs using CRT parallelism.
//
// A: [M × K] float, device pointer on GPU 0 (will be quantized)
// B: [K × N] float, device pointer on GPU 0 (will be broadcast)
// C: [M × N] float, device pointer on GPU 0 (output, reconstructed)
//
// The function:
//   1. Quantizes A and B to residue ring integers
//   2. Broadcasts quantized B to GPU 1
//   3. Launches modular matmul on both GPUs concurrently
//   4. Copies residue results to host (async D2H)
//   5. Garner-reconstructs on host (AVX2/NEON vectorized)
//   6. Copies result back to GPU 0
//
// This is a drop-in replacement for a single-GPU matmul when two GPUs
// are available. The caller doesn't need to know about the CRT internals.

int sp_crt_matmul(sp_crt_context_t *ctx,
                  const float *d_A, const float *d_B, float *d_C,
                  int M, int N, int K);

// ── GPU-accelerated CRT matmul ───────────────────────────────────
//
// Allocates device buffers and enables GPU dispatch. After this call,
// sp_crt_matmul_gpu uses CUDA kernels on stream_0 (ring M1) and
// stream_1 (ring M2) concurrently. Host-side Garner reconstruction
// runs on the CPU after D2H copies complete.
//
// Both streams can be on the same GPU (concurrent kernel execution)
// or on different GPUs (Beast Canyon: RTX 2060 + Intel UHD).
// Call once after sp_crt_init. Free'd by sp_crt_free.

int sp_crt_init_gpu(sp_crt_context_t *ctx);

// Initialize GPU dispatch with heterogeneous split:
//   Ring M1 (Mersenne) → CUDA on GPU 0 (e.g. RTX 2060)
//   Ring M2 (generic)  → Vulkan on GPU 1 (e.g. Intel UHD)
// barrier: initialized sp_hetero_barrier_t* (cast to void* to avoid
// circular header dependency — sp_crt.h must not include sp_hetero_sync.h).
// Falls back to CUDA-only if Vulkan init fails.
int sp_crt_init_gpu_hetero(sp_crt_context_t *ctx, void *barrier);

// GPU-dispatch matmul: quantize on GPU, dual-ring matmul on two
// streams, Garner on host. Falls back to CPU if gpu_ready == 0.
// Inputs A/B are host floats; C is host float output.
int sp_crt_matmul_gpu(sp_crt_context_t *ctx,
                      const float *A, const float *B, float *C,
                      int M, int N, int K);

// ── Host-only Garner reconstruction (for testing / CPU-only path) ──
//
// Reconstructs n elements from two residue arrays into float output.
// Uses the act_quant scaling from the context for dequantization.
// This is the function the main dispatch calls after D2H copies complete.
void sp_crt_garner_batch(const uint32_t *residue_0,
                         const uint32_t *residue_1,
                         float *output,
                         size_t n,
                         const sp_crt_quant_t *quant);

// ============================================================================
// CUDA-specific modular matmul kernel launcher
// ============================================================================
//
// Performs C_mod = (A_int × B_int) mod M using uint32 accumulation.
// A_int: [M × K] uint32 in residue ring
// B_int: [K × N] uint32 in residue ring
// C_mod: [M × N] uint32 output residues
//
// For M1 (Mersenne): uses bit-shift reduction in the inner loop.
// For M2: uses standard uint64 accumulation with final % M2.

void sp_crt_cuda_matmul_mersenne(const uint32_t *d_A, const uint32_t *d_B,
                                  uint32_t *d_C,
                                  int M, int N, int K,
                                  void *stream);

void sp_crt_cuda_matmul_mod(const uint32_t *d_A, const uint32_t *d_B,
                             uint32_t *d_C,
                             int M, int N, int K,
                             uint64_t modulus,
                             void *stream);

// Quantize a float tensor to uint32 residues on GPU.
void sp_crt_cuda_quantize(const float *d_input, uint32_t *d_output,
                           int n, double scale, int64_t zero_point,
                           uint64_t modulus, void *stream);

// ============================================================================
// Vulkan-specific modular matmul (optional — for Intel UHD / Adreno)
// ============================================================================
//
// Same interface as CUDA but dispatches via Vulkan compute shaders.
// Used when the secondary GPU is Intel UHD or another Vulkan-only device.

void sp_crt_vulkan_matmul_mod(const uint32_t *d_A, const uint32_t *d_B,
                               uint32_t *d_C,
                               int M, int N, int K,
                               uint64_t modulus,
                               void *vk_queue);

// ── CUDA memory/stream helpers (for GPU dispatch) ───────────────
//
// Thin wrappers so the pure-C host code can manage CUDA memory
// without including cuda_runtime.h.

void* sp_crt_cuda_alloc(size_t bytes);
void  sp_crt_cuda_free(void *ptr);
int   sp_crt_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes, void *stream);
int   sp_crt_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes, void *stream);
int   sp_crt_cuda_stream_sync(void *stream);
void* sp_crt_cuda_stream_create(void);
void  sp_crt_cuda_stream_destroy(void *stream);

// GPU-side zero-centered symmetric quantization (same math as CPU version).
void sp_crt_cuda_quantize_symmetric(const float *d_input, uint32_t *d_output,
                                     int n, double scale, uint64_t modulus,
                                     void *stream);

// ── Verification / test utilities ────────────────────────────────
//
// Round-trip test: quantize → split → matmul_mod → Garner → dequantize
// for a single a × b multiplication. Returns 0 on success (error < 1%).
int sp_crt_verify_roundtrip(float a, float b, float *error_out);

#ifdef __cplusplus
}
#endif

#endif // SP_CRT_H
