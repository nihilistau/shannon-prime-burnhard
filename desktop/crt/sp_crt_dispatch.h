// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#ifndef SP_CRT_DISPATCH_H
#define SP_CRT_DISPATCH_H

// ============================================================================
// CRT Dispatch — integration layer between forward paths and CRT backend
// ============================================================================
//
// Provides two integration hooks:
//
//   1. ggml_map_custom2 callback — for the ggml graph-based forward path.
//      Replaces ggml_mul_mat(W, x) with a CRT dual-ring matmul when
//      --crt-split is enabled. The callback handles:
//        * Weight dequantization (if quantized) or direct float access
//        * Column-major → row-major transpose for ggml's [K, M] weight layout
//        * Calibration of quantization parameters
//        * Dispatch to sp_crt_matmul_gpu (or CPU fallback)
//        * Writing the result into the ggml destination tensor
//
//   2. mm_dispatch callback — for the native forward path (forward_native.h).
//      Matches the mm_dispatch_fn_t signature exactly. The native path already
//      provides weights as fp16 [K, N] and input as fp32 [M, K], which maps
//      directly to CRT matmul with no transpose needed.
//
// Both hooks share a single sp_crt_context_t that owns the GPU streams,
// device buffers, and quantization state. Create it once at engine init;
// destroy at shutdown. The context is thread-safe for single-writer use
// (one matmul at a time per context, which is the normal serial graph flow).

#include "sp_crt.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Dispatch context — wraps sp_crt_context_t with forward-path metadata
// ============================================================================

typedef struct {
    sp_crt_context_t crt;          // The underlying CRT engine

    // Maximum dimensions this context was initialised for. Any matmul
    // where M*N*K exceeds the init-time maxima will fall through to
    // the default (non-CRT) compute path.
    int max_M;
    int max_N;
    int max_K;

    // Scratch buffer for transposing ggml's column-major weights to
    // row-major for the CRT quantization path. Allocated once at init,
    // sized for max_M * max_K floats.
    float* transpose_scratch;

    // Stats (for diagnostics / benchmarking)
    int    n_dispatched;           // total matmuls routed through CRT
    int    n_fallthrough;          // matmuls that fell through to default
    double total_us;               // cumulative microseconds in CRT dispatch
} sp_crt_dispatch_t;

// Initialise the dispatch context. max_dim is the largest expected
// matrix dimension (used for all three of M, N, K). Allocates GPU
// buffers if CUDA is available.
//
// Returns 0 on success, non-zero on allocation failure.
int sp_crt_dispatch_init(sp_crt_dispatch_t* d, int max_dim);

// Free all resources.
void sp_crt_dispatch_free(sp_crt_dispatch_t* d);

// Print dispatch statistics.
void sp_crt_dispatch_stats(const sp_crt_dispatch_t* d);

// ============================================================================
// Hook 1: ggml_map_custom2 callback
// ============================================================================
//
// Usage in forward.cpp graph builder:
//
//   // Instead of:
//   //   Q = ggml_mul_mat(gctx, L.wq, xa);
//   // Use:
//   Q = ggml_map_custom2(gctx, L.wq, xa,
//                        sp_crt_ggml_matmul, 1, &dispatch_ctx);
//
// The callback signature matches ggml_custom2_op_t:
//   void (ggml_tensor* dst, const ggml_tensor* a, const ggml_tensor* b,
//         int ith, int nth, void* userdata)
//
// Where:
//   a = weight tensor, shape [K, M] in ggml convention (column-major)
//   b = input tensor,  shape [K, N] (activations)
//   dst = output,      shape [M, N]
//   userdata = sp_crt_dispatch_t*
//
// The callback transposes 'a' from [K, M] to [M, K], calibrates
// quantization, and dispatches through CRT. Only thread 0 (ith == 0)
// does work; other threads return immediately.

struct ggml_tensor;

void sp_crt_ggml_matmul(struct ggml_tensor* dst,
                         const struct ggml_tensor* a,
                         const struct ggml_tensor* b,
                         int ith, int nth, void* userdata);

// ============================================================================
// Hook 2: native forward mm_dispatch callback
// ============================================================================
//
// Matches forward_native.h mm_dispatch_fn_t exactly.
//
// Usage when binding layers in ForwardNativeContext:
//   layer.mm_dispatch = sp_crt_native_mm_dispatch;
//   layer.mm_dispatch_userdata = &dispatch_ctx;
//
// The native path provides weights in [K, N] fp16 layout and input in
// [M, K] fp32. This is row-major for both — no transpose needed.
// The function converts fp16 weights to fp32, calibrates, and dispatches.
//
// Returns 0 on success (CRT result written), non-zero on fallthrough
// (caller should use the default CPU matmul).

int sp_crt_native_mm_dispatch(void* userdata,
                               const float*    lhs,       // [M, K] fp32
                               const uint16_t* W_fp16,    // [K, N] fp16
                               int M, int K, int N,
                               float*          out);      // [M, N] fp32

#ifdef __cplusplus
}
#endif

#endif // SP_CRT_DISPATCH_H
