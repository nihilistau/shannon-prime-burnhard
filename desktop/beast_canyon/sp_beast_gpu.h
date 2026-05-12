// sp_beast_gpu.h — RTX/cuBLAS gemv fast path for Beast Canyon
//
// Stages "hot" weight tensors (attn Q/K/V/O, RMS norms, router, shared
// expert FFN) for the first N layers as fp16 in RTX VRAM at boot, then
// dispatches per-token matvecs to cublasGemvEx instead of CPU AVX-512.
//
// Why: i9-11900KB peaks at ~50 GB/s DRAM. RTX 2060 has 336 GB/s VRAM.
// For tensors that fit (12 GB budget; ~2.5 GB needed for 20 hot layers),
// the GPU path gives ~6x bandwidth + ~10 TFLOPs fp16, dwarfing the
// per-call PCIe round-trip (~30 us H2D 8 KB + D2H 16 KB at PCIe 4.0).
//
// Compiled in only when SP_WITH_CUDA is defined.

#ifndef SP_BEAST_GPU_H
#define SP_BEAST_GPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef SP_WITH_CUDA

typedef struct sp_beast_gpu_ctx sp_beast_gpu_ctx_t;

// Initialise a cuBLAS handle on the default CUDA device. Returns NULL on
// failure (no device, no cublas, OOM).
sp_beast_gpu_ctx_t *sp_beast_gpu_init(void);

// Allocate fp16 VRAM and copy a row-major fp32 weight matrix into it.
// host_fp32 must be n_out * n_in floats. Returns the device __half *
// pointer (opaque void * here to avoid pulling cuda_fp16.h into C TUs).
// NULL on failure.
void *sp_beast_gpu_stage_fp32(sp_beast_gpu_ctx_t *ctx,
                              const float *host_fp32,
                              int n_out, int n_in);

// y[n_out] = W[n_out, n_in] @ x[n_in], synchronous.
// dW_fp16 was previously returned by sp_beast_gpu_stage_fp32.
// Returns 0 on success, non-zero on cuBLAS / cudaMemcpy error.
int sp_beast_gpu_matvec(sp_beast_gpu_ctx_t *ctx,
                        const void *dW_fp16,
                        const float *x, float *out,
                        int n_out, int n_in);

// Free all VRAM tracked by ctx (does not free dW_fp16 pointers; caller
// must walk the staged tensor list separately if it tracked them).
void sp_beast_gpu_destroy(sp_beast_gpu_ctx_t *ctx);

// Bytes currently allocated in VRAM via this ctx (sum of stage_fp32
// allocations + scratch).
size_t sp_beast_gpu_used_bytes(const sp_beast_gpu_ctx_t *ctx);

// Free a single staged weight allocation.
void sp_beast_gpu_free_weight(sp_beast_gpu_ctx_t *ctx, void *dW_fp16);

// ── Q4_K-resident path (custom CUDA kernel; 2x less VRAM than fp16) ──
//
// Stages a row-major Q4_K weight tensor as raw bytes in VRAM.
// `host_q4k_bytes` is the linear Q4_K buffer (n_out rows, each row with
// n_in/256 super-blocks of 144 bytes). Returns device pointer or NULL.
void *sp_beast_gpu_stage_q4k(sp_beast_gpu_ctx_t *ctx,
                              const void *host_q4k_bytes,
                              int n_out, int n_in);

// y[n_out] = W_q4k @ x[n_in] using the custom kernel that dequants
// inline. Synchronous. Returns 0 on success.
int sp_beast_gpu_q4k_matvec(sp_beast_gpu_ctx_t *ctx,
                             const void *dW_q4k,
                             const float *x, float *out,
                             int n_out, int n_in);

#endif // SP_WITH_CUDA

#ifdef __cplusplus
}
#endif

#endif // SP_BEAST_GPU_H
