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

// ── Persistent device-side scratch (for full-block GPU execution) ──
//
// Allocates / returns a named device float buffer that survives across
// matvec calls. Used to keep layer-block intermediates (residual stream,
// normalised activations, QKV buffer, etc.) resident in VRAM so the
// per-call sync cost is paid once per BLOCK (~36 syncs/token) instead
// of once per MATVEC (~120 syncs/token).
//
// Names are caller-defined string keys; up to 32 buffers tracked.
// First call with a given (name, n_floats) allocates; subsequent calls
// return the existing pointer (re-allocating if n_floats grew).
float *sp_beast_gpu_buf(sp_beast_gpu_ctx_t *ctx, const char *name, int n_floats);

// H2D / D2H against a persistent buffer. Async on the ctx's stream.
int sp_beast_gpu_upload(sp_beast_gpu_ctx_t *ctx,
                         float *d_dst, const float *h_src, int n);
int sp_beast_gpu_download(sp_beast_gpu_ctx_t *ctx,
                           float *h_dst, const float *d_src, int n);

// Sync the ctx's stream (block until all queued ops complete).
int sp_beast_gpu_sync(sp_beast_gpu_ctx_t *ctx);

// Run a Q4_K matvec where input AND output stay device-resident.
// No H2D / D2H — caller already placed d_x and will read d_y later.
int sp_beast_gpu_q4k_matvec_dd(sp_beast_gpu_ctx_t *ctx,
                                const void *dW_q4k,
                                const float *d_x, float *d_y,
                                int n_out, int n_in);

// Device-side RMS norm: d_y[i] = d_x[i] * weight[i] / sqrt(mean(x^2) + eps).
int sp_beast_gpu_rms_norm(sp_beast_gpu_ctx_t *ctx,
                           const float *d_x, const float *d_w,
                           float *d_y, int n, float eps);

// Device-side residual add: d_y[i] += d_a[i].
int sp_beast_gpu_add(sp_beast_gpu_ctx_t *ctx,
                      float *d_y, const float *d_a, int n);

// Stage a raw fp32 buffer to VRAM (for 1D weights — norms, biases,
// ssm_a, dt_bias). Returns device pointer or NULL.
float *sp_beast_gpu_stage_fp32_raw(sp_beast_gpu_ctx_t *ctx,
                                     const float *host, int n);

// SSM-block GPU kernel launchers (one per kernel in sp_beast_gpu_q4k.cu).
// All take the ctx's stream implicitly via the wrappers below.
int sp_beast_gpu_silu(sp_beast_gpu_ctx_t *ctx, float *d_x, int n);
int sp_beast_gpu_silu_gate(sp_beast_gpu_ctx_t *ctx,
                            float *d_y, const float *d_g, int n);
int sp_beast_gpu_conv1d(sp_beast_gpu_ctx_t *ctx,
                         float *d_qkv, float *d_conv_state,
                         const float *d_conv_w, const float *d_conv_b,
                         int conv_dim, int conv_k);
int sp_beast_gpu_l2_qk(sp_beast_gpu_ctx_t *ctx,
                        float *d_q, float *d_k,
                        int n_k_heads, int head_k_dim);
int sp_beast_gpu_ssm_norm(sp_beast_gpu_ctx_t *ctx,
                           float *d_output, const float *d_weight,
                           int n_v_heads, int head_v_dim, int norm_dim,
                           float eps);
int sp_beast_gpu_gate_beta(sp_beast_gpu_ctx_t *ctx,
                            const float *alpha_raw, const float *beta_raw,
                            const float *ssm_a, const float *dt_bias,
                            float *gate_vals, float *beta_vals,
                            int n_v_heads);
int sp_beast_gpu_delta_net(sp_beast_gpu_ctx_t *ctx,
                            const float *q_all, const float *k_all,
                            const float *v_all,
                            float *S_all, float *output,
                            const float *gate_vals, const float *beta_vals,
                            int n_v_heads, int n_k_heads,
                            int head_k_dim, int head_v_dim);

// ── CUDA Graphs: record once, replay many ──
//
// The per-layer SSM kernel sequence is fixed once weights are staged;
// capturing it as a graph and launching the captured graph each token
// costs ~5-10 us total instead of paying per-kernel WDDM launch overhead
// (~100-500 us per kernel observed on RTX 2060 + Windows). Buffer
// addresses inside the captured graph are persistent device pointers,
// so updated contents flow through naturally on every replay.
int sp_beast_gpu_capture_begin(sp_beast_gpu_ctx_t *ctx);
int sp_beast_gpu_capture_end(sp_beast_gpu_ctx_t *ctx, void **graph_out);
int sp_beast_gpu_graph_instantiate(void *graph, void **exec_out);
int sp_beast_gpu_graph_launch_and_sync(sp_beast_gpu_ctx_t *ctx, void *exec);

#endif // SP_WITH_CUDA

#ifdef __cplusplus
}
#endif

#endif // SP_BEAST_GPU_H
