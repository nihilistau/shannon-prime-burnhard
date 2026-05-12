// sp_beast_gpu.c — cuBLAS gemv backend for hot-staged weights.
//
// All entry points are no-ops unless SP_WITH_CUDA is defined; the rest of
// Beast Canyon then compiles and links unchanged on systems without CUDA.

#include "sp_beast_gpu.h"

#ifdef SP_WITH_CUDA

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// ============================================================================
// Internal state
// ============================================================================

struct sp_beast_gpu_ctx {
    cublasHandle_t  cublas;
    cudaStream_t    stream;

    // Per-call scratch (sized for the largest matvec we expect).
    // n_in_max=8192, n_out_max=8192 covers Qwen3.6 SSM/attn shapes
    // (wqkv 8192x2048, wz 4096x2048, ssm_out 2048x4096, shared_expert 30k...).
    // We grow lazily on demand.
    void   *d_x_fp16;          // device fp16 input vector
    int     d_x_cap;
    void   *d_y_fp32;          // device fp32 output vector
    int     d_y_cap;
    __half *h_x_fp16_pinned;   // pinned host fp16 staging
    int     h_x_cap;
    float  *h_y_fp32_pinned;   // pinned host fp32 result staging
    int     h_y_cap;

    size_t  vram_bytes;        // total fp16 weights staged
};

// ============================================================================
// Helpers
// ============================================================================

#define SP_GPU_LOG(fmt, ...) \
    fprintf(stderr, "[sp-beast-gpu] " fmt "\n", ##__VA_ARGS__)

static int sp_gpu_check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        SP_GPU_LOG("%s failed: %s", what, cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

static int sp_gpu_grow_scratch(struct sp_beast_gpu_ctx *ctx,
                               int n_in, int n_out) {
    if (n_in > ctx->d_x_cap) {
        if (ctx->d_x_fp16) cudaFree(ctx->d_x_fp16);
        if (ctx->h_x_fp16_pinned) cudaFreeHost(ctx->h_x_fp16_pinned);
        if (sp_gpu_check(cudaMalloc(&ctx->d_x_fp16,
                                     (size_t)n_in * sizeof(__half)),
                          "cudaMalloc d_x_fp16")) return -1;
        if (sp_gpu_check(cudaMallocHost((void **)&ctx->h_x_fp16_pinned,
                                         (size_t)n_in * sizeof(__half)),
                          "cudaMallocHost h_x_fp16")) return -1;
        ctx->d_x_cap = n_in;
        ctx->h_x_cap = n_in;
    }
    if (n_out > ctx->d_y_cap) {
        if (ctx->d_y_fp32) cudaFree(ctx->d_y_fp32);
        if (ctx->h_y_fp32_pinned) cudaFreeHost(ctx->h_y_fp32_pinned);
        if (sp_gpu_check(cudaMalloc(&ctx->d_y_fp32,
                                     (size_t)n_out * sizeof(float)),
                          "cudaMalloc d_y_fp32")) return -1;
        if (sp_gpu_check(cudaMallocHost((void **)&ctx->h_y_fp32_pinned,
                                         (size_t)n_out * sizeof(float)),
                          "cudaMallocHost h_y_fp32")) return -1;
        ctx->d_y_cap = n_out;
        ctx->h_y_cap = n_out;
    }
    return 0;
}

// ============================================================================
// Public API
// ============================================================================

sp_beast_gpu_ctx_t *sp_beast_gpu_init(void) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        SP_GPU_LOG("no CUDA device available");
        return NULL;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        SP_GPU_LOG("cudaSetDevice(0) failed");
        return NULL;
    }

    struct sp_beast_gpu_ctx *ctx = (struct sp_beast_gpu_ctx *)
        calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    if (cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS) {
        SP_GPU_LOG("cublasCreate failed");
        free(ctx);
        return NULL;
    }
    if (cudaStreamCreate(&ctx->stream) != cudaSuccess) {
        SP_GPU_LOG("cudaStreamCreate failed");
        cublasDestroy(ctx->cublas);
        free(ctx);
        return NULL;
    }
    cublasSetStream(ctx->cublas, ctx->stream);
    cublasSetMathMode(ctx->cublas, CUBLAS_TF32_TENSOR_OP_MATH);

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        SP_GPU_LOG("init OK: device=%s sm=%d.%d total_vram=%.2f GB",
                   prop.name, prop.major, prop.minor,
                   (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    } else {
        SP_GPU_LOG("init OK (device props unavailable)");
    }
    return ctx;
}

void *sp_beast_gpu_stage_fp32(sp_beast_gpu_ctx_t *ctx,
                              const float *host_fp32,
                              int n_out, int n_in) {
    if (!ctx || !host_fp32) return NULL;
    size_t n = (size_t)n_out * (size_t)n_in;
    if (n == 0) return NULL;

    // Convert fp32 → fp16 on host (one-shot at boot, not in hot path).
    __half *host_fp16 = (__half *)malloc(n * sizeof(__half));
    if (!host_fp16) {
        SP_GPU_LOG("stage: malloc %zu bytes failed",
                   n * sizeof(__half));
        return NULL;
    }
    for (size_t i = 0; i < n; ++i) {
        host_fp16[i] = __float2half(host_fp32[i]);
    }

    void *d_w = NULL;
    if (cudaMalloc(&d_w, n * sizeof(__half)) != cudaSuccess) {
        SP_GPU_LOG("stage: cudaMalloc %zu fp16 failed (used=%.2f GB)",
                   n,
                   (double)ctx->vram_bytes / (1024.0 * 1024.0 * 1024.0));
        free(host_fp16);
        return NULL;
    }
    if (cudaMemcpy(d_w, host_fp16, n * sizeof(__half),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        SP_GPU_LOG("stage: cudaMemcpy H2D failed");
        cudaFree(d_w);
        free(host_fp16);
        return NULL;
    }
    free(host_fp16);
    ctx->vram_bytes += n * sizeof(__half);
    return d_w;
}

int sp_beast_gpu_matvec(sp_beast_gpu_ctx_t *ctx,
                        const void *dW_fp16,
                        const float *x, float *out,
                        int n_out, int n_in) {
    if (!ctx || !dW_fp16 || !x || !out) return -1;
    if (sp_gpu_grow_scratch(ctx, n_in, n_out) != 0) return -1;

    // Convert input vector fp32 → fp16 on host into pinned buffer.
    for (int i = 0; i < n_in; ++i) {
        ctx->h_x_fp16_pinned[i] = __float2half(x[i]);
    }
    if (cudaMemcpyAsync(ctx->d_x_fp16, ctx->h_x_fp16_pinned,
                         (size_t)n_in * sizeof(__half),
                         cudaMemcpyHostToDevice, ctx->stream)
        != cudaSuccess) return -1;

    // y = W @ x  in row-major: W is [n_out, n_in], x is [n_in].
    // cuBLAS sees that linear buffer as a column-major matrix of shape
    // (n_in, n_out) — i.e. W^T. So with op=T it computes (W^T)^T @ x = W @ x.
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        n_out, 1, n_in,
        &alpha,
        dW_fp16,         CUDA_R_16F, n_in,   // A: (n_in, n_out) col-major
        ctx->d_x_fp16,   CUDA_R_16F, n_in,   // B: (n_in, 1)
        &beta,
        ctx->d_y_fp32,   CUDA_R_32F, n_out,  // C: (n_out, 1)
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        SP_GPU_LOG("cublasGemmEx failed: %d (n_out=%d n_in=%d)",
                   (int)st, n_out, n_in);
        return -1;
    }

    if (cudaMemcpyAsync(ctx->h_y_fp32_pinned, ctx->d_y_fp32,
                         (size_t)n_out * sizeof(float),
                         cudaMemcpyDeviceToHost, ctx->stream)
        != cudaSuccess) return -1;
    if (cudaStreamSynchronize(ctx->stream) != cudaSuccess) return -1;

    memcpy(out, ctx->h_y_fp32_pinned, (size_t)n_out * sizeof(float));
    return 0;
}

void sp_beast_gpu_free_weight(sp_beast_gpu_ctx_t *ctx, void *dW_fp16) {
    if (!ctx || !dW_fp16) return;
    cudaFree(dW_fp16);
}

// ────────────────────────────────────────────────────────────────────────
// Q4_K-resident path (custom CUDA kernel; halves VRAM read traffic vs fp16)
// ────────────────────────────────────────────────────────────────────────

extern "C" cudaError_t sp_beast_gpu_q4k_matvec_launch(
    const void *dW_raw,
    const float *d_x, float *d_y,
    int n_out, int n_in,
    cudaStream_t stream);

extern "C" cudaError_t sp_beast_gpu_rms_norm_launch(
    const float *d_x, const float *d_w, float *d_y,
    int n, float eps, cudaStream_t stream);

extern "C" cudaError_t sp_beast_gpu_add_launch(
    float *d_y, const float *d_a, int n, cudaStream_t stream);

// ── Persistent named device buffers ──
//
// Small fixed-size registry. Used to keep layer-block intermediates
// (residual stream, normalised activations, qkv buffer, etc.) device-
// resident so per-call sync is paid once per BLOCK instead of per matvec.
struct sp_gpu_buf_entry {
    char     name[64];
    float   *ptr;
    int      n_floats;
};
#define SP_GPU_BUF_MAX 32
static sp_gpu_buf_entry g_gpu_bufs[SP_GPU_BUF_MAX] = {};
static int              g_gpu_buf_count = 0;

float *sp_beast_gpu_buf(sp_beast_gpu_ctx_t *ctx, const char *name,
                         int n_floats) {
    if (!ctx || !name || n_floats <= 0) return NULL;
    for (int i = 0; i < g_gpu_buf_count; ++i) {
        if (strcmp(g_gpu_bufs[i].name, name) == 0) {
            if (g_gpu_bufs[i].n_floats >= n_floats) return g_gpu_bufs[i].ptr;
            cudaFree(g_gpu_bufs[i].ptr);
            if (cudaMalloc(&g_gpu_bufs[i].ptr,
                            (size_t)n_floats * sizeof(float)) != cudaSuccess) {
                g_gpu_bufs[i].ptr = NULL;
                return NULL;
            }
            g_gpu_bufs[i].n_floats = n_floats;
            return g_gpu_bufs[i].ptr;
        }
    }
    if (g_gpu_buf_count >= SP_GPU_BUF_MAX) {
        SP_GPU_LOG("buf registry full (%d entries)", g_gpu_buf_count);
        return NULL;
    }
    sp_gpu_buf_entry *e = &g_gpu_bufs[g_gpu_buf_count];
    snprintf(e->name, sizeof(e->name), "%s", name);
    if (cudaMalloc(&e->ptr, (size_t)n_floats * sizeof(float))
        != cudaSuccess) {
        SP_GPU_LOG("buf '%s': cudaMalloc %d floats failed", name, n_floats);
        return NULL;
    }
    e->n_floats = n_floats;
    ++g_gpu_buf_count;
    return e->ptr;
}

int sp_beast_gpu_upload(sp_beast_gpu_ctx_t *ctx,
                         float *d_dst, const float *h_src, int n) {
    if (!ctx || !d_dst || !h_src || n <= 0) return -1;
    return (cudaMemcpyAsync(d_dst, h_src, (size_t)n * sizeof(float),
                              cudaMemcpyHostToDevice, ctx->stream)
            == cudaSuccess) ? 0 : -1;
}

int sp_beast_gpu_download(sp_beast_gpu_ctx_t *ctx,
                           float *h_dst, const float *d_src, int n) {
    if (!ctx || !h_dst || !d_src || n <= 0) return -1;
    return (cudaMemcpyAsync(h_dst, d_src, (size_t)n * sizeof(float),
                              cudaMemcpyDeviceToHost, ctx->stream)
            == cudaSuccess) ? 0 : -1;
}

int sp_beast_gpu_sync(sp_beast_gpu_ctx_t *ctx) {
    if (!ctx) return -1;
    return cudaStreamSynchronize(ctx->stream) == cudaSuccess ? 0 : -1;
}

int sp_beast_gpu_q4k_matvec_dd(sp_beast_gpu_ctx_t *ctx,
                                const void *dW_q4k,
                                const float *d_x, float *d_y,
                                int n_out, int n_in) {
    if (!ctx || !dW_q4k || !d_x || !d_y) return -1;
    cudaError_t rc = sp_beast_gpu_q4k_matvec_launch(
        dW_q4k, d_x, d_y, n_out, n_in, ctx->stream);
    return (rc == cudaSuccess) ? 0 : -1;
}

int sp_beast_gpu_rms_norm(sp_beast_gpu_ctx_t *ctx,
                           const float *d_x, const float *d_w,
                           float *d_y, int n, float eps) {
    if (!ctx || !d_x || !d_w || !d_y || n <= 0) return -1;
    return (sp_beast_gpu_rms_norm_launch(d_x, d_w, d_y, n, eps,
                                          ctx->stream)
            == cudaSuccess) ? 0 : -1;
}

int sp_beast_gpu_add(sp_beast_gpu_ctx_t *ctx,
                      float *d_y, const float *d_a, int n) {
    if (!ctx || !d_y || !d_a || n <= 0) return -1;
    return (sp_beast_gpu_add_launch(d_y, d_a, n, ctx->stream)
            == cudaSuccess) ? 0 : -1;
}

void *sp_beast_gpu_stage_q4k(sp_beast_gpu_ctx_t *ctx,
                              const void *host_q4k_bytes,
                              int n_out, int n_in) {
    if (!ctx || !host_q4k_bytes) return NULL;
    if (n_in % 256 != 0) {
        SP_GPU_LOG("stage_q4k: n_in=%d not divisible by 256", n_in);
        return NULL;
    }
    const size_t row_bytes = (size_t)(n_in / 256) * 144;
    const size_t total = row_bytes * (size_t)n_out;
    void *d_w = NULL;
    if (cudaMalloc(&d_w, total) != cudaSuccess) {
        SP_GPU_LOG("stage_q4k: cudaMalloc %zu bytes failed (used=%.2f GB)",
                   total,
                   (double)ctx->vram_bytes / (1024.0 * 1024.0 * 1024.0));
        return NULL;
    }
    if (cudaMemcpy(d_w, host_q4k_bytes, total,
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        SP_GPU_LOG("stage_q4k: H2D copy failed");
        cudaFree(d_w);
        return NULL;
    }
    ctx->vram_bytes += total;
    return d_w;
}

// Per-row matvec scratch for the Q4_K path. Separate from the fp16 path's
// d_x_fp16 because we need fp32 input here. Lazily grown.
static void   *g_q4k_d_x_fp32     = NULL;
static int     g_q4k_d_x_fp32_cap = 0;

int sp_beast_gpu_q4k_matvec(sp_beast_gpu_ctx_t *ctx,
                             const void *dW_q4k,
                             const float *x, float *out,
                             int n_out, int n_in) {
    if (!ctx || !dW_q4k || !x || !out) return -1;

    if (n_in > g_q4k_d_x_fp32_cap) {
        if (g_q4k_d_x_fp32) cudaFree(g_q4k_d_x_fp32);
        if (cudaMalloc(&g_q4k_d_x_fp32,
                        (size_t)n_in * sizeof(float)) != cudaSuccess)
            return -1;
        g_q4k_d_x_fp32_cap = n_in;
    }
    // Reuse the fp16 path's output scratch (it's just a byte buffer).
    // Pass a dummy n_in for the fp16 grow path so we only grow d_y/h_y.
    if (sp_gpu_grow_scratch(ctx, /*fp16 input dummy=*/16, n_out) != 0)
        return -1;

    if (cudaMemcpyAsync(g_q4k_d_x_fp32, x,
                         (size_t)n_in * sizeof(float),
                         cudaMemcpyHostToDevice, ctx->stream)
        != cudaSuccess) return -1;

    cudaError_t krc = sp_beast_gpu_q4k_matvec_launch(
        dW_q4k, (const float *)g_q4k_d_x_fp32,
        (float *)ctx->d_y_fp32, n_out, n_in, ctx->stream);
    if (krc != cudaSuccess) {
        SP_GPU_LOG("q4k_matvec_launch failed: %s",
                   cudaGetErrorString(krc));
        return -1;
    }

    if (cudaMemcpyAsync(ctx->h_y_fp32_pinned, ctx->d_y_fp32,
                         (size_t)n_out * sizeof(float),
                         cudaMemcpyDeviceToHost, ctx->stream)
        != cudaSuccess) return -1;
    if (cudaStreamSynchronize(ctx->stream) != cudaSuccess) return -1;

    memcpy(out, ctx->h_y_fp32_pinned, (size_t)n_out * sizeof(float));
    return 0;
}

void sp_beast_gpu_destroy(sp_beast_gpu_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->d_x_fp16) cudaFree(ctx->d_x_fp16);
    if (ctx->d_y_fp32) cudaFree(ctx->d_y_fp32);
    if (ctx->h_x_fp16_pinned) cudaFreeHost(ctx->h_x_fp16_pinned);
    if (ctx->h_y_fp32_pinned) cudaFreeHost(ctx->h_y_fp32_pinned);
    if (g_q4k_d_x_fp32) { cudaFree(g_q4k_d_x_fp32); g_q4k_d_x_fp32 = NULL; g_q4k_d_x_fp32_cap = 0; }
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    if (ctx->cublas) cublasDestroy(ctx->cublas);
    free(ctx);
}

size_t sp_beast_gpu_used_bytes(const sp_beast_gpu_ctx_t *ctx) {
    return ctx ? ctx->vram_bytes : 0;
}

#endif // SP_WITH_CUDA
