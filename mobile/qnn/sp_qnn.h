/*
 * Shannon-Prime / Phase 2.3 — minimal QNN HTP runner shim.
 *
 * Public C API for loading a QNN context binary (compiled via AI Hub or
 * Qualcomm tools) and executing it as a matmul kernel inside a host
 * application. Designed to be linked into shannon-prime-llama as the
 * backend for ggml_map_custom2-style attention-matmul ops, but
 * standalone-usable as a plain C library.
 *
 * Why a shim instead of pulling in QnnSampleApp.cpp wholesale: the
 * SampleApp pulls 80+ KB of C++ across IOTensor / QnnWrapperUtils /
 * QnnSampleAppUtils plus the entire PAL/Logger machinery — useful as
 * reference but heavy to integrate. We only need the load-execute
 * primitives, and we can supply our own input/output buffers (rpcmem-
 * backed, allocated by the caller).
 *
 * Threading model: the handle is NOT thread-safe. Wrap externally if
 * you need concurrency. shannon-prime-llama uses one handle per layer
 * sequentially, so single-threaded is fine for the typical use.
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3.
 * Commercial license: raydaniels@gmail.com.
 */
#ifndef SHANNON_PRIME_QNN_H
#define SHANNON_PRIME_QNN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sp_qnn_handle sp_qnn_handle;

/* Status codes returned by every API. 0 == success. */
typedef enum {
    SP_QNN_OK                       =  0,
    SP_QNN_ERR_DLOPEN               = -1,  /* libQnnHtp.so / libQnnSystem.so not loadable */
    SP_QNN_ERR_GET_INTERFACE        = -2,  /* QnnInterface_getProviders failed */
    SP_QNN_ERR_BACKEND_CREATE       = -3,  /* QnnBackend_create failed */
    SP_QNN_ERR_DEVICE_CREATE        = -4,  /* QnnDevice_create failed */
    SP_QNN_ERR_READ_FILE            = -5,  /* couldn't read context binary */
    SP_QNN_ERR_BINARY_INFO          = -6,  /* QnnSystemContext_getBinaryInfo failed */
    SP_QNN_ERR_CONTEXT_CREATE       = -7,  /* QnnContext_createFromBinary failed */
    SP_QNN_ERR_GRAPH_RETRIEVE       = -8,  /* QnnContext_retrieveGraph failed */
    SP_QNN_ERR_TENSOR_SHAPE         = -9,  /* caller-supplied tensor doesn't match graph */
    SP_QNN_ERR_EXECUTE              = -10, /* QnnGraph_execute failed */
    SP_QNN_ERR_INVALID              = -11, /* generic invalid argument */
} sp_qnn_status;

/* Library-level init: dlopen the two QNN libs once per process.
 * Subsequent calls are no-ops.
 *
 * On Android, pass the canonical filenames; LD_LIBRARY_PATH should be
 * set to the directory containing libQnnHtp.so + libQnnSystem.so.
 *
 * NULL paths fall back to "libQnnHtp.so" + "libQnnSystem.so".
 */
sp_qnn_status sp_qnn_init(const char *backend_lib_path,
                          const char *system_lib_path);

/* Optional: clean up the dlopen'd libs at process exit. */
void sp_qnn_shutdown(void);

/* Reset the QNN backend+device at the prefill→decode transition.
 * Call this after all prefill QnnContext_free calls and before the first
 * decode QnnContext_createFromBinary.
 *
 * Root cause: After graph-switching prefill (4 × load→execute→destroy),
 * QnnGraph_execute on the first decode context blocks indefinitely in
 * fastrpc_wait_for_completion. Confirmed NOT a timing issue: 30s sleep
 * before or after contextCreateFromBinary, contextCreate drain barriers,
 * and keepalive empty contexts all fail. The DSP remains responsive between
 * attempts (contextCreate barriers return normally), but the 5th execute
 * never gets a response.
 *
 * The underlying cause is FastRPC session state accumulation: after 4
 * context create+free cycles the FastRPC kernel driver or DSP session layer
 * retains stale state (leaked SMMU entries, a hung IPC slot, or a stale
 * power wakelock) that prevents the next graphExecute from being acknowledged.
 *
 * Fix: call QnnBackend_free + QnnDevice_free, which closes the FastRPC
 * /dev/cdsp session (flushing all pending contextFree IPCs), then
 * QnnBackend_create + QnnDevice_create to open a fresh session. Decode
 * contexts attach to the clean session and execute normally.
 *
 * One-time cost: ~200–500ms at the prefill→decode boundary.
 */
void sp_qnn_drain_htp(void);

/* Load a QNN context binary from disk and prepare it for execution.
 * The binary must have been produced by a compile flow that targets
 * the same backend (HTP V69 in our case). On success, *out_h is a
 * fresh handle that owns the loaded context + graph + device.
 *
 * `graph_name` may be NULL to auto-pick the first graph in the binary.
 * Most AI-Hub-produced binaries have a single graph; pass the name
 * explicitly only for multi-graph contexts.
 */
sp_qnn_status sp_qnn_load_binary(const char *context_bin_path,
                                 const char *graph_name,  /* may be NULL */
                                 sp_qnn_handle **out_h);

/* Parse tensor schema from a context binary WITHOUT creating an HTP context.
 * QnnSystemContext only — no DSP resource allocation. The handle has
 * n_inputs/n_outputs/inputs/outputs populated; context and graph are NULL.
 * Use sp_qnn_get_io_info() only; do NOT call sp_qnn_execute().
 * sp_qnn_destroy() is safe (contextFree skipped when context==NULL).
 *
 * Use this in place of sp_qnn_load_binary when you only need tensor metadata
 * (buffer sizes, tensor names) and do not need to execute the graph. This
 * avoids saturating the HTP's deferred-cleanup queue with short-lived contexts.
 */
sp_qnn_status sp_qnn_parse_schema(const char *context_bin_path,
                                   sp_qnn_handle **out_h);

/* Load a list of QNN context binaries into a shared HTP resource pool.
 * This is critical for V69 hardware where total context size exceeds 
 * the working memory budget; shareResources=true lets contexts overlap.
 */
sp_qnn_status sp_qnn_load_binary_list(const char *const *paths,
                                      const char *const *graph_names,
                                      size_t n,
                                      sp_qnn_handle **out_handles);

/* Free the handle and all resources it owns. */
void sp_qnn_destroy(sp_qnn_handle **h);

/* === Runtime graph build (NO .bin) ===========================================
 * Build a single-op MatMul graph at runtime with both inputs as APP_WRITE,
 * so weights are bound per execute() call from caller's buffer (not baked
 * into a context binary). This is the "our kernel, dispatched directly to
 * HTP" path — bypasses the AOT-compile + .bin loading flow entirely.
 *
 * dtype is QNN_DATATYPE_FLOAT_32 (default). For native HTP performance
 * use QNN_DATATYPE_FLOAT_16 or QNN_DATATYPE_UFIXED_POINT_8.
 */
sp_qnn_status sp_qnn_runtime_matmul_create(uint32_t M, uint32_t K, uint32_t N,
                                           uint32_t qnn_dtype,
                                           sp_qnn_handle **out_h);

/* Build a 2-op KQ attention prefix graph at runtime: KQ = Q @ K^T, then
 * Softmax(KQ). Inputs Q[M,K_dim] and K[N,K_dim] both APP_WRITE; output
 * is attention weights [M,N]. K can later be promoted to a persistent
 * memhandle via sp_qnn_register_persistent_input() so K bytes are bound
 * once instead of per-call.
 *
 * This is the primitive shannon-prime-llama's FUSED_KQ slot calls into:
 * Q is generated fresh per attention step, K is the cumulative KV cache
 * (changes between tokens but not within a graph execute), so registering
 * K once per token amortizes the bind cost across all heads.
 */
sp_qnn_status sp_qnn_runtime_kq_softmax_create(uint32_t M_q,
                                               uint32_t K_dim,
                                               uint32_t N_kv,
                                               uint32_t qnn_dtype,
                                               sp_qnn_handle **out_h);

/* Register a per-handle persistent tensor input via QnnMem_register.
 * After this call, executes don't need that tensor's clientBuf rebound;
 * the backend reads from the registered memory directly. Use for stable
 * weights or KV history that persists across multiple executes.
 *
 * tensor_idx is the input-tensor index (0-based, matching the order
 * created in sp_qnn_runtime_*_create — typically 0=Q, 1=K, etc.).
 *
 * data must remain valid until sp_qnn_destroy() — caller owns the buffer.
 *
 * Subsequent calls to sp_qnn_execute() must pass NULL for this input slot
 * (or the binding is a no-op since memhandle tensors ignore clientBuf).
 *
 * NOTE: this is the legacy "pseudo-persistent clientBuf" path, kept for
 * backwards compat. For real DMA-resident persistence on V69 HTP use
 * sp_qnn_alloc_persistent() below instead.
 */
sp_qnn_status sp_qnn_register_persistent_input(sp_qnn_handle *h,
                                               size_t tensor_idx,
                                               void *data,
                                               size_t bytes);

/* Allocate an rpcmem-ION-backed buffer and bind it as a persistent
 * QNN MemHandle to the given input tensor. Phase 2.6b primitive.
 *
 * Mechanism:
 *   1. dlopen libcdsprpc.so (once per process), resolve rpcmem_*.
 *   2. rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes).
 *   3. rpcmem_to_fd(ptr) — get the ION fd.
 *   4. QnnMem_register(context, {ion fd, shape, dtype}, 1, &handle).
 *   5. Tensor template: memType = MEMHANDLE; memHandle = handle.
 *
 * On every subsequent sp_qnn_execute(), the caller passes NULL for
 * this input slot (tensor_set_buf detects MEMHANDLE and skips rebind).
 * Caller writes new data into *user_ptr directly between dispatches —
 * the same physical pages are DMA-mapped into the HTP, so writes are
 * visible without an explicit copy.
 *
 * Caller does NOT free *user_ptr — sp_qnn_destroy() handles deregister
 * + rpcmem_free for all persistent tensors registered to a handle.
 *
 * Returns SP_QNN_ERR_DLOPEN if libcdsprpc.so isn't present (e.g. on
 * a non-FastRPC system); caller should fall back to the legacy
 * sp_qnn_register_persistent_input() or per-call clientBuf rebind.
 */
sp_qnn_status sp_qnn_alloc_persistent(sp_qnn_handle *h,
                                      size_t tensor_idx,
                                      size_t bytes,
                                      void **out_user_ptr);

/* Load N context binaries as a GROUP via QnnContext_createFromBinaryListAsync,
 * with HTP `shareResources=true` so they share underlying kernel/workspace
 * state. This is the canonical multi-context API and fixes the case where
 * sequential per-binary loads exceed HTP working memory (e.g., our 4 PP
 * splits where split 4 alone is 1 GB).
 *
 * Caller provides parallel arrays:
 *   paths:        N context binary file paths
 *   graph_names:  N graph names (NULL or NULL entry = pick first)
 *   out_handles:  N output slots filled on success
 *
 * On any failure mid-group, all already-created contexts are destroyed
 * and the function returns the failing status. Out_handles is unmodified
 * unless the function returns SP_QNN_OK.
 */
sp_qnn_status sp_qnn_load_binary_list(const char *const *paths,
                                      const char *const *graph_names,
                                      size_t n,
                                      sp_qnn_handle **out_handles);

/* Query graph metadata. Returned strings + dim arrays are owned by the
 * handle; do not free. NULL out-params are skipped.
 *
 * Typical use: after sp_qnn_load_binary, call this to learn the input
 * tensor shape so the caller knows how big to make the input buffer.
 */
typedef struct {
    const char  *name;
    uint32_t     rank;
    const uint32_t *dims;       /* rank entries */
    uint32_t     bytes_per_element;
    uint32_t     dtype;          /* QNN_DATATYPE_* */
    /* Quantization params for SCALE_OFFSET / BW_SCALE_OFFSET tensors.
     * Critical for UFIXED_POINT_16/8 tensors which AI Hub-exported
     * Qwen-style .bins use for activations: input mask, position_ids,
     * residual streams, and logits are all uint16-quantized. To send
     * a fp32 value `v` to such a tensor:
     *     uint16 q = round(v / quant_scale) + quant_offset
     * To recover fp32 from a tensor's output bytes:
     *     fp32 v = (q - quant_offset) * quant_scale
     * For non-quantized tensors quant_scale=1.0 and quant_offset=0
     * so identity formulas apply.
     * quant_encoding mirrors Qnn_QuantizationEncoding_t. */
    uint32_t     quant_encoding;
    float        quant_scale;
    int32_t      quant_offset;
} sp_qnn_tensor_info;

sp_qnn_status sp_qnn_get_io_info(sp_qnn_handle *h,
                                 size_t *out_n_inputs,
                                 const sp_qnn_tensor_info **out_inputs,
                                 size_t *out_n_outputs,
                                 const sp_qnn_tensor_info **out_outputs);

/* One-shot execute. Caller supplies input buffers (one per declared
 * input tensor, in order) and pre-allocated output buffers. On success,
 * outputs are filled and *out_exec_us holds the wall-clock execution
 * time in microseconds (excluding tensor binding).
 *
 * The buffers do NOT have to be rpcmem-backed for correctness, but
 * rpcmem-backed buffers avoid an internal copy on aarch64 Android.
 * Pass an rpcmem-backed buffer if you have one.
 */
sp_qnn_status sp_qnn_execute(sp_qnn_handle *h,
                             const void *const *inputs,    /* n_inputs pointers */
                             const size_t *input_sizes,    /* n_inputs sizes in bytes */
                             void *const *outputs,         /* n_outputs pointers */
                             const size_t *output_sizes,   /* n_outputs sizes in bytes */
                             uint64_t *out_exec_us);       /* optional, may be NULL */

/* Convenience: bench N executions of the same input, returning min/
 * avg/max in microseconds. Useful for the per-layer matmul profiling
 * we'll do during shannon-prime-llama integration.
 */
typedef struct {
    uint32_t n_iterations;
    uint64_t min_us;
    uint64_t avg_us;
    uint64_t max_us;
} sp_qnn_bench_result;

sp_qnn_status sp_qnn_bench(sp_qnn_handle *h,
                           const void *const *inputs,
                           const size_t *input_sizes,
                           void *const *outputs,
                           const size_t *output_sizes,
                           uint32_t n_iterations,
                           sp_qnn_bench_result *out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SHANNON_PRIME_QNN_H */
