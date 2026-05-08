// Shannon-Prime Beast Canyon: Optane DAX Reservoir
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Monolithic mmap of a GGUF file on Optane (or any NVMe).
// At boot: one mapping call, parse tensor offsets, store direct pointers.
// At inference: zero I/O — AVX-512 Shredder reads via pre-calculated pointers.
//
// Design invariants:
//   1. ONE mmap call at boot. No file I/O during inference.
//   2. All pointers are 4KB-aligned (Optane page size).
//   3. Expert pointer table is a flat array — O(1) lookup.
//   4. Windows (MapViewOfFile) and Linux (mmap+MAP_POPULATE) paths.

#ifndef SP_OPTANE_H
#define SP_OPTANE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define SP_OPTANE_PAGE_SIZE      4096        // Optane 4KB page granularity
#define SP_OPTANE_MAX_EXPERTS_PER_LAYER 256  // Max experts per layer (Qwen3.6 has 256)
#define SP_OPTANE_MAX_LAYERS    64          // Max layers for MoE expert table
#define SP_OPTANE_MAX_EXPERTS   (SP_OPTANE_MAX_EXPERTS_PER_LAYER * SP_OPTANE_MAX_LAYERS)  // 16384 total
#define SP_OPTANE_MAX_TENSORS    32768       // Max tensors (incl. synthetic per-expert slices)
#define SP_OPTANE_PREFETCH_DIST  8           // Pages to prefetch ahead

// GGUF magic and constants (from gguf spec)
#define SP_GGUF_MAGIC            0x46554747  // "GGUF" as uint32 LE
#define SP_GGUF_VERSION_3        3

// GGUF value types
enum sp_gguf_type {
    SP_GGUF_TYPE_UINT8   = 0,
    SP_GGUF_TYPE_INT8    = 1,
    SP_GGUF_TYPE_UINT16  = 2,
    SP_GGUF_TYPE_INT16   = 3,
    SP_GGUF_TYPE_UINT32  = 4,
    SP_GGUF_TYPE_INT32   = 5,
    SP_GGUF_TYPE_FLOAT32 = 6,
    SP_GGUF_TYPE_BOOL    = 7,
    SP_GGUF_TYPE_STRING  = 8,
    SP_GGUF_TYPE_ARRAY   = 9,
    SP_GGUF_TYPE_UINT64  = 10,
    SP_GGUF_TYPE_INT64   = 11,
    SP_GGUF_TYPE_FLOAT64 = 12,
};

// ggml quantization types we care about for the Shredder
enum sp_ggml_type {
    SP_GGML_TYPE_F32     = 0,
    SP_GGML_TYPE_F16     = 1,
    SP_GGML_TYPE_Q4_0    = 2,
    SP_GGML_TYPE_Q4_1    = 3,
    SP_GGML_TYPE_Q5_0    = 6,
    SP_GGML_TYPE_Q5_1    = 7,
    SP_GGML_TYPE_Q8_0    = 8,
    SP_GGML_TYPE_Q8_1    = 9,
    SP_GGML_TYPE_Q2_K    = 10,
    SP_GGML_TYPE_Q3_K    = 11,
    SP_GGML_TYPE_Q4_K    = 12,
    SP_GGML_TYPE_Q5_K    = 13,
    SP_GGML_TYPE_Q6_K    = 14,
    SP_GGML_TYPE_Q8_K    = 15,
    SP_GGML_TYPE_IQ2_XXS = 16,
    SP_GGML_TYPE_IQ2_XS  = 17,
};

// ============================================================================
// Tensor Descriptor — parsed at boot, used at inference
// ============================================================================

typedef struct {
    char        name[128];         // Tensor name from GGUF metadata
    uint32_t    type;              // sp_ggml_type
    uint32_t    n_dims;            // Number of dimensions
    uint64_t    ne[4];             // Shape (row-major)
    uint64_t    offset;            // Byte offset from data section start
    uint64_t    n_bytes;           // Total size in bytes
    void       *ptr;               // DIRECT pointer into mmap region
} sp_optane_tensor_t;

// ============================================================================
// Expert Descriptor — for MoE routing
// ============================================================================

typedef struct {
    int          expert_id;
    int          layer;
    // Gate projection (up-proj)
    sp_optane_tensor_t *gate_proj;    // w1
    sp_optane_tensor_t *up_proj;      // w3
    sp_optane_tensor_t *down_proj;    // w2
    uint64_t     total_bytes;         // Sum of all three projections
} sp_optane_expert_t;

// ============================================================================
// Optane Reservoir — the monolithic map
// ============================================================================

typedef struct {
    // --- Memory map state ---
    void        *base_ptr;           // Base of mmap region (entire file)
    uint64_t     file_size;          // Total file size in bytes
    void        *data_ptr;           // Start of tensor data section
    uint64_t     data_offset;        // Byte offset where tensor data begins

#ifdef _WIN32
    void        *file_handle;        // HANDLE (CreateFile)
    void        *mapping_handle;     // HANDLE (CreateFileMapping)
#else
    int          fd;                 // File descriptor
#endif

    // --- GGUF metadata (parsed at boot) ---
    uint32_t     gguf_version;
    uint64_t     n_kv;               // Number of KV metadata pairs
    uint64_t     n_tensors;          // Number of tensors in file

    // --- Tensor table (flat array, O(1) by index) ---
    sp_optane_tensor_t  tensors[SP_OPTANE_MAX_TENSORS];
    uint32_t            tensor_count;

    // --- Expert table (for MoE models) ---
    sp_optane_expert_t  experts[SP_OPTANE_MAX_EXPERTS];
    int                 n_experts;
    int                 n_experts_per_token;  // Top-K routing (typically 2)

    // --- Model metadata (parsed from GGUF KV) ---
    char         architecture[64];
    uint32_t     n_layer;
    uint32_t     n_head;
    uint32_t     n_head_kv;
    uint32_t     n_embd;
    uint32_t     vocab_size;
    uint32_t     head_dim;            // n_embd / n_head
    float        rope_freq_base;     // RoPE theta (10000 default, 1000000 for Qwen3)
    float        rms_norm_eps;       // RMSNorm epsilon (1e-6 for Qwen3)
    int          n_expert_shared;    // Shared experts (Qwen3-MoE has 4)
    uint32_t     n_ff;               // Feed-forward intermediate size
    uint32_t     rope_dim;           // RoPE dimension count (mRoPE: only first rope_dim dims rotated)

    // --- Hybrid SSM+Attention (Qwen3.6 / Qwen3-Next) ---
    uint32_t     key_length;         // Per-head K dim (256 for Qwen3.6, 0 = use head_dim)
    uint32_t     value_length;       // Per-head V dim (256 for Qwen3.6, 0 = use head_dim)
    uint32_t     full_attn_interval; // Full attention every N layers (4 for Qwen3.6, 0 = all attention)
    uint32_t     ssm_state_size;     // Mamba state size (128 for Qwen3.6)
    uint32_t     ssm_conv_kernel;    // Mamba conv1d kernel (4 for Qwen3.6)
    uint32_t     ssm_inner_size;     // Mamba inner dim (4096 for Qwen3.6)
    uint32_t     ssm_n_groups;       // Mamba group count (16 for Qwen3.6)
    uint32_t     ssm_dt_rank;        // Mamba time step rank (32 for Qwen3.6)
    bool         is_hybrid;          // True if model has SSM layers

    // --- Diagnostics ---
    uint64_t     boot_map_us;         // Time to mmap the file (microseconds)
    uint64_t     boot_parse_us;       // Time to parse GGUF header
    uint64_t     boot_index_us;       // Time to build expert pointer table
    bool         dax_enabled;         // True if DAX was detected
    bool         is_moe;              // True if model has expert tensors
} sp_optane_reservoir_t;

// ============================================================================
// Public API
// ============================================================================

// Boot-time: map the GGUF file, parse metadata, build pointer table.
// Returns 0 on success, negative on error.
// After this call, all tensor data is accessible via direct pointers.
int sp_optane_init(sp_optane_reservoir_t *res, const char *gguf_path);

// Shutdown: unmap and close handles. One surgical strike.
void sp_optane_free(sp_optane_reservoir_t *res);

// --- Tensor access (O(1)) ---

// Get tensor by index. Returns NULL if out of range.
static inline const sp_optane_tensor_t *sp_optane_tensor(
    const sp_optane_reservoir_t *res, uint32_t idx)
{
    return (idx < res->tensor_count) ? &res->tensors[idx] : NULL;
}

// Get tensor by name. Linear scan — use at boot, not inference.
const sp_optane_tensor_t *sp_optane_find_tensor(
    const sp_optane_reservoir_t *res, const char *name);

// --- Expert access (O(1)) ---

// Get expert descriptor for MoE routing by (layer, expert_id).
// Returns NULL if out of range. O(1) via flat index = layer * n_experts + expert_id.
static inline const sp_optane_expert_t *sp_optane_layer_expert(
    const sp_optane_reservoir_t *res, int layer, int expert_id)
{
    if (res->n_experts <= 0 || expert_id < 0 || expert_id >= res->n_experts ||
        layer < 0 || layer >= (int)res->n_layer)
        return NULL;
    int idx = layer * res->n_experts + expert_id;
    if (idx >= SP_OPTANE_MAX_EXPERTS) return NULL;
    return &res->experts[idx];
}

// Legacy flat lookup (deprecated — use sp_optane_layer_expert for MoE).
static inline const sp_optane_expert_t *sp_optane_expert(
    const sp_optane_reservoir_t *res, int expert_id)
{
    return (expert_id >= 0 && expert_id < SP_OPTANE_MAX_EXPERTS)
        ? &res->experts[expert_id] : NULL;
}

// --- Prefetch hints ---

// Issue prefetch for an expert's weight pages into CPU cache hierarchy.
// Call this BEFORE the Shredder needs the data — overlaps with GPU compute.
void sp_optane_prefetch_expert(const sp_optane_reservoir_t *res,
                               int expert_id);

// Prefetch a specific tensor's data pages.
void sp_optane_prefetch_tensor(const sp_optane_tensor_t *tensor,
                               uint64_t offset, uint64_t length);

// --- Diagnostics ---

// Print reservoir status: mapping info, expert table, timing.
void sp_optane_print_status(const sp_optane_reservoir_t *res);

// Measure Optane-to-LLC stride latency for a 4KB page read.
// Returns latency in microseconds.
double sp_optane_measure_stride_latency(const sp_optane_reservoir_t *res);

#ifdef __cplusplus
}
#endif

#endif // SP_OPTANE_H
