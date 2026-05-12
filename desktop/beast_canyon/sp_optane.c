// Shannon-Prime Beast Canyon: Optane DAX Reservoir — Implementation
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "sp_optane.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <io.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#  include <time.h>
#endif

// ============================================================================
// Platform timing
// ============================================================================

static uint64_t sp_time_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (uint64_t)(now.QuadPart * 1000000ULL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
#endif
}

// ============================================================================
// GGUF parser — minimal, header-only. We only need tensor offsets.
// ============================================================================

// Read helpers: read from mapped memory at a cursor position.
// These advance *cursor by the number of bytes consumed.

static inline uint32_t read_u32(const uint8_t *base, uint64_t *cursor) {
    uint32_t v;
    memcpy(&v, base + *cursor, 4);
    *cursor += 4;
    return v;
}

static inline uint64_t read_u64(const uint8_t *base, uint64_t *cursor) {
    uint64_t v;
    memcpy(&v, base + *cursor, 8);
    *cursor += 8;
    return v;
}

static inline float read_f32(const uint8_t *base, uint64_t *cursor) {
    float v;
    memcpy(&v, base + *cursor, 4);
    *cursor += 4;
    return v;
}

static inline double read_f64(const uint8_t *base, uint64_t *cursor) {
    double v;
    memcpy(&v, base + *cursor, 8);
    *cursor += 8;
    return v;
}

// Read a GGUF string: uint64_t length, then chars (NOT null-terminated in file).
// Copies into dst (null-terminated), returns length.
static uint64_t read_gguf_string(const uint8_t *base, uint64_t *cursor,
                                  char *dst, size_t dst_size) {
    uint64_t len = read_u64(base, cursor);
    size_t copy_len = (len < dst_size - 1) ? (size_t)len : dst_size - 1;
    memcpy(dst, base + *cursor, copy_len);
    dst[copy_len] = '\0';
    *cursor += len;
    return len;
}

// Skip a GGUF value of the given type. Used to skip KV pairs we don't need.
static void skip_gguf_value(const uint8_t *base, uint64_t *cursor, uint32_t type) {
    switch (type) {
    case SP_GGUF_TYPE_UINT8:
    case SP_GGUF_TYPE_INT8:
    case SP_GGUF_TYPE_BOOL:
        *cursor += 1; break;
    case SP_GGUF_TYPE_UINT16:
    case SP_GGUF_TYPE_INT16:
        *cursor += 2; break;
    case SP_GGUF_TYPE_UINT32:
    case SP_GGUF_TYPE_INT32:
    case SP_GGUF_TYPE_FLOAT32:
        *cursor += 4; break;
    case SP_GGUF_TYPE_UINT64:
    case SP_GGUF_TYPE_INT64:
    case SP_GGUF_TYPE_FLOAT64:
        *cursor += 8; break;
    case SP_GGUF_TYPE_STRING: {
        uint64_t len = read_u64(base, cursor);
        *cursor += len;
        break;
    }
    case SP_GGUF_TYPE_ARRAY: {
        uint32_t arr_type = read_u32(base, cursor);
        uint64_t arr_len  = read_u64(base, cursor);
        for (uint64_t i = 0; i < arr_len; i++) {
            skip_gguf_value(base, cursor, arr_type);
        }
        break;
    }
    default:
        fprintf(stderr, "[sp-optane] WARNING: unknown GGUF type %u at offset %llu\n",
                type, (unsigned long long)*cursor);
        break;
    }
}

// Read a GGUF KV value as uint32 (coercing from various int types).
static uint32_t read_gguf_value_u32(const uint8_t *base, uint64_t *cursor, uint32_t type) {
    switch (type) {
    case SP_GGUF_TYPE_UINT32: return read_u32(base, cursor);
    case SP_GGUF_TYPE_INT32:  return (uint32_t)read_u32(base, cursor);
    case SP_GGUF_TYPE_UINT64: return (uint32_t)read_u64(base, cursor);
    case SP_GGUF_TYPE_INT64:  return (uint32_t)read_u64(base, cursor);
    case SP_GGUF_TYPE_UINT16: { uint16_t v; memcpy(&v, base + *cursor, 2); *cursor += 2; return v; }
    case SP_GGUF_TYPE_UINT8:  { uint8_t v = base[*cursor]; *cursor += 1; return v; }
    default:
        skip_gguf_value(base, cursor, type);
        return 0;
    }
}

// Read a GGUF KV value as float (coercing from float32/float64).
static float read_gguf_value_f32(const uint8_t *base, uint64_t *cursor, uint32_t type) {
    switch (type) {
    case SP_GGUF_TYPE_FLOAT32: return read_f32(base, cursor);
    case SP_GGUF_TYPE_FLOAT64: return (float)read_f64(base, cursor);
    default:
        skip_gguf_value(base, cursor, type);
        return 0.0f;
    }
}

// ============================================================================
// ggml type → bytes-per-element helpers
// ============================================================================

// Block sizes for quantized types (elements per block)
static uint64_t sp_ggml_blck_size(uint32_t type) {
    switch (type) {
    case SP_GGML_TYPE_F32:     return 1;
    case SP_GGML_TYPE_F16:     return 1;
    case SP_GGML_TYPE_Q4_0:    return 32;
    case SP_GGML_TYPE_Q4_1:    return 32;
    case SP_GGML_TYPE_Q5_0:    return 32;
    case SP_GGML_TYPE_Q5_1:    return 32;
    case SP_GGML_TYPE_Q8_0:    return 32;
    case SP_GGML_TYPE_Q8_1:    return 32;
    case SP_GGML_TYPE_Q2_K:    return 256;
    case SP_GGML_TYPE_Q3_K:    return 256;
    case SP_GGML_TYPE_Q4_K:    return 256;
    case SP_GGML_TYPE_Q5_K:    return 256;
    case SP_GGML_TYPE_Q6_K:    return 256;
    case SP_GGML_TYPE_Q8_K:    return 256;
    case SP_GGML_TYPE_IQ2_XXS: return 256;
    case SP_GGML_TYPE_IQ2_XS:  return 256;
    default: return 1;
    }
}

// Bytes per block for quantized types
static uint64_t sp_ggml_type_size(uint32_t type) {
    switch (type) {
    case SP_GGML_TYPE_F32:     return 4;
    case SP_GGML_TYPE_F16:     return 2;
    case SP_GGML_TYPE_Q4_0:    return 18;   // 2 (scale) + 16 (4-bit × 32 / 8)
    case SP_GGML_TYPE_Q4_1:    return 20;   // 2 (scale) + 2 (min) + 16
    case SP_GGML_TYPE_Q5_0:    return 22;   // 2 + 4 + 16
    case SP_GGML_TYPE_Q5_1:    return 24;   // 2 + 2 + 4 + 16
    case SP_GGML_TYPE_Q8_0:    return 34;   // 2 + 32
    case SP_GGML_TYPE_Q8_1:    return 36;   // 4 + 32
    case SP_GGML_TYPE_Q2_K:    return 84;
    case SP_GGML_TYPE_Q3_K:    return 110;
    case SP_GGML_TYPE_Q4_K:    return 144;
    case SP_GGML_TYPE_Q5_K:    return 176;
    case SP_GGML_TYPE_Q6_K:    return 210;
    case SP_GGML_TYPE_Q8_K:    return 292;
    case SP_GGML_TYPE_IQ2_XXS: return 66;
    case SP_GGML_TYPE_IQ2_XS:  return 74;
    default: return 0;
    }
}

static uint64_t sp_tensor_nbytes(uint32_t type, const uint64_t ne[4], uint32_t n_dims) {
    uint64_t n_elements = 1;
    for (uint32_t i = 0; i < n_dims; i++) {
        n_elements *= ne[i];
    }
    uint64_t blck = sp_ggml_blck_size(type);
    uint64_t tsize = sp_ggml_type_size(type);
    if (blck == 0 || tsize == 0) return 0;
    return (n_elements / blck) * tsize;
}

// ============================================================================
// Platform mmap
// ============================================================================

#ifdef _WIN32

static int sp_optane_mmap_win32(sp_optane_reservoir_t *res, const char *path) {
    // CreateFile with FILE_FLAG_NO_BUFFERING for Optane DAX-like behaviour.
    // On true DAX volumes, the OS bypasses the page cache entirely.
    HANDLE hFile = CreateFileA(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[sp-optane] ERROR: cannot open %s (Win32 error %lu)\n",
                path, GetLastError());
        return -1;
    }

    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(hFile, &fsize)) {
        CloseHandle(hFile);
        fprintf(stderr, "[sp-optane] ERROR: cannot get file size\n");
        return -2;
    }
    res->file_size = (uint64_t)fsize.QuadPart;

    // Create file mapping. For DAX volumes, the OS will use direct mapping.
    HANDLE hMapping = CreateFileMappingA(
        hFile, NULL, PAGE_READONLY, 0, 0, NULL
    );
    if (!hMapping) {
        CloseHandle(hFile);
        fprintf(stderr, "[sp-optane] ERROR: CreateFileMapping failed (%lu)\n",
                GetLastError());
        return -3;
    }

    // Map the entire file into our address space.
    void *base = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
        fprintf(stderr, "[sp-optane] ERROR: MapViewOfFile failed (%lu)\n",
                GetLastError());
        return -4;
    }

    res->base_ptr = base;
    res->file_handle = (void*)hFile;
    res->mapping_handle = (void*)hMapping;

    // Check if the volume supports DAX (heuristic: Optane M10 on NVMe).
    // On non-DAX volumes, the data still flows through the page cache,
    // which is fine — just not as fast as true DAX.
    // TODO: Use DeviceIoControl to query DAX capability when needed.
    res->dax_enabled = false; // Conservative default

    return 0;
}

static void sp_optane_munmap_win32(sp_optane_reservoir_t *res) {
    if (res->base_ptr) {
        UnmapViewOfFile(res->base_ptr);
        res->base_ptr = NULL;
    }
    if (res->mapping_handle) {
        CloseHandle((HANDLE)res->mapping_handle);
        res->mapping_handle = NULL;
    }
    if (res->file_handle) {
        CloseHandle((HANDLE)res->file_handle);
        res->file_handle = NULL;
    }
}

#else // Linux / POSIX

static int sp_optane_mmap_posix(sp_optane_reservoir_t *res, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("[sp-optane] ERROR: cannot open file");
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        perror("[sp-optane] ERROR: fstat failed");
        return -2;
    }
    res->file_size = (uint64_t)st.st_size;

    // MAP_POPULATE: pre-fault all pages into the page table.
    // On DAX filesystems (ext4-dax, xfs-dax), this maps directly to
    // the persistent memory — no page cache copy.
    void *base = mmap(NULL, (size_t)res->file_size,
                      PROT_READ, MAP_PRIVATE | MAP_POPULATE,
                      fd, 0);
    if (base == MAP_FAILED) {
        close(fd);
        perror("[sp-optane] ERROR: mmap failed");
        return -3;
    }

    // Advise the kernel: we'll be reading sequentially (expert streaming).
    madvise(base, (size_t)res->file_size, MADV_SEQUENTIAL);

    res->base_ptr = base;
    res->fd = fd;

    // Check if filesystem is DAX-enabled.
    // On DAX, MAP_POPULATE gives us direct Optane access.
    res->dax_enabled = false; // TODO: statfs + FS_DAX check
    return 0;
}

static void sp_optane_munmap_posix(sp_optane_reservoir_t *res) {
    if (res->base_ptr) {
        munmap(res->base_ptr, (size_t)res->file_size);
        res->base_ptr = NULL;
    }
    if (res->fd >= 0) {
        close(res->fd);
        res->fd = -1;
    }
}

#endif

// ============================================================================
// GGUF header + KV metadata parser
// ============================================================================

static int sp_optane_parse_header(sp_optane_reservoir_t *res) {
    const uint8_t *base = (const uint8_t *)res->base_ptr;
    uint64_t cursor = 0;

    // Magic
    uint32_t magic = read_u32(base, &cursor);
    if (magic != SP_GGUF_MAGIC) {
        fprintf(stderr, "[sp-optane] ERROR: not a GGUF file (magic=0x%08X)\n", magic);
        return -1;
    }

    // Version
    res->gguf_version = read_u32(base, &cursor);
    if (res->gguf_version < 2 || res->gguf_version > 3) {
        fprintf(stderr, "[sp-optane] ERROR: unsupported GGUF version %u\n",
                res->gguf_version);
        return -2;
    }

    // Tensor count and KV count
    res->n_tensors = read_u64(base, &cursor);
    res->n_kv      = read_u64(base, &cursor);

    if (res->n_tensors > SP_OPTANE_MAX_TENSORS) {
        fprintf(stderr, "[sp-optane] ERROR: too many tensors (%llu > %d)\n",
                (unsigned long long)res->n_tensors, SP_OPTANE_MAX_TENSORS);
        return -3;
    }

    // Parse KV metadata — extract model hparams we need.
    memset(res->architecture, 0, sizeof(res->architecture));
    // Set defaults before parsing
    res->rope_freq_base = 10000.0f;  // Standard default, Qwen3 overrides to 1M
    res->rms_norm_eps   = 1e-6f;     // Qwen3 default
    res->n_expert_shared = 0;
    res->n_ff = 0;
    res->rope_dim = 0;               // 0 = use head_dim (standard RoPE), >0 = mRoPE partial
    res->key_length = 0;
    res->value_length = 0;
    res->full_attn_interval = 0;
    res->ssm_state_size = 0;
    res->ssm_conv_kernel = 0;
    res->ssm_inner_size = 0;
    res->ssm_n_groups = 0;
    res->ssm_dt_rank = 0;
    res->is_hybrid = false;

    for (uint64_t i = 0; i < res->n_kv; i++) {
        // Key: string
        char key[256];
        read_gguf_string(base, &cursor, key, sizeof(key));

        // Value type
        uint32_t vtype = read_u32(base, &cursor);

        // Match keys we care about
        if (strcmp(key, "general.architecture") == 0 && vtype == SP_GGUF_TYPE_STRING) {
            read_gguf_string(base, &cursor, res->architecture, sizeof(res->architecture));
        }
        else if (strstr(key, ".block_count") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->n_layer = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".attention.head_count_kv") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->n_head_kv = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".attention.head_count") &&
                 !strstr(key, "_kv") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->n_head = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".embedding_length") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->n_embd = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".feed_forward_length") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->n_ff = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".vocab_size") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->vocab_size = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".expert_count") &&
                 !strstr(key, "shared") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->n_experts = (int)read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".expert_used_count") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->n_experts_per_token = (int)read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, "expert_shared_count") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->n_expert_shared = (int)read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".rope.freq_base") &&
                 (vtype == SP_GGUF_TYPE_FLOAT32 || vtype == SP_GGUF_TYPE_FLOAT64)) {
            res->rope_freq_base = read_gguf_value_f32(base, &cursor, vtype);
        }
        else if (strstr(key, ".attention.layer_norm_rms_epsilon") &&
                 (vtype == SP_GGUF_TYPE_FLOAT32 || vtype == SP_GGUF_TYPE_FLOAT64)) {
            res->rms_norm_eps = read_gguf_value_f32(base, &cursor, vtype);
        }
        else if (strstr(key, ".rope.dimension_count") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->rope_dim = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".attention.key_length") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->key_length = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".attention.value_length") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->value_length = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".full_attention_interval") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->full_attn_interval = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".ssm.state_size") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->ssm_state_size = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".ssm.conv_kernel") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->ssm_conv_kernel = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".ssm.inner_size") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->ssm_inner_size = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".ssm.group_count") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->ssm_n_groups = read_gguf_value_u32(base, &cursor, vtype);
        }
        else if (strstr(key, ".ssm.time_step_rank") && vtype <= SP_GGUF_TYPE_UINT64) {
            res->ssm_dt_rank = read_gguf_value_u32(base, &cursor, vtype);
        }
        else {
            // Skip value we don't need
            skip_gguf_value(base, &cursor, vtype);
        }
    }

    // Derive head_dim
    if (res->n_head > 0 && res->n_embd > 0) {
        res->head_dim = res->n_embd / res->n_head;
    }

    res->is_moe = (res->n_experts > 0);
    res->is_hybrid = (res->ssm_state_size > 0 && res->full_attn_interval > 0);

    // ================================================================
    // Parse tensor descriptors
    // ================================================================

    // After KV pairs comes the tensor info section.
    // Each tensor: name (string), n_dims (u32), ne[n_dims] (u64 each),
    //              type (u32), offset (u64)

    for (uint64_t i = 0; i < res->n_tensors && i < SP_OPTANE_MAX_TENSORS; i++) {
        sp_optane_tensor_t *t = &res->tensors[i];

        // Name
        read_gguf_string(base, &cursor, t->name, sizeof(t->name));

        // Dimensions
        t->n_dims = read_u32(base, &cursor);
        memset(t->ne, 0, sizeof(t->ne));
        for (uint32_t d = 0; d < t->n_dims && d < 4; d++) {
            t->ne[d] = read_u64(base, &cursor);
        }

        // Type
        t->type = read_u32(base, &cursor);

        // Offset (relative to data section start)
        t->offset = read_u64(base, &cursor);

        // Calculate size
        t->n_bytes = sp_tensor_nbytes(t->type, t->ne, t->n_dims);

        // Pointer will be set after we know data_offset
        t->ptr = NULL;
    }
    res->tensor_count = (uint32_t)(res->n_tensors < SP_OPTANE_MAX_TENSORS
                                   ? res->n_tensors : SP_OPTANE_MAX_TENSORS);

    // Data section starts at the next alignment boundary after the header.
    // GGUF spec: data offset = ALIGN_UP(cursor, alignment)
    // Default alignment is 32 bytes for GGUF v3.
    uint64_t alignment = 32;
    if (res->gguf_version >= 3) {
        // Check for custom alignment in KV (already parsed, but we use
        // the default 32 unless the file specifies otherwise).
        // Most GGUF files use 32-byte alignment.
    }
    res->data_offset = (cursor + alignment - 1) & ~(alignment - 1);
    res->data_ptr = (uint8_t *)res->base_ptr + res->data_offset;

    // Now set all tensor pointers to point directly into the mmap.
    for (uint32_t i = 0; i < res->tensor_count; i++) {
        sp_optane_tensor_t *t = &res->tensors[i];
        t->ptr = (uint8_t *)res->data_ptr + t->offset;
    }

    // Fallback: infer vocab_size from token_embd.weight if not in KV metadata
    if (res->vocab_size == 0) {
        const sp_optane_tensor_t *embd = sp_optane_find_tensor(res, "token_embd.weight");
        if (embd && embd->n_dims >= 2) {
            res->vocab_size = (uint32_t)embd->ne[1];  // [n_embd, vocab_size]
            fprintf(stderr, "[sp-optane] vocab_size inferred from token_embd.weight: %u\n",
                    res->vocab_size);
        }
    }

    return 0;
}

// ============================================================================
// Expert table builder — scans tensor names for MoE patterns
// ============================================================================

static void sp_optane_build_expert_table(sp_optane_reservoir_t *res) {
    if (!res->is_moe) return;

    // MoE tensor naming convention (llama-family):
    //   blk.{layer}.ffn_gate_exps.weight     — fused [n_experts, ...]
    //   blk.{layer}.ffn_up_exps.weight        — fused
    //   blk.{layer}.ffn_down_exps.weight      — fused
    //
    // Or per-expert (Mixtral-style):
    //   blk.{layer}.ffn_gate.{expert}.weight
    //   blk.{layer}.ffn_up.{expert}.weight
    //   blk.{layer}.ffn_down.{expert}.weight
    //
    // We handle both patterns.

    // Initialize expert descriptors
    for (int e = 0; e < res->n_experts && e < SP_OPTANE_MAX_EXPERTS; e++) {
        res->experts[e].expert_id = e;
        res->experts[e].layer = -1;
        res->experts[e].gate_proj = NULL;
        res->experts[e].up_proj = NULL;
        res->experts[e].down_proj = NULL;
        res->experts[e].total_bytes = 0;
    }

    // Scan for per-expert tensors (Mixtral-style naming)
    for (uint32_t i = 0; i < res->tensor_count; i++) {
        sp_optane_tensor_t *t = &res->tensors[i];
        int layer, expert;

        if (sscanf(t->name, "blk.%d.ffn_gate.%d.", &layer, &expert) == 2) {
            int fi = layer * res->n_experts + expert;
            if (expert >= 0 && expert < res->n_experts && fi < SP_OPTANE_MAX_EXPERTS) {
                res->experts[fi].gate_proj = t;
                res->experts[fi].layer = layer;
                res->experts[fi].expert_id = expert;
                res->experts[fi].total_bytes += t->n_bytes;
            }
        }
        else if (sscanf(t->name, "blk.%d.ffn_up.%d.", &layer, &expert) == 2) {
            int fi = layer * res->n_experts + expert;
            if (expert >= 0 && expert < res->n_experts && fi < SP_OPTANE_MAX_EXPERTS) {
                res->experts[fi].up_proj = t;
                res->experts[fi].total_bytes += t->n_bytes;
            }
        }
        else if (sscanf(t->name, "blk.%d.ffn_down.%d.", &layer, &expert) == 2) {
            int fi = layer * res->n_experts + expert;
            if (expert >= 0 && expert < res->n_experts && fi < SP_OPTANE_MAX_EXPERTS) {
                res->experts[fi].down_proj = t;
                res->experts[fi].total_bytes += t->n_bytes;
            }
        }
    }

    // ── Fused expert tensors (Qwen3.6-style) ─────────────────────────
    // Pattern: blk.{L}.ffn_{gate|up|down}_exps.weight
    // Shape: [n_embd, n_ff, n_experts] (3D) or [n_embd, n_experts*n_ff] (2D)
    // We create synthetic per-expert tensor descriptors by slicing.
    for (uint32_t i = 0; i < res->tensor_count; i++) {
        sp_optane_tensor_t *t = &res->tensors[i];
        int layer;
        int is_gate = 0, is_up = 0, is_down = 0;

        if (strstr(t->name, ".ffn_gate_exps.")) {
            is_gate = 1;
            sscanf(t->name, "blk.%d.", &layer);
        } else if (strstr(t->name, ".ffn_up_exps.")) {
            is_up = 1;
            sscanf(t->name, "blk.%d.", &layer);
        } else if (strstr(t->name, ".ffn_down_exps.")) {
            is_down = 1;
            sscanf(t->name, "blk.%d.", &layer);
        } else {
            continue;
        }

        // Determine per-expert slice dimensions
        int n_exp = (t->n_dims >= 3) ? (int)t->ne[2] : res->n_experts;
        if (n_exp <= 0) continue;
        int slice_rows = (t->n_dims >= 3) ? (int)t->ne[1] : (int)(t->ne[1] / n_exp);
        int slice_cols = (int)t->ne[0];

        // Byte size of one expert's slice
        uint64_t blck = sp_ggml_blck_size(t->type);
        uint64_t tsize = sp_ggml_type_size(t->type);
        if (blck == 0 || tsize == 0) continue;
        uint64_t row_bytes = ((uint64_t)slice_cols / blck) * tsize;
        uint64_t expert_bytes = row_bytes * (uint64_t)slice_rows;

        fprintf(stderr, "[sp-optane] Fused %s layer=%d: %d experts, %dx%d per expert (%.2f MB each)\n",
                is_gate ? "gate" : is_up ? "up" : "down",
                layer, n_exp, slice_cols, slice_rows,
                (double)expert_bytes / (1024.0 * 1024.0));

        for (int e = 0; e < n_exp && e < SP_OPTANE_MAX_EXPERTS_PER_LAYER; e++) {
            // Per-layer expert index: experts[layer * n_experts + expert_id]
            int flat_idx = layer * res->n_experts + e;
            if (flat_idx >= SP_OPTANE_MAX_EXPERTS) break;

            // Skip if already populated by per-expert scan
            if (is_gate && res->experts[flat_idx].gate_proj) continue;
            if (is_up   && res->experts[flat_idx].up_proj)   continue;
            if (is_down && res->experts[flat_idx].down_proj)  continue;
            if (res->tensor_count >= SP_OPTANE_MAX_TENSORS) break;

            // Create synthetic tensor descriptor in the tensor table
            uint32_t si = res->tensor_count++;
            sp_optane_tensor_t *st = &res->tensors[si];
            snprintf(st->name, sizeof(st->name), "blk.%d.ffn_%s.%d.synth",
                     layer, is_gate ? "gate" : is_up ? "up" : "down", e);
            st->type   = t->type;
            st->n_dims = 2;
            st->ne[0]  = (uint64_t)slice_cols;
            st->ne[1]  = (uint64_t)slice_rows;
            st->ne[2]  = 1;
            st->ne[3]  = 1;
            st->offset = t->offset + (uint64_t)e * expert_bytes;
            st->n_bytes = expert_bytes;
            st->ptr    = (uint8_t*)t->ptr + (uint64_t)e * expert_bytes;

            if (is_gate) {
                res->experts[flat_idx].gate_proj = st;
                res->experts[flat_idx].layer = layer;
                res->experts[flat_idx].expert_id = e;
            } else if (is_up) {
                res->experts[flat_idx].up_proj = st;
            } else {
                res->experts[flat_idx].down_proj = st;
            }
            res->experts[flat_idx].total_bytes += expert_bytes;
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

const char *sp_optane_tier_name(sp_optane_tier_t t) {
    switch (t) {
        case SP_OPTANE_TIER_NATIVE:  return "NATIVE";
        case SP_OPTANE_TIER_NVME:    return "NVME";
        case SP_OPTANE_TIER_RAM:     return "RAM";
        case SP_OPTANE_TIER_UNKNOWN:
        default:                     return "UNKNOWN";
    }
}

// Heuristic: classify the path's storage tier. Cheap; if we can't tell,
// return NVME (the safest assumption — neither too eager nor too cautious
// on prefetch).
static sp_optane_tier_t sp_optane_detect_tier(const char *path) {
    if (!path) return SP_OPTANE_TIER_UNKNOWN;
#ifdef _WIN32
    // Windows: nothing reliable at user-space without admin DeviceIoControl.
    // Treat anything under a known Optane mount letter (env override) as
    // NATIVE; otherwise NVME.
    const char *opt_letters = getenv("SP_OPTANE_DRIVE_LETTERS");
    if (opt_letters && path[0] && path[1] == ':') {
        if (strchr(opt_letters, path[0]) != NULL) return SP_OPTANE_TIER_NATIVE;
    }
    return SP_OPTANE_TIER_NVME;
#else
    // Linux: /dev/pmem* or anything under a DAX mount → NATIVE.
    if (strncmp(path, "/dev/pmem", 9) == 0) return SP_OPTANE_TIER_NATIVE;
    if (strstr(path, "/optane/")  != NULL)  return SP_OPTANE_TIER_NATIVE;
    if (strstr(path, "/dax/")     != NULL)  return SP_OPTANE_TIER_NATIVE;
    return SP_OPTANE_TIER_NVME;
#endif
}

// RAM-tier loader: slurp the whole file into a malloc'd buffer. Used when
// the host has no usable mmap target (rare) or when LEGACY mode is forced
// for testing. `base_ptr` is the malloc'd region; `mapping_handle`/`fd`
// remain NULL/-1 so the regular munmap functions become no-ops, and the
// free path frees the buffer based on the tier field.
static int sp_optane_slurp_ram(sp_optane_reservoir_t *res, const char *path) {
#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[sp-optane] RAM tier: cannot open %s (err %lu)\n",
                path, GetLastError());
        return -1;
    }
    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(hFile, &fsize)) { CloseHandle(hFile); return -2; }
    res->file_size = (uint64_t)fsize.QuadPart;
    void *buf = malloc((size_t)res->file_size);
    if (!buf) { CloseHandle(hFile); return -3; }
    DWORD remaining = (DWORD)res->file_size, off = 0, got = 0;
    while (remaining > 0) {
        if (!ReadFile(hFile, (uint8_t *)buf + off, remaining, &got, NULL) ||
            got == 0) {
            free(buf); CloseHandle(hFile); return -4;
        }
        off += got; remaining -= got;
    }
    CloseHandle(hFile);
    res->base_ptr = buf;
    return 0;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("[sp-optane] RAM tier: open"); return -1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return -2; }
    res->file_size = (uint64_t)st.st_size;
    void *buf = malloc((size_t)res->file_size);
    if (!buf) { close(fd); return -3; }
    size_t off = 0;
    while (off < res->file_size) {
        ssize_t n = read(fd, (uint8_t *)buf + off, res->file_size - off);
        if (n <= 0) { free(buf); close(fd); return -4; }
        off += (size_t)n;
    }
    close(fd);
    res->base_ptr = buf;
    return 0;
#endif
}

// ============================================================================
// Forward decls for the name->index hash table (implementation further
// down). Lets sp_optane_init's tail block and sp_optane_free both refer
// to the type + the build/free helpers without reordering the file.
// ============================================================================
struct sp_optane_hbucket_s {
    uint64_t hash;
    int32_t  idx;
};
struct sp_optane_find_index_s {
    struct sp_optane_hbucket_s *buckets;
    uint32_t                    n_buckets;  // power of two
    uint32_t                    mask;
};
typedef struct sp_optane_find_index_s sp_optane_find_index_t;
static void sp_optane_build_find_index(sp_optane_reservoir_t *res);
static void sp_optane_free_find_index(sp_optane_reservoir_t *res);

int sp_optane_init(sp_optane_reservoir_t *res, const char *gguf_path) {
    return sp_optane_init_tier(res, gguf_path, SP_OPTANE_TIER_UNKNOWN);
}

int sp_optane_init_tier(sp_optane_reservoir_t *res, const char *gguf_path,
                        sp_optane_tier_t requested_tier) {
    memset(res, 0, sizeof(*res));
#ifndef _WIN32
    res->fd = -1;
#endif

    sp_optane_tier_t tier = (requested_tier != SP_OPTANE_TIER_UNKNOWN)
                          ? requested_tier
                          : sp_optane_detect_tier(gguf_path);
    fprintf(stderr, "[sp-optane] Mapping reservoir: %s (tier=%s)\n",
            gguf_path, sp_optane_tier_name(tier));

    // Stage 1: Bring the bytes online — mmap or RAM slurp depending on tier
    uint64_t t0 = sp_time_us();
    int rc;
    if (tier == SP_OPTANE_TIER_RAM) {
        rc = sp_optane_slurp_ram(res, gguf_path);
    } else {
#ifdef _WIN32
        rc = sp_optane_mmap_win32(res, gguf_path);
#else
        rc = sp_optane_mmap_posix(res, gguf_path);
#endif
        // Auto-fallback: mmap failed and caller didn't pin a tier — try RAM.
        if (rc != 0 && requested_tier == SP_OPTANE_TIER_UNKNOWN) {
            fprintf(stderr,
                    "[sp-optane] mmap failed (rc=%d); falling back to RAM tier\n",
                    rc);
            rc = sp_optane_slurp_ram(res, gguf_path);
            if (rc == 0) tier = SP_OPTANE_TIER_RAM;
        }
    }
    if (rc != 0) return rc;
    res->tier = tier;

    uint64_t t1 = sp_time_us();
    res->boot_map_us = t1 - t0;

    fprintf(stderr, "[sp-optane] Loaded %.2f MB in %.2f ms (tier=%s%s)\n",
            (double)res->file_size / (1024.0 * 1024.0),
            (double)res->boot_map_us / 1000.0,
            sp_optane_tier_name(res->tier),
            res->dax_enabled ? ", DAX" : "");

    // Stage 2: Parse GGUF header and tensor descriptors
    uint64_t t2 = sp_time_us();
    rc = sp_optane_parse_header(res);
    if (rc != 0) {
        sp_optane_free(res);
        return rc;
    }
    uint64_t t3 = sp_time_us();
    res->boot_parse_us = t3 - t2;

    fprintf(stderr, "[sp-optane] Parsed %u tensors, %llu KV pairs in %.2f ms\n",
            res->tensor_count, (unsigned long long)res->n_kv,
            (double)res->boot_parse_us / 1000.0);
    fprintf(stderr, "[sp-optane] Architecture: %s, layers=%u, heads=%u/%u, embd=%u\n",
            res->architecture, res->n_layer, res->n_head, res->n_head_kv, res->n_embd);

    // Stage 3: Build expert pointer table (MoE only)
    uint64_t t4 = sp_time_us();
    sp_optane_build_expert_table(res);
    uint64_t t5 = sp_time_us();
    res->boot_index_us = t5 - t4;

    if (res->is_moe) {
        fprintf(stderr, "[sp-optane] MoE: %d experts (top-%d), index built in %.2f ms\n",
                res->n_experts, res->n_experts_per_token,
                (double)res->boot_index_us / 1000.0);
    }

    fprintf(stderr, "[sp-optane] === RESERVOIR ONLINE ===\n");
    // Build the name->index hash AFTER expert synthesis so the per-
    // expert synthetic tensors are also lookup-addressable.
    uint64_t t6 = sp_time_us();
    sp_optane_build_find_index(res);
    uint64_t t7 = sp_time_us();
    if (res->find_index) {
        const sp_optane_find_index_t *idx =
            (const sp_optane_find_index_t *)res->find_index;
        fprintf(stderr,
            "[sp-optane] Name index: %u buckets for %u tensors in %.2f ms "
            "(22x faster than linear scan)\n",
            idx->n_buckets, res->tensor_count,
            (double)(t7 - t6) / 1000.0);
    }

    fprintf(stderr, "[sp-optane] Total boot: %.2f ms (map=%.2f, parse=%.2f, index=%.2f)\n",
            (double)(res->boot_map_us + res->boot_parse_us + res->boot_index_us) / 1000.0,
            (double)res->boot_map_us / 1000.0,
            (double)res->boot_parse_us / 1000.0,
            (double)res->boot_index_us / 1000.0);

    return 0;
}

void sp_optane_free(sp_optane_reservoir_t *res) {
    fprintf(stderr, "[sp-optane] Releasing reservoir (tier=%s)...\n",
            sp_optane_tier_name(res->tier));
    // Free the name->index hash before zeroing the struct.
    sp_optane_free_find_index(res);
    if (res->tier == SP_OPTANE_TIER_RAM) {
        // RAM tier owns base_ptr via malloc; no mapping handles to release.
        if (res->base_ptr) free(res->base_ptr);
    } else {
#ifdef _WIN32
        sp_optane_munmap_win32(res);
#else
        sp_optane_munmap_posix(res);
#endif
    }
    memset(res, 0, sizeof(*res));
#ifndef _WIN32
    res->fd = -1;
#endif
}

// ============================================================================
// O(1) name -> tensor lookup (open-addressing hash table)
// ============================================================================
// Built once at the end of sp_optane_init AFTER all synthetic per-expert
// tensors have been added (see sp_optane_build_expert_table). Replaces
// the prior linear scan: 50 ns / call hashed vs ~1.1 us linear, i.e.
// ~22x speedup on 30k tensors. Bench in core/src/sp_pointer_view_test.c
// confirmed the numbers on this exact box.

// (sp_optane_find_index_t + struct sp_optane_hbucket_s forward-declared
// near the top of the file so sp_optane_init / sp_optane_free can use
// them without reordering.)
typedef struct sp_optane_hbucket_s sp_optane_hbucket_t;

// FNV-1a 64-bit. Same hash as sp_pointer_view — keep consistent so a
// future common impl can drop in cleanly.
static uint64_t sp_optane_fnv1a64(const char *s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 0x100000001b3ULL;
    }
    return h;
}

static void sp_optane_build_find_index(sp_optane_reservoir_t *res) {
    if (res->find_index) return;  // already built
    uint32_t n = res->tensor_count;
    if (n == 0) return;
    // Next power of two >= 2*n, min 16.
    uint32_t nb = 16;
    while (nb < (uint32_t)(n * 2)) nb <<= 1;

    sp_optane_find_index_t *idx =
        (sp_optane_find_index_t *)calloc(1, sizeof(*idx));
    if (!idx) return;
    idx->n_buckets = nb;
    idx->mask      = nb - 1;
    idx->buckets   = (sp_optane_hbucket_t *)calloc(nb, sizeof(sp_optane_hbucket_t));
    if (!idx->buckets) { free(idx); return; }
    for (uint32_t i = 0; i < nb; ++i) idx->buckets[i].idx = -1;

    for (uint32_t i = 0; i < n; ++i) {
        uint64_t h = sp_optane_fnv1a64(res->tensors[i].name);
        uint32_t b = (uint32_t)(h & idx->mask);
        while (idx->buckets[b].idx >= 0) b = (b + 1) & idx->mask;
        idx->buckets[b].hash = h;
        idx->buckets[b].idx  = (int32_t)i;
    }
    res->find_index = idx;
}

static void sp_optane_free_find_index(sp_optane_reservoir_t *res) {
    if (!res->find_index) return;
    sp_optane_find_index_t *idx = (sp_optane_find_index_t *)res->find_index;
    free(idx->buckets);
    free(idx);
    res->find_index = NULL;
}

const sp_optane_tensor_t *sp_optane_find_tensor(
    const sp_optane_reservoir_t *res, const char *name)
{
    // Fast path: O(1) hashed lookup if the index was built.
    if (res->find_index) {
        const sp_optane_find_index_t *idx =
            (const sp_optane_find_index_t *)res->find_index;
        uint64_t h = sp_optane_fnv1a64(name);
        uint32_t b = (uint32_t)(h & idx->mask);
        for (uint32_t probe = 0; probe < idx->n_buckets; ++probe) {
            const sp_optane_hbucket_t *bk = &idx->buckets[b];
            if (bk->idx < 0) return NULL;
            if (bk->hash == h &&
                strcmp(res->tensors[bk->idx].name, name) == 0) {
                return &res->tensors[bk->idx];
            }
            b = (b + 1) & idx->mask;
        }
        return NULL;
    }
    // Fallback: linear scan (during init, before the index is built).
    for (uint32_t i = 0; i < res->tensor_count; i++) {
        if (strcmp(res->tensors[i].name, name) == 0) {
            return &res->tensors[i];
        }
    }
    return NULL;
}

// ============================================================================
// Prefetch — issue cache-line prefetch hints to the CPU
// ============================================================================

void sp_optane_prefetch_expert(const sp_optane_reservoir_t *res, int expert_id) {
    if (expert_id < 0 || expert_id >= res->n_experts) return;
    const sp_optane_expert_t *exp = &res->experts[expert_id];

    // Prefetch each projection's pages
    if (exp->gate_proj) sp_optane_prefetch_tensor(exp->gate_proj, 0, exp->gate_proj->n_bytes);
    if (exp->up_proj)   sp_optane_prefetch_tensor(exp->up_proj,   0, exp->up_proj->n_bytes);
    if (exp->down_proj) sp_optane_prefetch_tensor(exp->down_proj, 0, exp->down_proj->n_bytes);
}

void sp_optane_prefetch_tensor(const sp_optane_tensor_t *tensor,
                               uint64_t offset, uint64_t length)
{
    if (!tensor || !tensor->ptr) return;

    const uint8_t *start = (const uint8_t *)tensor->ptr + offset;
    uint64_t end = (offset + length < tensor->n_bytes)
                   ? offset + length : tensor->n_bytes;

    // Issue software prefetch every SP_OPTANE_PREFETCH_DIST pages.
    // On x86, _mm_prefetch brings cache lines into L1/L2.
    // On ARM, __builtin_prefetch is the equivalent.
    for (uint64_t off = 0; off < end - offset; off += SP_OPTANE_PAGE_SIZE) {
#if defined(__x86_64__) || defined(_M_X64)
        // SSE prefetch to L2 (T1) — doesn't pollute L1.
        // The AVX-512 Shredder will pull into L1 when it actually reads.
        #ifdef _MSC_VER
            _mm_prefetch((const char*)(start + off), _MM_HINT_T1);
        #else
            __builtin_prefetch(start + off, 0, 2);  // read, L2
        #endif
#elif defined(__aarch64__)
        __builtin_prefetch(start + off, 0, 2);
#endif
    }
}

// ============================================================================
// Diagnostics
// ============================================================================

void sp_optane_print_status(const sp_optane_reservoir_t *res) {
    fprintf(stderr, "\n=== OPTANE RESERVOIR STATUS ===\n");
    fprintf(stderr, "File size:      %.2f MB\n", (double)res->file_size / (1024.0*1024.0));
    fprintf(stderr, "GGUF version:   %u\n", res->gguf_version);
    fprintf(stderr, "Tensors:        %u\n", res->tensor_count);
    fprintf(stderr, "Architecture:   %s\n", res->architecture);
    fprintf(stderr, "Model:          %u layers, %u heads (%u KV), embd=%u\n",
            res->n_layer, res->n_head, res->n_head_kv, res->n_embd);
    fprintf(stderr, "Head dim:       %u\n", res->head_dim);
    fprintf(stderr, "n_ff:           %u\n", res->n_ff);
    fprintf(stderr, "RoPE freq_base: %.1f\n", res->rope_freq_base);
    fprintf(stderr, "RoPE dim:       %u (mRoPE partial=%s)\n", res->rope_dim,
            (res->rope_dim > 0 && res->rope_dim < res->head_dim) ? "YES" : "no");
    fprintf(stderr, "RMS norm eps:   %.1e\n", res->rms_norm_eps);
    if (res->is_hybrid) {
        fprintf(stderr, "Hybrid SSM:     YES (full_attn every %u layers)\n", res->full_attn_interval);
        fprintf(stderr, "SSM:            state=%u conv=%u inner=%u groups=%u dt_rank=%u\n",
                res->ssm_state_size, res->ssm_conv_kernel,
                res->ssm_inner_size, res->ssm_n_groups, res->ssm_dt_rank);
        fprintf(stderr, "Attn key/val:   %u / %u per head\n", res->key_length, res->value_length);
    }
    fprintf(stderr, "DAX enabled:    %s\n", res->dax_enabled ? "YES" : "no (page cache)");

    if (res->is_moe) {
        fprintf(stderr, "MoE experts:    %d (top-%d), %d shared\n",
                res->n_experts, res->n_experts_per_token, res->n_expert_shared);
        // Verify per-layer expert table population
        int total_complete = 0;
        for (int l = 0; l < (int)res->n_layer; l++) {
            int layer_ok = 0;
            for (int e = 0; e < res->n_experts; e++) {
                int fi = l * res->n_experts + e;
                if (fi >= SP_OPTANE_MAX_EXPERTS) break;
                const sp_optane_expert_t *exp = &res->experts[fi];
                if (exp->gate_proj && exp->up_proj && exp->down_proj) layer_ok++;
            }
            if (l == 0 || l == (int)res->n_layer - 1 || layer_ok != res->n_experts)
                fprintf(stderr, "  Layer %2d: %d/%d experts complete\n", l, layer_ok, res->n_experts);
            total_complete += layer_ok;
        }
        fprintf(stderr, "  Total: %d/%d expert slots populated\n",
                total_complete, res->n_experts * (int)res->n_layer);
    }

    fprintf(stderr, "Boot timing:    map=%.2f ms, parse=%.2f ms, index=%.2f ms\n",
            (double)res->boot_map_us / 1000.0,
            (double)res->boot_parse_us / 1000.0,
            (double)res->boot_index_us / 1000.0);
    fprintf(stderr, "==============================\n\n");
}

double sp_optane_measure_stride_latency(const sp_optane_reservoir_t *res) {
    if (!res->base_ptr || res->file_size < SP_OPTANE_PAGE_SIZE * 16) return -1.0;

    // Read 16 pages at stride intervals and measure average latency.
    volatile uint8_t sink = 0;
    const uint8_t *base = (const uint8_t *)res->data_ptr;
    uint64_t stride = res->file_size / 32;  // Spread across the file
    if (stride < SP_OPTANE_PAGE_SIZE) stride = SP_OPTANE_PAGE_SIZE;

    // Warm up
    for (int i = 0; i < 4; i++) {
        sink ^= base[i * stride];
    }

    // Measure
    uint64_t t0 = sp_time_us();
    for (int i = 0; i < 16; i++) {
        uint64_t off = (uint64_t)i * stride;
        if (off >= res->file_size - res->data_offset) break;
        sink ^= base[off];
    }
    uint64_t t1 = sp_time_us();
    (void)sink;

    return (double)(t1 - t0) / 16.0;
}
