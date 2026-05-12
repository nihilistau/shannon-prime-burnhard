// Shannon-Prime Engine — DFlash-draft-3.6 weight binding (implementation)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

// Suppress MSVC deprecation warnings for standard C I/O functions.
#ifdef _MSC_VER
#  ifndef _CRT_SECURE_NO_WARNINGS
#    define _CRT_SECURE_NO_WARNINGS
#  endif
#endif

#include "dflash_weights.h"

#include "ggml.h"
#include "gguf.h"

#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
#  include <sys/types.h>  // off_t for fseeko
#endif

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace sp::engine {

// ------------------------------------------------------------------
// Internal helpers — typed KV accessors over a raw gguf_context.
// Mirrors the helpers in gguf_loader.cpp but scoped to this TU.
// ------------------------------------------------------------------
static int64_t df_find_key(gguf_context* g, const char* key) {
    return gguf_find_key(g, key);
}

static uint32_t df_get_u32(gguf_context* g, const char* key, uint32_t fallback) {
    int64_t id = df_find_key(g, key);
    if (id < 0) return fallback;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT8:  return (uint32_t)gguf_get_val_u8 (g, id);
        case GGUF_TYPE_INT8:   return (uint32_t)gguf_get_val_i8 (g, id);
        case GGUF_TYPE_UINT16: return (uint32_t)gguf_get_val_u16(g, id);
        case GGUF_TYPE_INT16:  return (uint32_t)gguf_get_val_i16(g, id);
        case GGUF_TYPE_UINT32: return           gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return (uint32_t)gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (uint32_t)gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:  return (uint32_t)gguf_get_val_i64(g, id);
        default:               return fallback;
    }
}

static float df_get_f32(gguf_context* g, const char* key, float fallback) {
    int64_t id = df_find_key(g, key);
    if (id < 0) return fallback;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_FLOAT32: return           gguf_get_val_f32(g, id);
        case GGUF_TYPE_FLOAT64: return (float)   gguf_get_val_f64(g, id);
        default:                return fallback;
    }
}

// ------------------------------------------------------------------
// load_dflash_weights
// ------------------------------------------------------------------
std::unique_ptr<DFlashWeights> load_dflash_weights(const std::string& path) {
    // ---- Step 1: open GGUF with no_alloc=true to get metadata + tensor
    //             descriptors. We'll handle the actual data memory below.
    ggml_context* meta_ctx = nullptr;
    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx      = &meta_ctx;

    gguf_context* gguf_ctx = gguf_init_from_file(path.c_str(), params);
    if (!gguf_ctx) {
        std::fprintf(stderr,
            "[dflash] failed to open GGUF: %s\n", path.c_str());
        return nullptr;
    }

    // ---- Step 2: verify architecture
    {
        int64_t arch_id = gguf_find_key(gguf_ctx, "general.architecture");
        if (arch_id < 0) {
            std::fprintf(stderr,
                "[dflash] GGUF missing 'general.architecture': %s\n", path.c_str());
            if (meta_ctx) ggml_free(meta_ctx);
            gguf_free(gguf_ctx);
            return nullptr;
        }
        const char* arch = gguf_get_val_str(gguf_ctx, arch_id);
        if (!arch || std::strcmp(arch, "dflash-draft") != 0) {
            std::fprintf(stderr,
                "[dflash] wrong architecture '%s' (expected 'dflash-draft'): %s\n",
                arch ? arch : "(null)", path.c_str());
            if (meta_ctx) ggml_free(meta_ctx);
            gguf_free(gguf_ctx);
            return nullptr;
        }
    }

    auto w = std::make_unique<DFlashWeights>();

    // ---- Step 3: read metadata
    // Standard transformer hparams (using "dflash-draft." prefix)
    w->n_layers = (int)df_get_u32(gguf_ctx, "dflash-draft.block_count", 8);
    if (w->n_layers <= 0 || w->n_layers > 256) {
        std::fprintf(stderr, "[dflash] implausible n_layers=%d in '%s'\n",
                     w->n_layers, path.c_str());
        if (meta_ctx) ggml_free(meta_ctx);
        gguf_free(gguf_ctx);
        return nullptr;
    }
    w->hidden_size  = (int)df_get_u32(gguf_ctx, "dflash-draft.embedding_length",   2048);
    w->n_heads      = (int)df_get_u32(gguf_ctx, "dflash-draft.attention.head_count",  32);
    w->n_heads_kv   = (int)df_get_u32(gguf_ctx, "dflash-draft.attention.head_count_kv", 4);
    w->rope_freq_base = df_get_f32(gguf_ctx, "dflash-draft.rope.freq_base", 1e7f);

    // Derive head_dim from hidden_size / n_heads (standard formula).
    // Fallback to 128 if n_heads is 0 to avoid divide-by-zero.
    w->head_dim = (w->n_heads > 0) ? (w->hidden_size / w->n_heads) : 128;

    // DFlash-specific metadata
    // block_size: read from model but we always use our design default of 8.
    // The model stores 16; we log the model value but override it.
    {
        uint32_t model_block_size = df_get_u32(gguf_ctx,
            "dflash-draft.dflash.block_size", 16);
        std::fprintf(stderr,
            "[dflash] model block_size=%u (overriding with engine default=8)\n",
            model_block_size);
    }
    w->block_size = 8;  // engine design choice — always 8

    w->mask_token_id   = (int)df_get_u32(gguf_ctx,
        "dflash-draft.dflash.mask_token_id", 0);
    w->n_target_features = (int)df_get_u32(gguf_ctx,
        "dflash-draft.dflash.n_target_features", 0);

    // target_layer_ids: int32 array
    {
        int64_t tid = gguf_find_key(gguf_ctx, "dflash-draft.dflash.target_layer_ids");
        if (tid >= 0) {
            size_t n = gguf_get_arr_n(gguf_ctx, tid);
            const void* d = gguf_get_arr_data(gguf_ctx, tid);
            w->target_layer_ids.reserve(n);
            for (size_t i = 0; i < n; ++i)
                w->target_layer_ids.push_back((int)((const int32_t*)d)[i]);
        }
    }

    std::fprintf(stderr,
        "[dflash] metadata: n_layers=%d hidden=%d n_heads=%d n_heads_kv=%d "
        "head_dim=%d rope_base=%.0f mask_token=%d n_target_features=%d "
        "target_layers=%zu\n",
        w->n_layers, w->hidden_size, w->n_heads, w->n_heads_kv,
        w->head_dim, (double)w->rope_freq_base,
        w->mask_token_id, w->n_target_features,
        w->target_layer_ids.size());

    // ---- Step 4: compute total tensor data size and allocate
    const int64_t n_gguf_tensors = gguf_get_n_tensors(gguf_ctx);
    const size_t  data_offset    = gguf_get_data_offset(gguf_ctx);

    // Sum all tensor data bytes (needed for malloc if we copy;
    // for now we malloc and copy — no mmap in this simpler loader).
    size_t total_data_size = 0;
    for (int64_t i = 0; i < n_gguf_tensors; ++i) {
        total_data_size += gguf_get_tensor_size(gguf_ctx, i);
        total_data_size = (total_data_size + (GGML_MEM_ALIGN - 1)) & ~(size_t)(GGML_MEM_ALIGN - 1);
    }

    // Allocate a ggml_context that owns the tensor data.
    // no_alloc=false so ggml carves tensor data from mem_buffer.
    // NOTE: ggml_free() does NOT free an externally supplied mem_buffer —
    // we free ctx_data ourselves in unload_dflash_weights().
    const size_t ctx_overhead = ggml_tensor_overhead() * ((size_t)n_gguf_tensors + 16);
    const size_t mem_size     = ctx_overhead + total_data_size + (size_t)n_gguf_tensors * GGML_MEM_ALIGN;

    w->ctx_data = std::malloc(mem_size);
    if (!w->ctx_data) {
        std::fprintf(stderr,
            "[dflash] failed to allocate %.2f MiB for tensor data\n",
            mem_size / (1024.0 * 1024.0));
        if (meta_ctx) ggml_free(meta_ctx);
        gguf_free(gguf_ctx);
        return nullptr;
    }

    ggml_init_params gip = {};
    gip.mem_size   = mem_size;
    gip.mem_buffer = w->ctx_data;
    gip.no_alloc   = false;  // let ggml allocate tensor data from mem_buffer
    w->ctx = ggml_init(gip);
    if (!w->ctx) {
        std::fprintf(stderr, "[dflash] ggml_init failed\n");
        std::free(w->ctx_data);
        w->ctx_data = nullptr;
        if (meta_ctx) ggml_free(meta_ctx);
        gguf_free(gguf_ctx);
        return nullptr;
    }

    // ---- Step 5: open the file for reading tensor data and populate the
    //             ggml_context from the GGUF tensor section.
    // Re-open file to read raw tensor data bytes.
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "[dflash] failed to reopen '%s' for data read\n", path.c_str());
        ggml_free(w->ctx);
        std::free(w->ctx_data);
        w->ctx = nullptr;
        w->ctx_data = nullptr;
        if (meta_ctx) ggml_free(meta_ctx);
        gguf_free(gguf_ctx);
        return nullptr;
    }

    // Create tensors in our owned context and copy data from file.
    for (int64_t i = 0; i < n_gguf_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);

        // Get the descriptor from the no_alloc meta_ctx so we know
        // the shape and type.
        ggml_tensor* meta_t = meta_ctx ? ggml_get_tensor(meta_ctx, name) : nullptr;
        if (!meta_t) {
            // Fallback: create 1D tensor from the size.
            // This shouldn't happen if gguf_init_from_file with no_alloc
            // set ctx= correctly — but be defensive.
            std::fprintf(stderr,
                "[dflash] warning: tensor '%s' not in meta_ctx, skipping\n", name);
            continue;
        }

        // Create a tensor of the same shape in our data-owning ctx.
        ggml_tensor* t = nullptr;
        const int nd = ggml_n_dims(meta_t);
        if      (nd == 1) t = ggml_new_tensor_1d(w->ctx, meta_t->type, meta_t->ne[0]);
        else if (nd == 2) t = ggml_new_tensor_2d(w->ctx, meta_t->type, meta_t->ne[0], meta_t->ne[1]);
        else if (nd == 3) t = ggml_new_tensor_3d(w->ctx, meta_t->type, meta_t->ne[0], meta_t->ne[1], meta_t->ne[2]);
        else              t = ggml_new_tensor_4d(w->ctx, meta_t->type, meta_t->ne[0], meta_t->ne[1], meta_t->ne[2], meta_t->ne[3]);

        if (!t) {
            std::fprintf(stderr,
                "[dflash] ggml_new_tensor failed for '%s' (out of memory in ctx)\n", name);
            std::fclose(f);
            ggml_free(w->ctx);
            std::free(w->ctx_data);
            w->ctx = nullptr;
            w->ctx_data = nullptr;
            if (meta_ctx) ggml_free(meta_ctx);
            gguf_free(gguf_ctx);
            return nullptr;
        }
        ggml_set_name(t, name);

        // Read the raw bytes from the GGUF file into the tensor's data buffer.
        const size_t tensor_file_offset = data_offset + gguf_get_tensor_offset(gguf_ctx, i);
        const size_t tensor_size        = gguf_get_tensor_size(gguf_ctx, i);

#if defined(_WIN32)
        if (_fseeki64(f, (int64_t)tensor_file_offset, SEEK_SET) != 0) {
#else
        if (fseeko(f, (off_t)tensor_file_offset, SEEK_SET) != 0) {
#endif
            std::fprintf(stderr,
                "[dflash] fseek failed for tensor '%s' at offset %zu\n",
                name, tensor_file_offset);
            std::fclose(f);
            ggml_free(w->ctx);
            std::free(w->ctx_data);
            w->ctx = nullptr;
            w->ctx_data = nullptr;
            if (meta_ctx) ggml_free(meta_ctx);
            gguf_free(gguf_ctx);
            return nullptr;
        }

        const size_t n_read = std::fread(t->data, 1, tensor_size, f);
        if (n_read != tensor_size) {
            std::fprintf(stderr,
                "[dflash] short read for tensor '%s': got %zu / %zu bytes\n",
                name, n_read, tensor_size);
            std::fclose(f);
            ggml_free(w->ctx);
            std::free(w->ctx_data);
            w->ctx = nullptr;
            w->ctx_data = nullptr;
            if (meta_ctx) ggml_free(meta_ctx);
            gguf_free(gguf_ctx);
            return nullptr;
        }
    }
    std::fclose(f);
    f = nullptr;

    // meta_ctx no longer needed — all data is copied into w->ctx.
    if (meta_ctx) {
        ggml_free(meta_ctx);
        meta_ctx = nullptr;
    }
    gguf_free(gguf_ctx);
    gguf_ctx = nullptr;

    // ---- Step 6: bind tensor pointers by name
    auto bind_req = [&](const char* name) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(w->ctx, name);
        if (!t) {
            std::fprintf(stderr,
                "[dflash] required tensor missing: %s\n", name);
        }
        return t;
    };
    auto bind_opt = [&](const char* name) -> ggml_tensor* {
        return ggml_get_tensor(w->ctx, name);
    };

    // Extra non-layer tensors
    w->fc_weight    = bind_req("dflash_fc.weight");
    w->hidden_norm  = bind_req("dflash_hidden_norm.weight");
    w->output_norm  = bind_req("output_norm.weight");

    if (!w->fc_weight || !w->hidden_norm || !w->output_norm) {
        std::fprintf(stderr,
            "[dflash] one or more required global tensors are missing in '%s'\n",
            path.c_str());
        ggml_free(w->ctx);
        std::free(w->ctx_data);
        w->ctx = nullptr;
        w->ctx_data = nullptr;
        return nullptr;
    }

    // Per-layer tensors
    w->layers.resize((size_t)w->n_layers);
    char buf[128];
    bool any_layer_failed = false;

    for (int i = 0; i < w->n_layers; ++i) {
        DFlashLayer& L = w->layers[(size_t)i];

#define BIND_LAYER_REQ(field, suffix) \
        std::snprintf(buf, sizeof(buf), "blk.%d." suffix, i); \
        L.field = bind_req(buf); \
        if (!L.field) any_layer_failed = true;

#define BIND_LAYER_OPT(field, suffix) \
        std::snprintf(buf, sizeof(buf), "blk.%d." suffix, i); \
        L.field = bind_opt(buf);

        BIND_LAYER_REQ(attn_norm,     "attn_norm.weight")
        BIND_LAYER_REQ(attn_q,        "attn_q.weight")
        BIND_LAYER_REQ(attn_k,        "attn_k.weight")
        BIND_LAYER_REQ(attn_v,        "attn_v.weight")
        BIND_LAYER_REQ(attn_o,        "attn_output.weight")
        BIND_LAYER_OPT(attn_q_norm,   "attn_q_norm.weight")
        BIND_LAYER_OPT(attn_k_norm,   "attn_k_norm.weight")
        BIND_LAYER_OPT(post_attn_norm,"post_attention_norm.weight")
        BIND_LAYER_OPT(ffn_norm,      "ffn_norm.weight")
        BIND_LAYER_REQ(ffn_gate,      "ffn_gate.weight")
        BIND_LAYER_REQ(ffn_up,        "ffn_up.weight")
        BIND_LAYER_REQ(ffn_down,      "ffn_down.weight")

#undef BIND_LAYER_REQ
#undef BIND_LAYER_OPT
    }

    if (any_layer_failed) {
        std::fprintf(stderr,
            "[dflash] one or more required layer tensors are missing in '%s'\n",
            path.c_str());
        ggml_free(w->ctx);
        std::free(w->ctx_data);
        w->ctx = nullptr;
        w->ctx_data = nullptr;
        return nullptr;
    }

    std::fprintf(stderr,
        "[dflash] loaded '%s': %d layers, hidden=%d, fc_weight=[%lldx%lld]\n",
        path.c_str(), w->n_layers, w->hidden_size,
        (long long)(w->fc_weight ? w->fc_weight->ne[0] : 0),
        (long long)(w->fc_weight ? w->fc_weight->ne[1] : 0));

    return w;
}

// ------------------------------------------------------------------
// unload_dflash_weights
// ------------------------------------------------------------------
void unload_dflash_weights(DFlashWeights* w) {
    if (!w) return;
    if (w->ctx) {
        ggml_free(w->ctx);
        w->ctx = nullptr;
    }
    if (w->ctx_data) {
        std::free(w->ctx_data);
        w->ctx_data = nullptr;
    }
}

} // namespace sp::engine
