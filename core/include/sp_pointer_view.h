// sp_pointer_view.h — Lightweight mmap'd GGUF "pointer view".
// ----------------------------------------------------------------------------
// Maps a GGUF file into the address space, parses the tensor table once,
// and exposes O(1) hashed tensor lookup + platform-native prefetch / evict
// hints. ~250 LOC, no external dependencies, Win + POSIX.
//
// Replaces the linear-scan sp_optane_find_tensor() (~2.5 us / call on 30k
// tensors) with a hash lookup (~50 ns / call). Designed to coexist with
// the existing sp_optane_reservoir_t — burnhard code can keep using
// sp_optane for the rich expert-routing state while pointing-view-aware
// callers (sp-engine fallback, shannon-prime-llama bridge, benchmarks)
// take the fast path.
//
// Hints route to:
//   POSIX: madvise(MADV_*)
//   Win  : PrefetchVirtualMemory + VirtualFree(MEM_DECOMMIT for DONTNEED)
//          (best-effort — Windows lacks a true MADV_DONTNEED equivalent
//          on file-backed mappings, so DONTNEED is a no-op there)
//
// Quant types use the same enum values as sp_optane / ggml (Q4_K=12,
// Q6_K=14, etc.) to make pointer-view + sp_optane code interchangeable.

#ifndef SP_POINTER_VIEW_H
#define SP_POINTER_VIEW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----- Tensor descriptor (public, copy-by-value safe) ------------------
typedef struct sp_pv_tensor_s {
    char     name[128];     // null-terminated GGUF tensor name
    void    *ptr;           // direct pointer into the mmap region
    uint32_t dtype;         // ggml_type / sp_ggml_type
    uint32_t n_dims;
    uint64_t ne[4];         // shape (lowest-dim first, row-major)
    uint64_t n_bytes;       // total tensor size in bytes
    uint64_t offset;        // file offset of the tensor data
} sp_pv_tensor_t;

// ----- Opaque handle ---------------------------------------------------
typedef struct sp_pointer_view_s sp_pointer_view_t;

// ----- Hint codes ------------------------------------------------------
typedef enum {
    SP_PV_HINT_WILLNEED   = 1,   // page cache prefetch (warm before use)
    SP_PV_HINT_DONTNEED   = 2,   // can be evicted (POSIX only — see notes)
    SP_PV_HINT_RANDOM     = 3,   // random access pattern
    SP_PV_HINT_SEQUENTIAL = 4,   // sequential access pattern
} sp_pv_hint_t;

// ----- Lifecycle -------------------------------------------------------
//
// Returns NULL on:
//   - file not found / not openable
//   - mmap failure
//   - not a GGUF (bad magic) / unsupported version
//   - tensor table parse failure
//
// The view holds the file open via mmap for its lifetime. Closing the
// view munmaps and releases the OS handle.
sp_pointer_view_t *sp_pv_open(const char *path);
void               sp_pv_close(sp_pointer_view_t *pv);

// ----- Lookup ----------------------------------------------------------
// O(1) hashed lookup; returns NULL if `name` not in the file.
const sp_pv_tensor_t *sp_pv_get(const sp_pointer_view_t *pv, const char *name);

// Iterate. Callback returns non-zero to stop. Returns the number of
// tensors visited.
int sp_pv_iterate(const sp_pointer_view_t *pv,
                   int (*cb)(const sp_pv_tensor_t *t, void *user),
                   void *user);

// ----- Hints -----------------------------------------------------------
// All best-effort. Errors are silent. Both forms operate on file-backed
// pages.
void sp_pv_advise_tensor(const sp_pointer_view_t *pv,
                          const char *name,
                          sp_pv_hint_t hint);
void sp_pv_advise_range(const sp_pointer_view_t *pv,
                         void *base, size_t len,
                         sp_pv_hint_t hint);

// ----- Stats / introspection ------------------------------------------
size_t   sp_pv_file_size(const sp_pointer_view_t *pv);
uint32_t sp_pv_tensor_count(const sp_pointer_view_t *pv);
void    *sp_pv_data_section(const sp_pointer_view_t *pv);   // start of tensor data
const char *sp_pv_path(const sp_pointer_view_t *pv);

#ifdef __cplusplus
}
#endif

#endif // SP_POINTER_VIEW_H
