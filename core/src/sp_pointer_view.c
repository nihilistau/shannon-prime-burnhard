// sp_pointer_view.c — implementation. See sp_pointer_view.h for the API.
//
// Minimal GGUF v2/v3 parser + open-addressing hash table + cross-platform
// mmap + advisory hints. ~330 LOC.

#include "sp_pointer_view.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <memoryapi.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

// ============================================================================
// GGUF wire format constants
// ============================================================================
#define SP_GGUF_MAGIC 0x46554747u   // "GGUF" in little-endian
#define SP_GGUF_DATA_ALIGN 32       // GGUF default data alignment

// GGUF metadata value types (the subset we have to skip past)
enum {
    GGUF_TYPE_UINT8   = 0,  GGUF_TYPE_INT8   = 1,
    GGUF_TYPE_UINT16  = 2,  GGUF_TYPE_INT16  = 3,
    GGUF_TYPE_UINT32  = 4,  GGUF_TYPE_INT32  = 5,
    GGUF_TYPE_FLOAT32 = 6,  GGUF_TYPE_BOOL   = 7,
    GGUF_TYPE_STRING  = 8,  GGUF_TYPE_ARRAY  = 9,
    GGUF_TYPE_UINT64  = 10, GGUF_TYPE_INT64  = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// Sizes of fixed-width GGUF metadata scalars (bytes).
static const size_t k_gguf_scalar_size[13] = {
    1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8,
};

// ============================================================================
// Hash table (open addressing, linear probing)
// ============================================================================
// Capacity is the next power-of-two >= 2 * n_tensors. Tombstones unused —
// we never delete after build, so probe sequence stays monotone.

typedef struct {
    uint64_t hash;
    int32_t  idx;     // index into pv->tensors[], or -1 if empty
} sp_pv_hbucket_t;

// FNV-1a 64-bit. Sufficient for names; collisions are resolved by probe.
static uint64_t fnv1a64(const char *s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 0x100000001b3ULL;
    }
    return h;
}

// ============================================================================
// View struct
// ============================================================================
struct sp_pointer_view_s {
    void    *base;          // mmap base
    size_t   file_size;
    char    *path;          // strdup'd

#if defined(_WIN32)
    HANDLE   hFile;
    HANDLE   hMap;
#else
    int      fd;
#endif

    void    *data_section;  // start of tensor data within mmap

    sp_pv_tensor_t  *tensors;   // contiguous array
    uint32_t         n_tensors;

    sp_pv_hbucket_t *buckets;
    uint32_t         n_buckets; // power of two
    uint32_t         mask;      // n_buckets - 1
};

// ============================================================================
// Safe little-endian readers from a moving cursor
// ============================================================================
typedef struct {
    const uint8_t *p;
    const uint8_t *end;
    int            err;
} cur_t;

static uint8_t  rd_u8 (cur_t *c) {
    if (c->err || c->p + 1 > c->end) { c->err = 1; return 0; }
    return *c->p++;
}
static uint32_t rd_u32(cur_t *c) {
    if (c->err || c->p + 4 > c->end) { c->err = 1; return 0; }
    uint32_t v;
    memcpy(&v, c->p, 4); c->p += 4; return v;
}
static uint64_t rd_u64(cur_t *c) {
    if (c->err || c->p + 8 > c->end) { c->err = 1; return 0; }
    uint64_t v;
    memcpy(&v, c->p, 8); c->p += 8; return v;
}
static void rd_skip(cur_t *c, size_t n) {
    if (c->err || c->p + n > c->end) { c->err = 1; return; }
    c->p += n;
}
// GGUF strings are u64 length + bytes (NOT null-terminated).
static void rd_skip_string(cur_t *c) {
    uint64_t n = rd_u64(c);
    rd_skip(c, (size_t)n);
}
static int rd_copy_string(cur_t *c, char *dst, size_t dst_cap) {
    uint64_t n = rd_u64(c);
    if (c->err) return -1;
    if (c->p + n > c->end) { c->err = 1; return -1; }
    size_t copy = (n < dst_cap - 1) ? (size_t)n : dst_cap - 1;
    memcpy(dst, c->p, copy);
    dst[copy] = '\0';
    c->p += n;
    return 0;
}

// Skip one GGUF metadata value of the given type (recursive for ARRAY).
static void skip_gguf_value(cur_t *c, uint32_t type);
static void skip_gguf_value(cur_t *c, uint32_t type) {
    if (c->err) return;
    if (type <= 7 || type == 10 || type == 11 || type == 12) {
        rd_skip(c, k_gguf_scalar_size[type]);
    } else if (type == GGUF_TYPE_STRING) {
        rd_skip_string(c);
    } else if (type == GGUF_TYPE_ARRAY) {
        uint32_t inner = rd_u32(c);
        uint64_t count = rd_u64(c);
        for (uint64_t i = 0; i < count && !c->err; ++i)
            skip_gguf_value(c, inner);
    } else {
        c->err = 1;
    }
}

// ============================================================================
// Quant-aware tensor byte-size computation.
// Mirrors ggml block sizes for the K-quants this engine cares about.
// Other types fall through to a "best-effort" calc — adequate for pointer
// arithmetic since the data layout is contiguous in the file regardless.
// ============================================================================
static uint64_t tensor_n_bytes(uint32_t dtype,
                                const uint64_t *ne, uint32_t n_dims)
{
    uint64_t n_elements = 1;
    for (uint32_t d = 0; d < n_dims; ++d) n_elements *= ne[d];

    // (blk_size_in_bytes, n_elements_per_block) — same numbers ggml uses.
    static const struct { uint32_t type; uint32_t blk_bytes; uint32_t blk_n; } sz[] = {
        { 0, 4, 1 },         // F32
        { 1, 2, 1 },         // F16
        { 2, 18, 32 },       // Q4_0
        { 3, 20, 32 },       // Q4_1
        { 6, 22, 32 },       // Q5_0
        { 7, 24, 32 },       // Q5_1
        { 8, 34, 32 },       // Q8_0
        { 9, 36, 32 },       // Q8_1
        { 10, 84, 256 },     // Q2_K
        { 11, 110, 256 },    // Q3_K
        { 12, 144, 256 },    // Q4_K
        { 13, 176, 256 },    // Q5_K
        { 14, 210, 256 },    // Q6_K
        { 15, 256, 256 },    // Q8_K
        { 16, 66, 256 },     // IQ2_XXS
        { 17, 74, 256 },     // IQ2_XS
        { 30, 2, 1 },        // BF16
    };
    for (size_t i = 0; i < sizeof(sz)/sizeof(sz[0]); ++i) {
        if (sz[i].type == dtype)
            return (n_elements / sz[i].blk_n) * sz[i].blk_bytes;
    }
    // Unknown -> assume 1 byte per element (lets us still expose .ptr;
    // the caller is responsible for understanding the dtype).
    return n_elements;
}

// ============================================================================
// Cross-platform mmap helpers
// ============================================================================
static int mmap_open(const char *path, sp_pointer_view_t *pv) {
#if defined(_WIN32)
    pv->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                              OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
                              NULL);
    if (pv->hFile == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(pv->hFile, &sz)) { CloseHandle(pv->hFile); return -1; }
    pv->file_size = (size_t)sz.QuadPart;
    pv->hMap = CreateFileMappingA(pv->hFile, NULL, PAGE_READONLY,
                                    sz.HighPart, sz.LowPart, NULL);
    if (!pv->hMap) { CloseHandle(pv->hFile); return -1; }
    pv->base = MapViewOfFile(pv->hMap, FILE_MAP_READ, 0, 0, 0);
    if (!pv->base) { CloseHandle(pv->hMap); CloseHandle(pv->hFile); return -1; }
    return 0;
#else
    pv->fd = open(path, O_RDONLY);
    if (pv->fd < 0) return -1;
    struct stat st;
    if (fstat(pv->fd, &st) != 0) { close(pv->fd); return -1; }
    pv->file_size = (size_t)st.st_size;
    pv->base = mmap(NULL, pv->file_size, PROT_READ, MAP_PRIVATE, pv->fd, 0);
    if (pv->base == MAP_FAILED) { close(pv->fd); return -1; }
    return 0;
#endif
}

static void mmap_close(sp_pointer_view_t *pv) {
#if defined(_WIN32)
    if (pv->base) UnmapViewOfFile(pv->base);
    if (pv->hMap) CloseHandle(pv->hMap);
    if (pv->hFile && pv->hFile != INVALID_HANDLE_VALUE) CloseHandle(pv->hFile);
#else
    if (pv->base && pv->base != MAP_FAILED) munmap(pv->base, pv->file_size);
    if (pv->fd >= 0) close(pv->fd);
#endif
}

// ============================================================================
// Public API
// ============================================================================
// Set SP_PV_TRACE=1 in env to dump per-stage progress to stderr.
static int sp_pv_trace_on(void) {
    const char *e = getenv("SP_PV_TRACE");
    return e && e[0] == '1';
}
#define TRC(fmt, ...) do { if (sp_pv_trace_on()) { fprintf(stderr, "[pv] " fmt "\n", ##__VA_ARGS__); fflush(stderr); } } while (0)

sp_pointer_view_t *sp_pv_open(const char *path) {
    if (!path) return NULL;

    sp_pointer_view_t *pv = (sp_pointer_view_t *)calloc(1, sizeof(*pv));
    if (!pv) return NULL;
#if !defined(_WIN32)
    pv->fd = -1;
#endif
    pv->path = NULL;

    TRC("opening %s", path);
    if (mmap_open(path, pv) != 0) { TRC("mmap_open FAILED"); free(pv); return NULL; }
    TRC("mmap ok base=%p size=%zu", pv->base, pv->file_size);

    cur_t c = { (const uint8_t *)pv->base,
                 (const uint8_t *)pv->base + pv->file_size, 0 };

    uint32_t magic = rd_u32(&c);
    TRC("magic=0x%08x (expect 0x%08x)", magic, SP_GGUF_MAGIC);
    if (magic != SP_GGUF_MAGIC) { sp_pv_close(pv); return NULL; }
    uint32_t version = rd_u32(&c);
    TRC("version=%u", version);
    if (version < 2 || version > 3) { sp_pv_close(pv); return NULL; }
    uint64_t n_tensors = rd_u64(&c);
    uint64_t n_kv      = rd_u64(&c);
    TRC("n_tensors=%llu  n_kv=%llu", (unsigned long long)n_tensors, (unsigned long long)n_kv);
    if (c.err || n_tensors == 0 || n_tensors > (1u << 20)) {
        sp_pv_close(pv); return NULL;
    }

    // Skip KV pairs.
    for (uint64_t i = 0; i < n_kv && !c.err; ++i) {
        rd_skip_string(&c);            // key
        uint32_t vt = rd_u32(&c);      // value type
        skip_gguf_value(&c, vt);       // value
    }
    TRC("KV skipped, cursor at offset %lld, err=%d",
        (long long)(c.p - (const uint8_t *)pv->base), c.err);
    if (c.err) { sp_pv_close(pv); return NULL; }

    // Allocate tensor array.
    pv->tensors   = (sp_pv_tensor_t *)calloc((size_t)n_tensors, sizeof(sp_pv_tensor_t));
    pv->n_tensors = (uint32_t)n_tensors;
    if (!pv->tensors) { sp_pv_close(pv); return NULL; }

    TRC("parsing %llu tensor infos starting at offset %lld",
        (unsigned long long)n_tensors,
        (long long)(c.p - (const uint8_t *)pv->base));
    // Parse tensor infos (relative offsets recorded here; absolute fixed
    // up after we align to the data section below).
    for (uint64_t i = 0; i < n_tensors && !c.err; ++i) {
        sp_pv_tensor_t *t = &pv->tensors[i];
        long long pre = (long long)(c.p - (const uint8_t *)pv->base);
        rd_copy_string(&c, t->name, sizeof(t->name));
        if (i < 3 || (i % 100) == 0) {
            TRC("  tensor[%llu] @%lld name='%.40s' err=%d",
                (unsigned long long)i, pre, t->name, c.err);
        }
        t->n_dims = rd_u32(&c);
        if (t->n_dims > 4) {
            TRC("  tensor[%llu] BAD n_dims=%u name='%s'",
                (unsigned long long)i, t->n_dims, t->name);
            c.err = 1; break;
        }
        for (uint32_t d = 0; d < t->n_dims; ++d) t->ne[d] = rd_u64(&c);
        for (uint32_t d = t->n_dims; d < 4; ++d) t->ne[d] = 1;
        t->dtype  = rd_u32(&c);
        t->offset = rd_u64(&c);          // relative to data section start
        t->n_bytes = tensor_n_bytes(t->dtype, t->ne, t->n_dims);
    }
    TRC("tensor info parse done. err=%d  cursor=%lld",
        c.err, (long long)(c.p - (const uint8_t *)pv->base));
    if (c.err) { sp_pv_close(pv); return NULL; }

    // Data section starts after alignment padding.
    size_t hdr_end = (size_t)(c.p - (const uint8_t *)pv->base);
    size_t pad = (SP_GGUF_DATA_ALIGN - (hdr_end % SP_GGUF_DATA_ALIGN))
                  % SP_GGUF_DATA_ALIGN;
    pv->data_section = (uint8_t *)pv->base + hdr_end + pad;
    TRC("data_section @%zu (hdr_end=%zu pad=%zu)", hdr_end + pad, hdr_end, pad);

    // Resolve absolute pointers.
    for (uint32_t i = 0; i < pv->n_tensors; ++i) {
        pv->tensors[i].ptr = (uint8_t *)pv->data_section + pv->tensors[i].offset;
    }
    TRC("pointers resolved; sample ptr[0]=%p offset=%llu",
        pv->tensors[0].ptr, (unsigned long long)pv->tensors[0].offset);

    // Build hash table — next pow2 >= 2 * n_tensors, min 16.
    uint32_t nb = 16;
    while (nb < (uint32_t)(n_tensors * 2)) nb <<= 1;
    TRC("hash table nb=%u", nb);
    pv->n_buckets = nb;
    pv->mask      = nb - 1;
    pv->buckets   = (sp_pv_hbucket_t *)calloc(nb, sizeof(sp_pv_hbucket_t));
    if (!pv->buckets) { TRC("buckets calloc FAILED"); sp_pv_close(pv); return NULL; }
    for (uint32_t i = 0; i < nb; ++i) pv->buckets[i].idx = -1;

    for (uint32_t i = 0; i < pv->n_tensors; ++i) {
        uint64_t h = fnv1a64(pv->tensors[i].name);
        uint32_t b = (uint32_t)(h & pv->mask);
        while (pv->buckets[b].idx >= 0) b = (b + 1) & pv->mask;
        pv->buckets[b].hash = h;
        pv->buckets[b].idx  = (int32_t)i;
    }
    TRC("hash table built");

    // Stash path.
    size_t plen = strlen(path);
    pv->path = (char *)malloc(plen + 1);
    if (pv->path) memcpy(pv->path, path, plen + 1);
    TRC("open complete, returning pv=%p", pv);

    return pv;
}

void sp_pv_close(sp_pointer_view_t *pv) {
    if (!pv) return;
    mmap_close(pv);
    free(pv->tensors);
    free(pv->buckets);
    free(pv->path);
    free(pv);
}

const sp_pv_tensor_t *sp_pv_get(const sp_pointer_view_t *pv, const char *name) {
    if (!pv || !name) return NULL;
    uint64_t h = fnv1a64(name);
    uint32_t b = (uint32_t)(h & pv->mask);
    for (uint32_t probe = 0; probe < pv->n_buckets; ++probe) {
        const sp_pv_hbucket_t *bk = &pv->buckets[b];
        if (bk->idx < 0) return NULL;
        if (bk->hash == h &&
            strcmp(pv->tensors[bk->idx].name, name) == 0) {
            return &pv->tensors[bk->idx];
        }
        b = (b + 1) & pv->mask;
    }
    return NULL;
}

int sp_pv_iterate(const sp_pointer_view_t *pv,
                   int (*cb)(const sp_pv_tensor_t *t, void *user),
                   void *user) {
    if (!pv || !cb) return 0;
    int n = 0;
    for (uint32_t i = 0; i < pv->n_tensors; ++i) {
        ++n;
        if (cb(&pv->tensors[i], user)) break;
    }
    return n;
}

void sp_pv_advise_range(const sp_pointer_view_t *pv,
                         void *base, size_t len, sp_pv_hint_t hint) {
    if (!pv || !base || len == 0) return;
#if defined(_WIN32)
    if (hint == SP_PV_HINT_WILLNEED) {
        WIN32_MEMORY_RANGE_ENTRY entry = { base, len };
        // Best-effort: function exists on Win8+; signature requires the
        // process handle and a count.
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
    }
    // DONTNEED / RANDOM / SEQUENTIAL: Windows file-backed mappings don't
    // expose a direct equivalent. Future work: VirtualUnlock on the range
    // for soft eviction; for now a no-op.
#else
    int adv = 0;
    switch (hint) {
        case SP_PV_HINT_WILLNEED:   adv = MADV_WILLNEED;   break;
        case SP_PV_HINT_DONTNEED:   adv = MADV_DONTNEED;   break;
        case SP_PV_HINT_RANDOM:     adv = MADV_RANDOM;     break;
        case SP_PV_HINT_SEQUENTIAL: adv = MADV_SEQUENTIAL; break;
        default: return;
    }
    (void)madvise(base, len, adv);
#endif
}

void sp_pv_advise_tensor(const sp_pointer_view_t *pv,
                          const char *name, sp_pv_hint_t hint) {
    const sp_pv_tensor_t *t = sp_pv_get(pv, name);
    if (!t) return;
    sp_pv_advise_range(pv, t->ptr, (size_t)t->n_bytes, hint);
}

size_t   sp_pv_file_size(const sp_pointer_view_t *pv)     { return pv ? pv->file_size : 0; }
uint32_t sp_pv_tensor_count(const sp_pointer_view_t *pv)  { return pv ? pv->n_tensors : 0; }
void    *sp_pv_data_section(const sp_pointer_view_t *pv)  { return pv ? pv->data_section : NULL; }
const char *sp_pv_path(const sp_pointer_view_t *pv)       { return pv ? pv->path : NULL; }
