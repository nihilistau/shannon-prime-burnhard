/*
 * burnhard_reservoir.c — Tiered mmap reservoir (Optane → NVMe → RAM)
 *
 * Fallback logic:
 *   1. If cfg says has_optane and optane_dev_path is set → DAX mmap
 *   2. Else if nvme_cache_path is set → file-backed mmap
 *   3. Else → aligned malloc (system RAM)
 *
 * On mobile (LONE_WOLF), we always land on RAM tier since there's
 * no Optane and we don't want to thrash flash storage with mmap.
 */

#include "burnhard_reservoir.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* ── Tier 0: Optane DAX ──────────────────────────────────────────── */

static int try_optane(reservoir_t *res, const burnhard_config_t *cfg, size_t size) {
    if (!cfg->caps.has_optane || !cfg->optane_dev_path)
        return -1;

#ifdef _WIN32
    /*
     * Windows: Optane appears as a DAX volume. Use CreateFileMapping
     * on the DAX device path. This is a simplified version — real
     * deployment uses the Windows SCM (Storage Class Memory) APIs.
     */
    HANDLE hFile = CreateFileA(
        cfg->optane_dev_path,
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL
    );
    if (hFile == INVALID_HANDLE_VALUE) return -1;

    HANDLE hMap = CreateFileMappingA(
        hFile, NULL, PAGE_READWRITE, (DWORD)(size >> 32), (DWORD)size, NULL
    );
    if (!hMap) { CloseHandle(hFile); return -1; }

    void *ptr = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, size);
    CloseHandle(hMap);
    if (!ptr) { CloseHandle(hFile); return -1; }

    res->base = ptr;
    res->size = size;
    res->tier = RESERVOIR_TIER_OPTANE;
    res->fd   = (int)(intptr_t)hFile; /* stash handle */
    return 0;
#else
    int fd = open(cfg->optane_dev_path, O_RDWR);
    if (fd < 0) return -1;

    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, fd, 0);
    if (ptr == MAP_FAILED) { close(fd); return -1; }

    res->base = ptr;
    res->size = size;
    res->tier = RESERVOIR_TIER_OPTANE;
    res->fd   = fd;
    return 0;
#endif
}


/* ── Tier 1: NVMe file-backed mmap ───────────────────────────────── */

static int try_nvme(reservoir_t *res, const burnhard_config_t *cfg, size_t size) {
    if (!cfg->nvme_cache_path)
        return -1;

#ifdef _WIN32
    HANDLE hFile = CreateFileA(
        cfg->nvme_cache_path,
        GENERIC_READ | GENERIC_WRITE,
        0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL
    );
    if (hFile == INVALID_HANDLE_VALUE) return -1;

    /* Extend file to requested size */
    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG)size;
    SetFilePointerEx(hFile, li, NULL, FILE_BEGIN);
    SetEndOfFile(hFile);

    HANDLE hMap = CreateFileMappingA(
        hFile, NULL, PAGE_READWRITE, (DWORD)(size >> 32), (DWORD)size, NULL
    );
    if (!hMap) { CloseHandle(hFile); return -1; }

    void *ptr = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, size);
    CloseHandle(hMap);
    if (!ptr) { CloseHandle(hFile); return -1; }

    res->base = ptr;
    res->size = size;
    res->tier = RESERVOIR_TIER_NVME;
    res->fd   = (int)(intptr_t)hFile;
    return 0;
#else
    int fd = open(cfg->nvme_cache_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return -1;

    if (ftruncate(fd, (off_t)size) != 0) { close(fd); return -1; }

    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { close(fd); return -1; }

    res->base = ptr;
    res->size = size;
    res->tier = RESERVOIR_TIER_NVME;
    res->fd   = fd;
    return 0;
#endif
}


/* ── Tier 2: System RAM (aligned malloc) ──────────────────────────── */

static int try_ram(reservoir_t *res, size_t size) {
#ifdef _WIN32
    void *ptr = _aligned_malloc(size, 64);
#else
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) ptr = NULL;
#endif
    if (!ptr) return -1;

    memset(ptr, 0, size);

    res->base = ptr;
    res->size = size;
    res->tier = RESERVOIR_TIER_RAM;
    res->fd   = -1;
    return 0;
}


/* ── Public API ───────────────────────────────────────────────────── */

int reservoir_open(reservoir_t *res, const burnhard_config_t *cfg, size_t size_bytes) {
    if (!res || size_bytes == 0) return -1;

    memset(res, 0, sizeof(*res));
    res->fd = -1;

    /* LONE_WOLF: skip Optane and NVMe, go straight to RAM */
    if (cfg->mode == BURNHARD_MODE_LONE_WOLF) {
        int rc = try_ram(res, size_bytes);
        if (rc == 0) {
            fprintf(stderr, "[reservoir] Opened %zu MB on %s\n",
                    size_bytes / (1024*1024), reservoir_tier_str(res->tier));
        }
        return rc;
    }

    /* Try tiers in order */
    if (try_optane(res, cfg, size_bytes) == 0) goto done;

    fprintf(stderr, "[reservoir] Optane unavailable, trying NVMe fallback...\n");
    if (try_nvme(res, cfg, size_bytes) == 0) goto done;

    fprintf(stderr, "[reservoir] NVMe unavailable, falling back to system RAM...\n");
    if (try_ram(res, size_bytes) == 0) goto done;

    fprintf(stderr, "[reservoir] FATAL: cannot allocate %zu MB on any tier\n",
            size_bytes / (1024*1024));
    return -1;

done:
    fprintf(stderr, "[reservoir] Opened %zu MB on %s\n",
            size_bytes / (1024*1024), reservoir_tier_str(res->tier));
    return 0;
}

void reservoir_close(reservoir_t *res) {
    if (!res || !res->base) return;

    switch (res->tier) {
    case RESERVOIR_TIER_OPTANE:
    case RESERVOIR_TIER_NVME:
#ifdef _WIN32
        UnmapViewOfFile(res->base);
        if (res->fd != -1)
            CloseHandle((HANDLE)(intptr_t)res->fd);
#else
        munmap(res->base, res->size);
        if (res->fd >= 0) close(res->fd);
#endif
        break;
    case RESERVOIR_TIER_RAM:
#ifdef _WIN32
        _aligned_free(res->base);
#else
        free(res->base);
#endif
        break;
    }

    memset(res, 0, sizeof(*res));
    res->fd = -1;
}

const char *reservoir_tier_str(reservoir_tier_t tier) {
    switch (tier) {
        case RESERVOIR_TIER_OPTANE: return "OPTANE";
        case RESERVOIR_TIER_NVME:   return "NVMe";
        case RESERVOIR_TIER_RAM:    return "RAM";
        default:                    return "UNKNOWN";
    }
}

void reservoir_prefetch(const reservoir_t *res, size_t offset, size_t len) {
    if (!res || !res->base || res->tier == RESERVOIR_TIER_RAM) return;

    /* Issue cacheline-level prefetch hints */
    const char *p = (const char *)res->base + offset;
    const char *end = p + len;
    for (; p < end; p += 64) {
#ifdef _MSC_VER
        _mm_prefetch(p, _MM_HINT_T0);
#elif defined(__GNUC__)
        __builtin_prefetch(p, 0, 3);
#endif
    }
}
