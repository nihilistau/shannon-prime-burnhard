// Shannon-Prime BurnHard — Reservoir probe + bind implementation.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Mostly a thin shim over platform mmap APIs.  The probe/auto-init
// reads system topology once and builds a burnhard_reservoir_t that
// the engine then queries — no per-token system calls.

#include "burnhard_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <sys/statvfs.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

// ----------------------------------------------------------------------------
// Tier name helpers — used by log strings + the Beast Canyon banner.
// ----------------------------------------------------------------------------
const char *burnhard_tier_name(burnhard_tier_t t) {
    switch (t) {
        case BURNHARD_TIER_OPTANE: return "OPTANE";
        case BURNHARD_TIER_NVME:   return "NVME";
        case BURNHARD_TIER_RAM:    return "RAM";
        default:                   return "NONE";
    }
}

const char *burnhard_mode_name(int mode) {
    switch (mode) {
        case 0: return "DUAL_OPTANE";
        case 1: return "OPTANE_NVME";
        case 2: return "OPTANE_ONLY";
        case 3: return "NVME_FALLBACK";
        case 4: return "RAM_FALLBACK";
        default: return "UNKNOWN";
    }
}

// ----------------------------------------------------------------------------
// Probe — figure out what's on this box.  Best-effort; never returns
// failure (probe always succeeds even if everything is missing).
// ----------------------------------------------------------------------------
int burnhard_probe(burnhard_reservoir_t *r) {
    if (!r) return -EINVAL;
    memset(r, 0, sizeof(*r));

#ifdef _WIN32
    // Walk drive letters A: through Z:.  For each fixed disk volume,
    // check the label — Optane sticks show up as "Optane" by convention
    // (or whatever the user labeled them).  We also tag MEMPEK* /
    // SSDPE21K* volumes via the volume name when GetVolumeInformation
    // can read it; otherwise we record capacity only and leave the
    // tier guess to the caller.
    DWORD drives = GetLogicalDrives();
    for (int i = 0; i < 26; ++i) {
        if (!(drives & (1u << i))) continue;
        char root[4] = { (char)('A' + i), ':', '\\', 0 };
        if (GetDriveTypeA(root) != DRIVE_FIXED) continue;
        ULARGE_INTEGER avail = {0}, total = {0}, free_bytes = {0};
        if (!GetDiskFreeSpaceExA(root, &avail, &total, &free_bytes)) continue;

        char vol_name[MAX_PATH] = {0};
        char fs_name[MAX_PATH]  = {0};
        DWORD serial = 0, max_comp = 0, fs_flags = 0;
        GetVolumeInformationA(root, vol_name, MAX_PATH, &serial, &max_comp,
                              &fs_flags, fs_name, MAX_PATH);

        // Heuristic: total ≤ 32 GB AND name contains "Optane" / "MEMPEK"
        // → counts as Optane for our purposes.  The label is what the
        // user controls; the size cap rules out a full SSD relabeled.
        const int looks_optane =
            (total.QuadPart <= (ULONGLONG)32 * 1024 * 1024 * 1024) &&
            ((vol_name[0] && (strstr(vol_name, "Optane") ||
                              strstr(vol_name, "OPTANE") ||
                              strstr(vol_name, "MEMPEK"))));

        if (looks_optane) {
            r->has_optane_total_count++;
            r->optane_total_bytes += (size_t)total.QuadPart;
        } else {
            // Treat large fixed drives as candidate NVMe scratch; sum
            // free bytes (not capacity) since we only need spillover.
            r->nvme_free_bytes += (size_t)free_bytes.QuadPart;
        }
    }

    MEMORYSTATUSEX mem = { sizeof(mem) };
    if (GlobalMemoryStatusEx(&mem)) {
        r->ram_free_bytes = (size_t)mem.ullAvailPhys;
    }
#else
    // Linux: walk /sys/block for nvme* / pmem*; check size + model
    // strings.  We stay conservative — no /dev probing without root.
    FILE *f = popen("lsblk -bnPo NAME,MODEL,SIZE -d 2>/dev/null", "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            const int looks_optane =
                strstr(line, "MEMPEK")  || strstr(line, "Optane") ||
                strstr(line, "OPTANE")  || strstr(line, "SSDPE21K") ||
                strstr(line, "P4800")   || strstr(line, "P5800");
            // SIZE="<bytes>" — parse the numeric value.
            char *sz = strstr(line, "SIZE=\"");
            if (!sz) continue;
            sz += 6;
            unsigned long long bytes = strtoull(sz, NULL, 10);
            if (looks_optane) {
                r->has_optane_total_count++;
                r->optane_total_bytes += (size_t)bytes;
            }
        }
        pclose(f);
    }
    struct statvfs sv;
    if (statvfs("/var/tmp", &sv) == 0) {
        r->nvme_free_bytes = (size_t)sv.f_bsize * (size_t)sv.f_bavail;
    }
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long pgsz  = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && pgsz > 0) {
        r->ram_free_bytes = (size_t)pages * (size_t)pgsz;
    }
#endif

    return 0;
}

// ----------------------------------------------------------------------------
// Bind one chamber.  RAM tier just allocates; OPTANE/NVME tiers mmap.
// On Windows we do a CreateFile + CreateFileMapping + MapViewOfFile;
// on POSIX it's open + mmap.
// ----------------------------------------------------------------------------
int burnhard_chamber_bind(burnhard_chamber_t *c,
                          burnhard_tier_t tier,
                          const char *path,
                          uint32_t roles,
                          size_t requested_bytes) {
    if (!c) return -EINVAL;
    burnhard_chamber_release(c);

    c->tier  = tier;
    c->roles = roles;
    c->dax_enabled = 0;

    if (tier == BURNHARD_TIER_RAM) {
        c->mapping = calloc(1, requested_bytes);
        if (!c->mapping) return -ENOMEM;
        c->capacity_bytes = requested_bytes;
        c->path[0] = '\0';
        return 0;
    }

    if (!path || !path[0]) return -EINVAL;
    strncpy(c->path, path, sizeof(c->path) - 1);
    c->path[sizeof(c->path) - 1] = '\0';

#ifdef _WIN32
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) return -ENOENT;

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz)) { CloseHandle(fh); return -EIO; }
    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) { CloseHandle(fh); return -EIO; }

    void *base = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mh);
    CloseHandle(fh);
    if (!base) return -EIO;

    c->mapping        = base;
    c->capacity_bytes = (size_t)sz.QuadPart;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -errno;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -errno; }
    void *base = mmap(NULL, (size_t)st.st_size, PROT_READ,
                      MAP_PRIVATE | MAP_POPULATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return -errno;
    c->mapping        = base;
    c->capacity_bytes = (size_t)st.st_size;
    // DAX detection on Linux: stat the path's filesystem flags; for
    // brevity we leave dax_enabled=0 here and let the engine probe via
    // /proc/mounts when it really needs to know.
#endif
    (void)requested_bytes;  // capacity is whatever the file is
    return 0;
}

void burnhard_chamber_release(burnhard_chamber_t *c) {
    if (!c || !c->mapping) return;
    if (c->tier == BURNHARD_TIER_RAM) {
        free(c->mapping);
    } else {
#ifdef _WIN32
        UnmapViewOfFile(c->mapping);
#else
        munmap(c->mapping, c->capacity_bytes);
#endif
    }
    c->mapping = NULL;
    c->capacity_bytes = 0;
    c->used_bytes = 0;
    c->tier = BURNHARD_TIER_NONE;
    c->roles = BURNHARD_ROLE_NONE;
}

// ----------------------------------------------------------------------------
// Auto-init.  Strategy:
//   - If 2 Optane sticks present: primary=biggest, secondary=other.
//   - If 1 Optane stick: primary=Optane, secondary=NVMe scratch file
//                       (if there's room) or NONE.
//   - If 0 Optane sticks: primary=NVMe-mapped GGUF, secondary=NONE.
//   - If even NVMe path fails: RAM_FALLBACK (engine still works, no
//                               persistent reservoir).
// ----------------------------------------------------------------------------
int burnhard_reservoir_init_auto(burnhard_reservoir_t *r,
                                 const char *gguf_path) {
    if (!r) return -EINVAL;
    burnhard_probe(r);

    fprintf(stderr,
        "[burnhard] probe: optane_sticks=%d optane_total=%.1f GiB "
        "nvme_free=%.1f GiB ram_free=%.1f GiB\n",
        r->has_optane_total_count,
        (double)r->optane_total_bytes / (1024.0 * 1024.0 * 1024.0),
        (double)r->nvme_free_bytes    / (1024.0 * 1024.0 * 1024.0),
        (double)r->ram_free_bytes     / (1024.0 * 1024.0 * 1024.0));

    // Stage 1: bind primary.  If a GGUF path was supplied, we always
    // map *that file* into primary so the engine has weight pointers.
    int rc = -1;
    if (gguf_path && gguf_path[0]) {
        burnhard_tier_t want = (r->has_optane_total_count > 0)
                                 ? BURNHARD_TIER_OPTANE
                                 : BURNHARD_TIER_NVME;
        rc = burnhard_chamber_bind(&r->primary, want, gguf_path,
                                   BURNHARD_ROLE_EXPERT_RESV |
                                   BURNHARD_ROLE_RoPE_TABLES,
                                   0);
        if (rc != 0 && want == BURNHARD_TIER_OPTANE) {
            // Optane bind failed (path may be on a non-Optane volume);
            // try NVMe tier with the same path before giving up.
            rc = burnhard_chamber_bind(&r->primary, BURNHARD_TIER_NVME,
                                       gguf_path,
                                       BURNHARD_ROLE_EXPERT_RESV |
                                       BURNHARD_ROLE_RoPE_TABLES, 0);
        }
    }

    if (rc != 0) {
        // RAM fallback — small allocation, no mapped weights.
        r->mode = 4;  // RAM_FALLBACK
        fprintf(stderr,
            "[burnhard] no persistent reservoir available — RAM fallback "
            "(engine will run but expert prefetch is disabled)\n");
        return 0;
    }

    // Stage 2: secondary chamber for KV-overflow / draft model.  Only
    // bind if a second Optane stick is detected (we don't auto-create
    // NVMe scratch files here — caller's job).
    if (r->has_optane_total_count >= 2) {
        // Future: walk drive letters again and pick the second-largest
        // Optane volume, mmap a per-session scratch file from it.
        // For now we just leave secondary unbound and tag the mode.
        r->mode = 0;  // DUAL_OPTANE
        fprintf(stderr,
            "[burnhard] dual-chamber mode armed (secondary not yet bound — "
            "scratch file allocation pending)\n");
    } else if (r->primary.tier == BURNHARD_TIER_OPTANE) {
        r->mode = 2;  // OPTANE_ONLY
    } else {
        r->mode = 3;  // NVME_FALLBACK
    }

    fprintf(stderr,
        "[burnhard] reservoir online: mode=%s primary=%s/%.1f GiB"
        " (used=%.1f GiB)\n",
        burnhard_mode_name(r->mode),
        burnhard_tier_name(r->primary.tier),
        (double)r->primary.capacity_bytes / (1024.0 * 1024.0 * 1024.0),
        (double)r->primary.used_bytes     / (1024.0 * 1024.0 * 1024.0));
    return 0;
}

void burnhard_reservoir_free(burnhard_reservoir_t *r) {
    if (!r) return;
    burnhard_chamber_release(&r->primary);
    burnhard_chamber_release(&r->secondary);
    memset(r, 0, sizeof(*r));
}
