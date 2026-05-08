/*
 * burnhard_config.c — Hardware detection & mode selection
 *
 * Probes the host for available silicon and selects the operating mode.
 * Platform-specific detection is gated by preprocessor; stubs return
 * false on unsupported platforms so the code compiles everywhere.
 */

#include "burnhard_config.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* ── Platform-specific probes ─────────────────────────────────────── */

#ifdef _WIN32
static uint64_t probe_system_ram_mb(void) {
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem))
        return (uint64_t)(mem.ullTotalPhys / (1024ULL * 1024ULL));
    return 0;
}
#else
static uint64_t probe_system_ram_mb(void) {
    /* Linux: read /proc/meminfo */
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemTotal: %llu kB", &kb) == 1) break;
    }
    fclose(f);
    return kb / 1024;
}
#endif

static bool probe_avx512(void) {
#if defined(__x86_64__) || defined(_M_X64)
    /* CPUID leaf 7, ECX bit 16 = AVX-512F */
    #ifdef _MSC_VER
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 16)) != 0;
    #elif defined(__GNUC__)
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0)
    );
    return (ebx & (1 << 16)) != 0;
    #endif
#endif
    return false;
}

static bool probe_optane(uint64_t *capacity_mb) {
    *capacity_mb = 0;
#ifdef _WIN32
    /*
     * Check for Intel Optane persistent memory via DAX volume.
     * Real detection would use NVML or Windows SCM APIs.
     * For now: check env override or known device path.
     */
    const char *dev = getenv("BURNHARD_OPTANE_DEV");
    if (dev && dev[0]) {
        /* Assume present; capacity from env or default 128 GB */
        const char *cap_str = getenv("BURNHARD_OPTANE_MB");
        *capacity_mb = cap_str ? (uint64_t)atoll(cap_str) : 131072;
        return true;
    }
#else
    /* Linux: check /dev/dax0.0 existence */
    FILE *f = fopen("/dev/dax0.0", "r");
    if (f) {
        fclose(f);
        *capacity_mb = 131072; /* placeholder */
        return true;
    }
#endif
    return false;
}

static bool probe_cuda(int *sm_version, uint64_t *vram_mb) {
    *sm_version = 0;
    *vram_mb = 0;
    /*
     * Lightweight check: try to load nvcuda.dll / libcuda.so.
     * Full integration uses the CUDA runtime; here we just detect presence.
     */
#ifdef _WIN32
    HMODULE h = LoadLibraryA("nvcuda.dll");
    if (h) {
        FreeLibrary(h);
        /* Default to RTX 2060 profile; real query via cuDeviceGetAttribute */
        *sm_version = 75;
        const char *vram_str = getenv("BURNHARD_VRAM_MB");
        *vram_mb = vram_str ? (uint64_t)atoll(vram_str) : 6144;
        return true;
    }
#endif
    return false;
}

static bool probe_l0(void) {
#ifdef _WIN32
    HMODULE h = LoadLibraryA("ze_loader.dll");
    if (h) { FreeLibrary(h); return true; }
#endif
    return false;
}

static bool probe_sidecar(uint16_t port) {
    /*
     * Check if ADB is forwarding our port.
     * We do a quick TCP connect to localhost:port.
     * If the S22 bridge daemon is listening, we get a connection.
     */
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2,2), &wsa) != 0) return false;

    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) { WSACleanup(); return false; }

    /* Non-blocking connect with 200ms timeout */
    u_long nonblock = 1;
    ioctlsocket(sock, FIONBIO, &nonblock);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    connect(sock, (struct sockaddr *)&addr, sizeof(addr));

    fd_set wset;
    FD_ZERO(&wset);
    FD_SET(sock, &wset);
    struct timeval tv = { .tv_sec = 0, .tv_usec = 200000 };
    int sel = select(0, NULL, &wset, NULL, &tv);

    closesocket(sock);
    WSACleanup();
    return sel > 0;
#else
    /* POSIX path — same logic */
    (void)port;
    return false; /* stub for now */
#endif
}


/* ── Mode selection logic ─────────────────────────────────────────── */

static burnhard_mode_t select_mode(const burnhard_capabilities_t *caps) {
    /*
     * Decision tree:
     *   1. If we're on mobile (no CUDA, no AVX-512, no Optane) → LONE_WOLF
     *   2. If sidecar is alive → FULL_PULSE
     *   3. If desktop but no sidecar → DESKTOP_SOLO
     *   4. If no Optane → demote to LEGACY variant of whichever above
     *
     * LEGACY can combine with DESKTOP_SOLO (no Optane + no phone).
     * We pick the most descriptive single mode.
     */

    bool is_desktop = caps->has_avx512 || caps->has_cuda || caps->has_l0;

    if (!is_desktop) {
        return BURNHARD_MODE_LONE_WOLF;
    }

    if (!caps->has_optane) {
        return BURNHARD_MODE_LEGACY;
    }

    if (caps->has_sidecar && caps->sidecar_responsive) {
        return BURNHARD_MODE_FULL_PULSE;
    }

    return BURNHARD_MODE_DESKTOP_SOLO;
}


/* ── Public API ───────────────────────────────────────────────────── */

int burnhard_detect(burnhard_config_t *cfg) {
    if (!cfg) return -1;

    /* Set defaults before probing */
    if (cfg->bridge_port == 0)           cfg->bridge_port = 8090;
    if (cfg->heartbeat_interval_ms == 0) cfg->heartbeat_interval_ms = 500;
    if (cfg->heartbeat_timeout_ms == 0)  cfg->heartbeat_timeout_ms = 2000;
    if (cfg->spec_draft_tokens == 0)     cfg->spec_draft_tokens = 8;

    burnhard_capabilities_t *c = &cfg->caps;

    /* Probe everything */
    c->system_ram_mb = probe_system_ram_mb();
    c->has_avx512    = probe_avx512();
    c->has_optane    = probe_optane(&c->optane_capacity_mb);
    c->has_cuda      = probe_cuda(&c->cuda_sm, &c->vram_mb);
    c->has_l0        = probe_l0();
    c->has_nvme      = true; /* assume NVMe present on any modern system */

    /* Sidecar: only probe if we're on desktop */
    if (c->has_avx512 || c->has_cuda) {
        c->has_sidecar       = probe_sidecar(cfg->bridge_port);
        c->sidecar_responsive = c->has_sidecar; /* first probe = responsive */
    } else {
        c->has_sidecar       = false;
        c->sidecar_responsive = false;
    }

    /* Select mode */
    cfg->mode = select_mode(c);

    /* Derive speculative routing */
    cfg->spec_use_sidecar = (cfg->mode == BURNHARD_MODE_FULL_PULSE);

    /* Reservoir sizing: Optane if available, else 1/4 system RAM */
    if (c->has_optane && cfg->reservoir_size_mb == 0) {
        cfg->reservoir_size_mb = c->optane_capacity_mb;
    } else if (cfg->reservoir_size_mb == 0) {
        cfg->reservoir_size_mb = c->system_ram_mb / 4;
    }

    return 0;
}

const char *burnhard_mode_str(burnhard_mode_t mode) {
    switch (mode) {
        case BURNHARD_MODE_FULL_PULSE:   return "FULL_PULSE";
        case BURNHARD_MODE_DESKTOP_SOLO: return "DESKTOP_SOLO";
        case BURNHARD_MODE_LEGACY:       return "LEGACY";
        case BURNHARD_MODE_LONE_WOLF:    return "LONE_WOLF";
        default:                         return "UNKNOWN";
    }
}

void burnhard_dump_caps(const burnhard_config_t *cfg) {
    const burnhard_capabilities_t *c = &cfg->caps;
    fprintf(stderr, "\n=== BurnHard Capability Snapshot ===\n");
    fprintf(stderr, "  Mode:        %s\n", burnhard_mode_str(cfg->mode));
    fprintf(stderr, "  RAM:         %llu MB\n", (unsigned long long)c->system_ram_mb);
    fprintf(stderr, "  AVX-512:     %s\n", c->has_avx512 ? "YES" : "no");
    fprintf(stderr, "  CUDA:        %s (SM %d, %llu MB VRAM)\n",
            c->has_cuda ? "YES" : "no", c->cuda_sm, (unsigned long long)c->vram_mb);
    fprintf(stderr, "  Level Zero:  %s\n", c->has_l0 ? "YES" : "no");
    fprintf(stderr, "  Optane:      %s (%llu MB)\n",
            c->has_optane ? "YES" : "no", (unsigned long long)c->optane_capacity_mb);
    fprintf(stderr, "  Sidecar:     %s (responsive: %s)\n",
            c->has_sidecar ? "CONNECTED" : "absent",
            c->sidecar_responsive ? "yes" : "no");
    fprintf(stderr, "  Reservoir:   %llu MB (%s)\n",
            (unsigned long long)cfg->reservoir_size_mb,
            c->has_optane ? "Optane" : "NVMe/RAM fallback");
    fprintf(stderr, "  Spec-decode: draft=%d via %s\n",
            cfg->spec_draft_tokens,
            cfg->spec_use_sidecar ? "SIDECAR" : "LOCAL");
    fprintf(stderr, "===================================\n\n");
}
