// Shannon-Prime: Runtime mode dispatcher — implementation. See sp_runmode.h.

#include "sp_runmode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <io.h>
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
#  define close_socket closesocket
typedef int socklen_t;
#else
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <sys/stat.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <unistd.h>
#  include <errno.h>
#  define close_socket close
#endif

// ---------------------------------------------------------------------
// CUDA probe
// ---------------------------------------------------------------------
//
// Compile-time gated on SP_WITH_CUDA so this file stays buildable on
// CPU-only hosts. Even with the runtime present we soft-fail — no error
// path here, just true/false.
bool sp_probe_cuda(void) {
#ifdef SP_WITH_CUDA
    // Late-bind via dlopen to avoid hard-linking cudart on every binary.
#  ifdef _WIN32
    HMODULE h = LoadLibraryA("nvcuda.dll");
    if (!h) return false;
    FreeLibrary(h);
    return true;
#  else
    void *h = NULL;
    extern void *dlopen(const char *, int);
    extern int   dlclose(void *);
    h = dlopen("libcuda.so.1", 0x102 /* RTLD_NOW|RTLD_GLOBAL */);
    if (!h) return false;
    dlclose(h);
    return true;
#  endif
#else
    return false;
#endif
}

// ---------------------------------------------------------------------
// Intel iGPU probe (via Vulkan vendor scan; full L0 is checked elsewhere)
// ---------------------------------------------------------------------
bool sp_probe_l0_igpu(void) {
#ifdef _WIN32
    // Cheap heuristic on Windows: ze_loader.dll must be present for L0.
    HMODULE h = LoadLibraryA("ze_loader.dll");
    if (h) { FreeLibrary(h); return true; }
    // Fall through to checking for an Intel display adapter via registry.
    HKEY key;
    LONG rc = RegOpenKeyExA(HKEY_LOCAL_MACHINE,
        "SOFTWARE\\Intel\\GFX", 0, KEY_READ, &key);
    if (rc == ERROR_SUCCESS) { RegCloseKey(key); return true; }
    return false;
#else
    // Linux: /sys/class/drm/card*/device/vendor == "0x8086"
    for (int i = 0; i < 8; ++i) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/class/drm/card%d/device/vendor", i);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        char buf[16] = {0};
        size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
        if (n > 0 && strstr(buf, "0x8086")) return true;
    }
    return false;
#endif
}

// ---------------------------------------------------------------------
// Optane reservoir probe
// ---------------------------------------------------------------------
//
// Resolution order:
//   1. SP_OPTANE_PATH env var (explicit override)
//   2. Default device paths per platform
//   3. Look for a sentinel reservoir file at known location
//
// We don't *open* the file here — sp_optane_init does that. We just
// confirm the path exists so the dispatcher knows whether to attempt
// FULL_PULSE / DESKTOP_SOLO at all.
bool sp_probe_optane(char *out_path, size_t out_path_len) {
    if (out_path && out_path_len > 0) out_path[0] = '\0';

    const char *env = getenv("SP_OPTANE_PATH");
    if (env && env[0]) {
#ifdef _WIN32
        DWORD attr = GetFileAttributesA(env);
        if (attr != INVALID_FILE_ATTRIBUTES) {
            if (out_path) snprintf(out_path, out_path_len, "%s", env);
            return true;
        }
#else
        struct stat st;
        if (stat(env, &st) == 0) {
            if (out_path) snprintf(out_path, out_path_len, "%s", env);
            return true;
        }
#endif
    }

#ifdef _WIN32
    // Default Beast Canyon location: D:\sp_reservoir.bin or P:\sp_reservoir.bin
    static const char *win_defaults[] = {
        "D:\\sp_reservoir.bin",
        "P:\\sp_reservoir.bin",
        "C:\\Optane\\sp_reservoir.bin",
        NULL,
    };
    for (int i = 0; win_defaults[i]; ++i) {
        DWORD attr = GetFileAttributesA(win_defaults[i]);
        if (attr != INVALID_FILE_ATTRIBUTES) {
            if (out_path) snprintf(out_path, out_path_len, "%s",
                                   win_defaults[i]);
            return true;
        }
    }
#else
    // Linux: /dev/pmem0 (DAX) or /mnt/optane/sp_reservoir.bin
    static const char *lin_defaults[] = {
        "/dev/pmem0",
        "/mnt/optane/sp_reservoir.bin",
        "/var/lib/sp/sp_reservoir.bin",
        NULL,
    };
    for (int i = 0; lin_defaults[i]; ++i) {
        struct stat st;
        if (stat(lin_defaults[i], &st) == 0) {
            if (out_path) snprintf(out_path, out_path_len, "%s",
                                   lin_defaults[i]);
            return true;
        }
    }
#endif
    return false;
}

// ---------------------------------------------------------------------
// Sidecar (S22 phone) heartbeat probe — TCP dial to ADB-forwarded port
// ---------------------------------------------------------------------
//
// We just check whether the port accepts a connect within timeout_ms.
// The full heartbeat protocol lives in bridge/heartbeat/. This is the
// "is the phone there at boot?" question.
bool sp_probe_sidecar(int timeout_ms) {
    if (timeout_ms <= 0) timeout_ms = 500;

#ifdef _WIN32
    WSADATA wsa;
    static int wsa_inited = 0;
    if (!wsa_inited) {
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
        wsa_inited = 1;
    }
#endif

    int s = (int)socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s < 0) return false;

    // Non-blocking so we can enforce timeout.
#ifdef _WIN32
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
#else
    int flags = fcntl(s, F_GETFL, 0);
    fcntl(s, F_SETFL, flags | O_NONBLOCK);
#endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(9876);  // bridge port (ADB-forwarded)
    addr.sin_addr.s_addr = htonl(0x7F000001);  // 127.0.0.1

    int rc = connect(s, (struct sockaddr *)&addr, sizeof(addr));
    bool ok = false;

    if (rc == 0) {
        ok = true;  // immediate connect (rare for non-blocking)
    } else {
        fd_set wset;
        FD_ZERO(&wset);
        FD_SET((unsigned)s, &wset);
        struct timeval tv;
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        int sel = select(s + 1, NULL, &wset, NULL, &tv);
        if (sel > 0) {
            int err = 0;
            socklen_t errlen = sizeof(err);
            getsockopt(s, SOL_SOCKET, SO_ERROR, (char *)&err, &errlen);
            ok = (err == 0);
        }
    }

    close_socket(s);
    return ok;
}

// ---------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------

sp_hw_caps_t sp_probe_hardware(void) {
    sp_hw_caps_t c;
    memset(&c, 0, sizeof(c));

#ifdef SP_TARGET_MOBILE
    c.is_mobile_host = true;
    return c;  // Short-circuit: mobile builds don't probe desktop fabric.
#endif

    c.has_cuda      = sp_probe_cuda();
    c.has_l0_igpu   = sp_probe_l0_igpu();
    c.has_optane    = sp_probe_optane(c.optane_path, sizeof(c.optane_path));
    c.has_sidecar   = sp_probe_sidecar(500);
    return c;
}

sp_run_mode_t sp_select_mode(const sp_hw_caps_t *caps) {
    // Env override wins.
    const char *forced = getenv("SP_RUN_MODE");
    if (forced && forced[0]) {
        if (strcmp(forced, "FULL_PULSE")   == 0) return SP_MODE_FULL_PULSE;
        if (strcmp(forced, "DESKTOP_SOLO") == 0) return SP_MODE_DESKTOP_SOLO;
        if (strcmp(forced, "LEGACY")       == 0) return SP_MODE_LEGACY;
        if (strcmp(forced, "LONE_WOLF")    == 0) return SP_MODE_LONE_WOLF;
        fprintf(stderr,
                "[sp-mode] WARN: SP_RUN_MODE=%s unrecognized, auto-selecting\n",
                forced);
    }

    if (!caps) return SP_MODE_LEGACY;
    if (caps->is_mobile_host) return SP_MODE_LONE_WOLF;
    if (!caps->has_optane)    return SP_MODE_LEGACY;
    if (caps->has_sidecar)    return SP_MODE_FULL_PULSE;
    return SP_MODE_DESKTOP_SOLO;
}

const char *sp_mode_name(sp_run_mode_t m) {
    switch (m) {
        case SP_MODE_LONE_WOLF:    return "LONE_WOLF";
        case SP_MODE_LEGACY:       return "LEGACY";
        case SP_MODE_DESKTOP_SOLO: return "DESKTOP_SOLO";
        case SP_MODE_FULL_PULSE:   return "FULL_PULSE";
    }
    return "?";
}

void sp_log_mode(sp_run_mode_t m, const sp_hw_caps_t *caps) {
    fprintf(stderr,
            "[sp-mode] %s  cuda=%d l0_igpu=%d optane=%d sidecar=%d mobile=%d\n",
            sp_mode_name(m),
            caps ? caps->has_cuda : 0,
            caps ? caps->has_l0_igpu : 0,
            caps ? caps->has_optane : 0,
            caps ? caps->has_sidecar : 0,
            caps ? caps->is_mobile_host : 0);
    if (caps && caps->has_optane && caps->optane_path[0]) {
        fprintf(stderr, "[sp-mode]   optane path: %s\n", caps->optane_path);
    }
}
