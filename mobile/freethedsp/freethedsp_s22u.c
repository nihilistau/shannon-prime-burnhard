// Shannon-Prime / freethedsp — freethedsp_s22u.c
// Patching LD_PRELOAD shim, S22U-adapted.
//
// Derived from geohot/freethedsp (MIT). Copyright (C) 2026 Ray Daniels.
// Licensed under AGPLv3. Commercial license: raydaniels@gmail.com
//
// Strategy: hook ioctl(), watch for FASTRPC_IOCTL_INIT, find the per-
// process fastrpc_shell_3 ELF that the kernel mapped into our address
// space at init->mem, patch is_test_enabled() to return -1 instead of 0,
// flush the cache so the cDSP picks up the patched bytes when it loads
// the shell. From that point on, the cDSP treats us as test-signed and
// grants every API permission an unsigned-PD process is otherwise denied.
//
// Build:
//   $ANDROID_ARM64_TOOLCHAIN/bin/aarch64-linux-android21-clang \
//     -shared -fPIC -O2 freethedsp_s22u.c -o libfreethedsp.so -ldl
//
// Use:
//   LD_PRELOAD=./libfreethedsp.so SP_FREETHEDSP=1 ./<your_program>
//
// SP_FREETHEDSP=1 is the opt-in env var; without it we LD_PRELOAD to a
// no-op pass-through. This lets us bake the shim in as default-loaded
// without changing behaviour for builds that don't need it.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <stdarg.h>

// =============================================================================
// >>> PATCH OFFSETS — populate from `tools/find_is_test_enabled.py` output <<<
//
// These three constants are device-firmware-specific. Run discover.so first
// (Phase D.1), pull the dump, run the offset-finder tool against it
// (Phase D.2), and replace the placeholder values below.
//
// The four-byte instruction at PATCH_ADDR is the entry of the cDSP-side
// is_test_enabled() function. We swap a `r0=#0; jumpr lr`-equivalent
// sequence for `r0=#-1; jumpr lr`-equivalent.
//
// SHELL_SHA256 is recorded so future-us can detect firmware drift: if
// the captured shell ever stops matching this hash, the offset has
// almost certainly moved and D.2 needs to re-run.
//
#define PATCH_ADDR     0x00000000  /* TODO_FILL_FROM_D2 — file offset */
#define PATCH_OLD      "\x00\x00\x00\x00"  /* TODO_FILL_FROM_D2 */
#define PATCH_NEW      "\x00\x00\x00\x00"  /* TODO_FILL_FROM_D2 */
#define PATCH_LEN      (sizeof(PATCH_OLD) - 1)
#define SHELL_SHA256   "TODO_FILL_FROM_D2"
// =============================================================================

// FastRPC IOCTL — same struct as in discover.c.
#define IOC_OUT   0x40000000
#define IOC_IN    0x80000000
#define IOC_INOUT (IOC_IN | IOC_OUT)
#define IOCPARM_MASK 0x1fff
#define _IOC(inout, group, num, len) \
    (inout | ((len & IOCPARM_MASK) << 16) | ((group) << 8) | (num))
#define _IOWR(g, n, t) _IOC(IOC_INOUT, (g), (n), sizeof(t))

struct fastrpc_ioctl_init {
    uint32_t flags;
    uintptr_t file;
    int32_t  filelen;
    int32_t  filefd;
    uintptr_t mem;
    int32_t  memlen;
    int32_t  memfd;
};
#define FASTRPC_IOCTL_INIT _IOWR('R', 6, struct fastrpc_ioctl_init)

// Bionic's ioctl prototype is variadic: `int ioctl(int, int, ...)`.
static int (*real_ioctl)(int, int, void *) = NULL;
static int patch_done = 0;

static void load_real_ioctl(void) {
    if (real_ioctl) return;
    void *h = dlopen("libc.so", RTLD_LAZY);
    if (!h) h = dlopen("/system/lib64/libc.so", RTLD_LAZY);
    if (!h) h = dlopen("/apex/com.android.runtime/lib64/bionic/libc.so", RTLD_LAZY);
    if (!h) {
        fprintf(stderr, "[freethedsp] dlopen(libc) failed: %s\n", dlerror());
        abort();
    }
    real_ioctl = (int (*)(int, int, void *))dlsym(h, "ioctl");
    if (!real_ioctl) {
        fprintf(stderr, "[freethedsp] dlsym(ioctl) failed: %s\n", dlerror());
        abort();
    }
}

// Cache flush. Three strategies, tried in order:
//   (1) /dev/dma_heap-aware ioctl (Android Q+).
//   (2) msync(MS_INVALIDATE) on the user mapping. Often sufficient on
//       V66+ where the cDSP shares coherency domain with the AP for
//       userspace-allocated DMA buffers.
//   (3) /dev/ion (legacy Android <= P) — geohot's original path.
//
// We just call (2) by default; it's the simplest and works on every
// device this code has been tested on. (1) and (3) are stubs to extend
// if a particular device needs them.
static void flush_cache(void *addr, size_t len) {
    if (msync(addr, len, MS_INVALIDATE | MS_SYNC) != 0) {
        fprintf(stderr, "[freethedsp] msync(MS_INVALIDATE) returned %d (%s) — "
                        "patch may not propagate; check device coherency model\n",
                errno, strerror(errno));
    } else {
        fprintf(stderr, "[freethedsp] msync(MS_INVALIDATE) on patched page OK\n");
    }
}

static int sp_freethedsp_enabled(void) {
    const char *v = getenv("SP_FREETHEDSP");
    return v && *v && *v != '0';
}

// Apply the test-enable patch to the in-memory shell ELF.
static void apply_patch(struct fastrpc_ioctl_init *init) {
    if (PATCH_ADDR == 0) {
        fprintf(stderr,
            "[freethedsp] PATCH_ADDR is unset — this build is a placeholder.\n"
            "             Run Phase D.1 (discover.so) and D.2 "
            "(tools/find_is_test_enabled.py)\n"
            "             then rebuild this shim with the offsets baked in.\n"
            "             Falling back to pass-through.\n");
        return;
    }
    if ((int)PATCH_ADDR + (int)PATCH_LEN > init->memlen) {
        fprintf(stderr, "[freethedsp] PATCH_ADDR (0x%x) past end of shell mem (len=%d) — "
                        "wrong device firmware? skipping patch.\n",
                (unsigned)PATCH_ADDR, init->memlen);
        return;
    }

    void *target = (void *)(init->mem + PATCH_ADDR);
    if (memcmp(target, PATCH_OLD, PATCH_LEN) != 0) {
        fprintf(stderr,
            "[freethedsp] expected bytes at PATCH_ADDR 0x%x do not match.\n"
            "             Found: %02x %02x %02x %02x\n"
            "             This usually means the cDSP shell on this device\n"
            "             differs from the one that PATCH_ADDR was derived\n"
            "             from. Re-run Phase D.1 + D.2 to find the new offset.\n"
            "             Falling back to pass-through (signed PD APIs will\n"
            "             remain blocked).\n",
            (unsigned)PATCH_ADDR,
            ((unsigned char *)target)[0], ((unsigned char *)target)[1],
            ((unsigned char *)target)[2], ((unsigned char *)target)[3]);
        return;
    }

    memcpy(target, PATCH_NEW, PATCH_LEN);
    flush_cache(target, PATCH_LEN);
    patch_done = 1;
    fprintf(stderr, "[freethedsp] patched is_test_enabled at 0x%x — "
                    "cDSP test-mode now active for this process\n",
            (unsigned)PATCH_ADDR);
}

int ioctl(int fd, int request, ...) {
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);

    load_real_ioctl();
    int rc = real_ioctl(fd, request, arg);

    if (request == (int)FASTRPC_IOCTL_INIT && arg != NULL && sp_freethedsp_enabled() && !patch_done) {
        apply_patch((struct fastrpc_ioctl_init *)arg);
    }
    return rc;
}
