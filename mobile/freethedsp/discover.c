// Shannon-Prime / freethedsp — discover.c
// Phase D.1: dump-only LD_PRELOAD shim. Captures the in-memory
// fastrpc_shell_N ELF that the kernel maps into our process during
// FASTRPC_IOCTL_INIT, writes it to disk, and lets the original ioctl
// proceed unchanged. No patching at this stage — we just need the
// exact bytes for our S22U cDSP shell so we can locate is_test_enabled
// offline.
//
// Derived from geohot/freethedsp (MIT). Copyright (C) 2026 Ray Daniels.
// Licensed under AGPLv3. Commercial license: raydaniels@gmail.com
//
// Build (cross to aarch64 Android via NDK):
//   $ANDROID_ARM64_TOOLCHAIN/bin/aarch64-linux-android21-clang \
//     -shared -fPIC -O2 -Iinclude discover.c -o discover.so -ldl
//
// Use:
//   adb push discover.so /data/local/tmp/sp_qnn/
//   adb shell "cd /data/local/tmp/sp_qnn; \
//     LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
//     LD_PRELOAD=./discover.so SP_DUMP_PATH=./fastrpc_shell.bin \
//     ./<any FastRPC-using binary> ..."
//   adb pull /data/local/tmp/sp_qnn/fastrpc_shell.bin
//
// Then run tools/find_is_test_enabled.py on it offline.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <unistd.h>
#include <errno.h>
#include <stdarg.h>

// FastRPC IOCTL definitions.
//
// IMPORTANT: the kernel's FASTRPC driver registers IOCTLs with the
// Linux-style _IOWR macros from <asm-generic/ioctl.h>, NOT the BSD-style
// macros geohot's reference shim uses. We pull from the system header
// to ensure our constant matches what userland actually emits.
#include <sys/ioctl.h>

// The upstream Linux kernel + Android NDK header (<misc/fastrpc.h>)
// declares INIT_CREATE as _IOWR('R', 5, struct fastrpc_init_create) where
// the userspace process passes a `file` pointer to the shell ELF for the
// kernel to load into the new PD. Samsung/Qualcomm's fork on the S22U
// renumbered this to request 4 (we observed 0xc0185204 firing repeatedly
// during a real FastRPC session, with size field 0x18 = 24 = sizeof of
// fastrpc_init_create — the upstream INIT_CREATE struct). That's why
// geohot's reference doesn't match: he targeted older fastrpc with a
// different struct layout (40 bytes, including a separate `mem` field).
//
// On this device the modern struct is what we hook on:
struct fastrpc_init_create {
    uint32_t filelen;     // size of the shell ELF in bytes
    int32_t  filefd;      // fd of an ION/dma_heap buffer holding the ELF
    uint32_t attrs;       // FASTRPC_MODE_* bits (UNSIGNED_MODULE etc)
    uint32_t siglen;      // size of attached signature blob (0 for unsigned PD)
    uint64_t file;        // userspace pointer to the shell ELF — our patch target
};
// Samsung's renumbering: request 4 with size 24 carries fastrpc_init_create.
#define FASTRPC_IOCTL_INIT_CREATE_SAMSUNG  _IOWR('R', 4, struct fastrpc_init_create)
// Upstream-spec equivalent (in case we run on a non-Samsung Snapdragon):
#define FASTRPC_IOCTL_INIT_CREATE_UPSTREAM _IOWR('R', 5, struct fastrpc_init_create)

__attribute__((constructor))
static void discover_init(void) {
    fprintf(stderr, "[discover] LD_PRELOAD active (pid=%d). Hooking init-create on:\n"
                    "[discover]   samsung  = 0x%lx (request 4, size 24)\n"
                    "[discover]   upstream = 0x%lx (request 5, size 24)\n",
            getpid(),
            (unsigned long)FASTRPC_IOCTL_INIT_CREATE_SAMSUNG,
            (unsigned long)FASTRPC_IOCTL_INIT_CREATE_UPSTREAM);
}

// Bionic's ioctl prototype is variadic: `int ioctl(int, int, ...)`.
// We hook with the same signature and forward via dlsym'd real_ioctl.
static int (*real_ioctl)(int, int, void *) = NULL;

static void load_real_ioctl(void) {
    if (real_ioctl) return;
    void *h = dlopen("libc.so", RTLD_LAZY);
    if (!h) h = dlopen("/system/lib64/libc.so", RTLD_LAZY);
    if (!h) h = dlopen("/apex/com.android.runtime/lib64/bionic/libc.so", RTLD_LAZY);
    if (!h) {
        fprintf(stderr, "[discover] dlopen(libc) failed: %s\n", dlerror());
        abort();
    }
    real_ioctl = (int (*)(int, int, void *))dlsym(h, "ioctl");
    if (!real_ioctl) {
        fprintf(stderr, "[discover] dlsym(ioctl) failed: %s\n", dlerror());
        abort();
    }
}

static void dump_shell(const struct fastrpc_init_create *init) {
    const char *path = getenv("SP_DUMP_PATH");
    if (!path || !*path) path = "./fastrpc_shell.bin";

    if (init->file == 0 || init->filelen == 0) {
        fprintf(stderr, "[discover] init->file=0x%lx filelen=%u — nothing to dump\n",
                (unsigned long)init->file, init->filelen);
        return;
    }

    // Sanity-check: the shell should be an ELF. If it isn't, our struct
    // interpretation is wrong and we'd dump garbage — bail with a clear
    // message.
    const unsigned char *m = (const unsigned char *)(uintptr_t)init->file;
    if (m[0] != 0x7f || m[1] != 'E' || m[2] != 'L' || m[3] != 'F') {
        fprintf(stderr, "[discover] file=0x%lx doesn't look like ELF "
                        "(bytes %02x %02x %02x %02x); skipping dump.\n"
                        "[discover] Likely struct layout mismatch — adjust "
                        "fastrpc_init_create in discover.c.\n",
                (unsigned long)init->file, m[0], m[1], m[2], m[3]);
        return;
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[discover] fopen(%s) failed: %s\n", path, strerror(errno));
        return;
    }
    size_t wrote = fwrite(m, 1, (size_t)init->filelen, f);
    fclose(f);
    fprintf(stderr, "[discover] dumped %zu bytes (file=0x%lx filelen=%u attrs=0x%x siglen=%u filefd=%d) to %s\n",
            wrote, (unsigned long)init->file, init->filelen, init->attrs,
            init->siglen, init->filefd, path);

    fprintf(stderr, "[discover] first 16 bytes:");
    for (unsigned i = 0; i < 16 && i < init->filelen; i++) {
        fprintf(stderr, " %02x", m[i]);
    }
    fprintf(stderr, "\n");
}

// Match bionic's variadic prototype (system/bionic/libc/include/sys/ioctl.h).
int ioctl(int fd, int request, ...) {
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);

    load_real_ioctl();
    int rc = real_ioctl(fd, request, arg);

    // Trace mode: log every ioctl request so we can see what FastRPC
    // actually emits and confirm our hook is intercepting.
    if (getenv("SP_DUMP_TRACE")) {
        fprintf(stderr, "[discover] ioctl(fd=%d, req=0x%x) -> %d\n",
                fd, (unsigned)request, rc);
    }

    int is_init_create =
        (request == (int)FASTRPC_IOCTL_INIT_CREATE_SAMSUNG) ||
        (request == (int)FASTRPC_IOCTL_INIT_CREATE_UPSTREAM);
    if (is_init_create && arg != NULL) {
        dump_shell((const struct fastrpc_init_create *)arg);
        if (getenv("SP_DUMP_ONCE")) {
            fprintf(stderr, "[discover] SP_DUMP_ONCE set — exiting\n");
            _exit(0);
        }
    }
    return rc;
}
