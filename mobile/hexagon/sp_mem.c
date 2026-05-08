/*
 * sp_mem.c — DMA-BUF / ION memory management for Shannon-Prime Phase 9.
 *
 * Implements the ARM-side of the DMA-BUF bridge:
 *   - Allocate via /dev/dma_heap/system (Android 12+) or /dev/ion
 *   - mmap into the caller's address space
 *   - Register with QNN as a MemHandle for zero-copy HTP DMA
 *   - Destroy in the mandatory order (QnnMem_unregister → munmap → close)
 *
 * Root cause of the 2 GB ION "vanish" (Phase 7 profiling finding):
 *   If close(fd) is called before QnnMem_unregister, the QNN HTP driver
 *   holds a kernel DMA-BUF reference to the pages. The kernel cannot
 *   reclaim the pages because a "ghost" refcount keeps them pinned until
 *   the process exits. sp_mem_destroy() enforces the correct order.
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
 */

#include "sp_mem.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ── Linux DMA-heap ioctl interface ────────────────────────────────────── */
/* Matches <linux/dma-heap.h> available in Android kernel 5.10+.
 * We define the structs inline to avoid kernel header dependency. */
#define DMA_HEAP_IOCTL_ALLOC   _IOWR('H', 0x0, struct dma_heap_allocation_data)

struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;    /* O_RDWR | O_CLOEXEC */
    uint64_t heap_flags;  /* must be 0 for system heap */
};

/* ── ION legacy fallback (Android <= 11) ──────────────────────────────── */
#define ION_IOC_MAGIC 'I'
#define ION_IOC_ALLOC _IOWR(ION_IOC_MAGIC, 0, struct ion_allocation_data)
#define ION_IOC_MAP   _IOWR(ION_IOC_MAGIC, 2, struct ion_fd_data)

struct ion_allocation_data {
    size_t  len;
    size_t  align;
    unsigned int heap_id_mask;
    unsigned int flags;
    int     handle;
};
struct ion_fd_data {
    int handle;
    int fd;
};

/* ION_HEAP_SYSTEM_MASK = 1 << 0 */
#define ION_HEAP_SYSTEM_MASK (1u << 0)

/* ── Helpers ────────────────────────────────────────────────────────────── */

/* Round up to 4096-byte page boundary. */
static size_t page_align(size_t n) {
    return (n + 4095u) & ~(size_t)4095u;
}

/* ── sp_mem_alloc ────────────────────────────────────────────────────────── */

int sp_mem_alloc(SP_Memory* mem, size_t bytes) {
    if (!mem || bytes == 0) return -1;
    memset(mem, 0, sizeof(*mem));
    mem->fd  = -1;
    mem->ptr = MAP_FAILED;

    size_t aligned = page_align(bytes);

    /* ── Try /dev/dma_heap/system first (Android 12+, preferred) ─────── */
    int heap_fd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    if (heap_fd >= 0) {
        struct dma_heap_allocation_data alloc;
        memset(&alloc, 0, sizeof(alloc));
        alloc.len      = aligned;
        alloc.fd_flags = O_RDWR | O_CLOEXEC;

        if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) == 0) {
            mem->fd = (int)alloc.fd;
        } else {
            fprintf(stderr, "[sp_mem] dma_heap IOCTL_ALLOC failed: %s\n",
                    strerror(errno));
        }
        close(heap_fd);
    }

    /* ── Fallback: /dev/ion (Android <= 11) ──────────────────────────── */
    if (mem->fd < 0) {
        int ion_fd = open("/dev/ion", O_RDONLY | O_CLOEXEC);
        if (ion_fd < 0) {
            fprintf(stderr, "[sp_mem] neither /dev/dma_heap/system nor /dev/ion "
                            "is accessible — check sp_mem_setup_permissions()\n");
            return -1;
        }
        struct ion_allocation_data ion_alloc;
        memset(&ion_alloc, 0, sizeof(ion_alloc));
        ion_alloc.len          = aligned;
        ion_alloc.align        = 4096;
        ion_alloc.heap_id_mask = ION_HEAP_SYSTEM_MASK;
        ion_alloc.flags        = 0;

        if (ioctl(ion_fd, ION_IOC_ALLOC, &ion_alloc) != 0) {
            fprintf(stderr, "[sp_mem] ION_IOC_ALLOC failed: %s\n", strerror(errno));
            close(ion_fd);
            return -1;
        }
        struct ion_fd_data fd_data;
        memset(&fd_data, 0, sizeof(fd_data));
        fd_data.handle = ion_alloc.handle;
        if (ioctl(ion_fd, ION_IOC_MAP, &fd_data) != 0) {
            fprintf(stderr, "[sp_mem] ION_IOC_MAP failed: %s\n", strerror(errno));
            close(ion_fd);
            return -1;
        }
        mem->fd = fd_data.fd;
        close(ion_fd);
    }

    /* ── mmap the DMA-BUF FD into process address space ─────────────── */
    mem->ptr = mmap(NULL, aligned,
                    PROT_READ | PROT_WRITE, MAP_SHARED,
                    mem->fd, 0);
    if (mem->ptr == MAP_FAILED) {
        fprintf(stderr, "[sp_mem] mmap failed for fd=%d size=%zu: %s\n",
                mem->fd, aligned, strerror(errno));
        close(mem->fd);
        mem->fd = -1;
        return -1;
    }

    mem->size = aligned;
    fprintf(stderr, "[sp_mem] alloc: ptr=%p fd=%d size=%zu bytes\n",
            mem->ptr, mem->fd, mem->size);
    return 0;
}

/* ── sp_mem_register_qnn ─────────────────────────────────────────────────── */

int sp_mem_register_qnn(SP_Memory* mem,
                        void* qnn_context,
                        sp_mem_qnn_register_fn register_fn,
                        uint32_t n_dims,
                        const uint32_t* dims,
                        uint32_t qnn_dtype) {
    if (!mem || !register_fn || mem->fd < 0 || !qnn_context) return -1;
    if (mem->qnn_h) {
        fprintf(stderr, "[sp_mem] already registered — unregister first\n");
        return -1;
    }

    /*
     * QnnMemDescriptor_t layout (version-independent synopsis):
     *   { type=ION, fd, size, offset, dims, rank, dtype }
     * We pass a minimal descriptor through the opaque fn pointer.
     * Caller is responsible for supplying the right struct layout for
     * their QNN SDK version; this function just calls the fn pointer.
     */
    (void)n_dims; (void)dims; (void)qnn_dtype;
    /* Placeholder: full Qnn_MemDescriptor_t construction belongs in the
     * QNN-SDK-aware layer (qnn_bin_driver.cpp or shannon_prime_hexagon.c).
     * This function provides the teardown-safe wrapper pattern. */
    int rc = register_fn(qnn_context, NULL, 1, &mem->qnn_h);
    if (rc != 0) {
        fprintf(stderr, "[sp_mem] QnnMem_register failed rc=%d\n", rc);
        mem->qnn_h = NULL;
        return -1;
    }
    fprintf(stderr, "[sp_mem] registered fd=%d with QNN handle=%p\n",
            mem->fd, (void*)mem->qnn_h);
    return 0;
}

/* ── sp_mem_destroy ──────────────────────────────────────────────────────── */
/*
 * MANDATORY teardown order to prevent ION FD leaks:
 *
 *   Step 1: QnnMem_unregister  — release HTP backend's reference to the pages
 *   Step 2: munmap             — release ARM virtual address mapping
 *   Step 3: close(fd)          — release kernel's DMA-BUF handle
 *
 * If you close(fd) before QnnMem_unregister, the kernel cannot reclaim the
 * physical pages because the HTP driver's MemHandle still pins them. This
 * is the "2 GB vanish" pattern: free RAM drops and doesn't recover until
 * the process exits. Check with:
 *   adb shell cat /sys/kernel/debug/dma_buf/bufinfo
 */
void sp_mem_destroy(SP_Memory* mem, sp_mem_qnn_unregister_fn unregister_fn) {
    if (!mem) return;

    /* Step 1: UNREGISTER FROM QNN FIRST */
    if (mem->qnn_h != NULL) {
        if (unregister_fn) {
            int err = unregister_fn(&mem->qnn_h, 1);
            if (err != 0) {
                /* Log but continue — skipping munmap/close would leak worse. */
                fprintf(stderr, "[sp_mem] Warning: QnnMem_unregister failed "
                                "(rc=%d) — memory leak likely, check bufinfo\n", err);
            }
        } else {
            fprintf(stderr, "[sp_mem] Warning: qnn_h=%p is non-NULL but no "
                            "unregister_fn supplied — HTP reference not released\n",
                    (void*)mem->qnn_h);
        }
        mem->qnn_h = NULL;
    }

    /* Step 2: UNMAP HOST POINTER */
    if (mem->ptr != MAP_FAILED && mem->ptr != NULL) {
        if (munmap(mem->ptr, mem->size) != 0) {
            fprintf(stderr, "[sp_mem] munmap failed: %s\n", strerror(errno));
        }
        mem->ptr = MAP_FAILED;
    }

    /* Step 3: CLOSE DMA-BUF FD */
    if (mem->fd >= 0) {
        close(mem->fd);
        mem->fd = -1;
    }

    fprintf(stderr, "[sp_mem] reclaimed %zu bytes (teardown complete)\n", mem->size);
    mem->size = 0;
}

/* ── sp_mem_dump_bufinfo ─────────────────────────────────────────────────── */

void sp_mem_dump_bufinfo(void) {
    const char* path = "/sys/kernel/debug/dma_buf/bufinfo";
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[sp_mem] bufinfo: cannot open %s "
                        "(need root, or debugfs not mounted): %s\n",
                path, strerror(errno));
        return;
    }
    char buf[4096];
    ssize_t n;
    fprintf(stderr, "[sp_mem] === /sys/kernel/debug/dma_buf/bufinfo ===\n");
    while ((n = read(fd, buf, sizeof(buf) - 1)) > 0) {
        buf[n] = '\0';
        fprintf(stderr, "%s", buf);
    }
    fprintf(stderr, "[sp_mem] === end bufinfo ===\n");
    close(fd);
}

/* ── sp_mem_setup_permissions ────────────────────────────────────────────── */

int sp_mem_setup_permissions(void) {
    /* chmod 666 /dev/dma_heap/system */
    if (chmod("/dev/dma_heap/system", 0666) != 0) {
        fprintf(stderr, "[sp_mem] chmod /dev/dma_heap/system: %s "
                        "(need root or system_server context)\n", strerror(errno));
    }
    /* chmod 666 /dev/ion if present */
    if (access("/dev/ion", F_OK) == 0) {
        if (chmod("/dev/ion", 0666) != 0) {
            fprintf(stderr, "[sp_mem] chmod /dev/ion: %s\n", strerror(errno));
        }
    }
    /* Enable FastRPC unauthenticated access (non-root workaround) */
    system("setprop grp.adsprpc.unauth_enable 1");

    /* Verify /dev/dma_heap/system is now accessible. */
    int fd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "[sp_mem] /dev/dma_heap/system still not accessible "
                        "after permission setup — DMA alloc will fail\n");
        return -1;
    }
    close(fd);
    fprintf(stderr, "[sp_mem] permissions OK: /dev/dma_heap/system accessible\n");
    return 0;
}
