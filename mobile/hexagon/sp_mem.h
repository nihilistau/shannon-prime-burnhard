/*
 * sp_mem.h — DMA-BUF / ION memory management for Shannon-Prime Phase 9.
 *
 * Manages contiguous memory objects that span the ARM↔DSP boundary.
 * Every allocation goes through /dev/dma_heap/system (Android 12+) with
 * an ION FD so the kernel can SMMU-map it into both address spaces without
 * a copy.
 *
 * CRITICAL teardown order (prevents the 2 GB ION "vanish"):
 *   1. QnnMem_unregister  — tell HTP backend to stop referencing the pages
 *   2. munmap             — release the ARM virtual address mapping
 *   3. close(fd)          — release the kernel's DMA-BUF handle
 *
 * Reversing steps 1 & 3 causes the kernel to keep a "ghost" reference to
 * the physical pages (QNN's MemHandle still points at them), preventing the
 * ION heap from reclaiming them until process death. This is the root cause
 * of the 2 GB silent leak reported in Phase 7 profiling.
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
 */
#pragma once
#ifndef SP_MEM_H
#define SP_MEM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque QNN memory handle pointer (avoids pulling in QNN headers here). */
typedef void* SP_QnnMemHandle;

/*
 * SP_Memory — one DMA-BUF-backed memory region.
 *
 * Fields must NOT be modified directly after sp_mem_alloc(). Use the
 * provided functions for registration and teardown.
 */
typedef struct SP_Memory {
    void*            ptr;    /* ARM-side mmap'd virtual address (MAP_FAILED if not mapped) */
    size_t           size;   /* byte count of the allocation                                */
    int              fd;     /* DMA-BUF file descriptor (-1 if not open)                   */
    SP_QnnMemHandle  qnn_h;  /* registered QNN MemHandle (NULL before registration)        */
} SP_Memory;

/* ── Allocation ─────────────────────────────────────────────────────────── */

/*
 * Allocate a DMA-BUF region of `bytes` via /dev/dma_heap/system (preferred,
 * Android 12+) or /dev/ion (fallback). Maps the region into the caller's
 * address space (PROT_READ|PROT_WRITE, MAP_SHARED).
 *
 * On success: mem->ptr, mem->fd, mem->size are populated; mem->qnn_h = NULL.
 * On failure: returns -1, mem is zeroed.
 *
 * Align: allocation is rounded up to 4096-byte pages by the kernel heap.
 */
int sp_mem_alloc(SP_Memory* mem, size_t bytes);

/* ── QNN registration ───────────────────────────────────────────────────── */

/*
 * Register the DMA-BUF FD with a QNN context so the HTP can DMA-map it.
 * Must be called AFTER sp_mem_alloc(). Stores the MemHandle in mem->qnn_h.
 *
 * `qnn_fn_mem_register` is the function pointer to QnnMem_register obtained
 * from the QNN interface table (caller owns the interface lifetime).
 *
 * Returns 0 on success, -1 on failure.
 */
typedef int (*sp_mem_qnn_register_fn)(void* ctx_handle,
                                       void* mem_descriptor,
                                       uint32_t num_descriptors,
                                       SP_QnnMemHandle* out_handle);

int sp_mem_register_qnn(SP_Memory* mem,
                        void* qnn_context,
                        sp_mem_qnn_register_fn register_fn,
                        uint32_t n_dims,
                        const uint32_t* dims,
                        uint32_t qnn_dtype);

/* ── Teardown ───────────────────────────────────────────────────────────── */

/*
 * Destroy a DMA-BUF memory object in the MANDATORY order:
 *   1. QnnMem_unregister (if qnn_h != NULL)
 *   2. munmap            (if ptr  != MAP_FAILED / NULL)
 *   3. close(fd)         (if fd   >= 0)
 *
 * Skipping step 1 before step 3 leaves orphaned ION buffers that cannot be
 * reclaimed — this is the "2 GB vanish" bug. Always call sp_mem_destroy()
 * rather than manually closing the FD or calling munmap().
 *
 * `unregister_fn` is the function pointer to QnnMem_deRegister from the QNN
 * interface (pass NULL to skip QNN unregistration — only safe if sp_mem_register_qnn
 * was never called or if the QNN context is already destroyed).
 *
 * After return, mem is zeroed (fd=-1, ptr=MAP_FAILED, qnn_h=NULL, size=0).
 */
typedef int (*sp_mem_qnn_unregister_fn)(SP_QnnMemHandle* handle, uint32_t num);

void sp_mem_destroy(SP_Memory* mem, sp_mem_qnn_unregister_fn unregister_fn);

/* ── Adaptive stream count ──────────────────────────────────────────────── */

/*
 * Read /proc/meminfo → MemAvailable and return the recommended stream count:
 *
 *   >= 7 GB → 4 streams  (Full residency: all 4 splits ~3.0 GB — "Lightning")
 *   >= 5 GB → 2 streams  (Ping-Pong: 2 of 4 splits ~1.4 GB — "Steady/Fast")
 *    < 5 GB → 1 stream   (Safe mode: one split at a time — "Bulletproof")
 *
 * NOTE: For Qwen3-4B on SM8450, HTP working set is limited to ~1.39 GB,
 * making 4-stream persistent impossible regardless of MemAvailable.
 * The thresholds here reflect SYSTEM ram for KV cache and DMA buffers, not
 * HTP capacity. The persistent-context decision is made separately in
 * sp_qnn_load_binary (it attempts and falls back on OOM).
 *
 * Returns 1, 2, or 4. Never returns 0 or 3.
 */
uint32_t sp_determine_stream_count(void);

/*
 * Read current MemAvailable from /proc/meminfo in bytes.
 * Returns 0 on parse failure.
 */
size_t sp_get_available_ram_bytes(void);

/* ── Global cleanup registry ────────────────────────────────────────────── */

/*
 * Register a SP_Memory object for emergency cleanup by sp_global_cleanup().
 * Call once per sp_mem_alloc(). The registry holds up to 256 entries.
 */
void sp_register_for_cleanup(SP_Memory* mem);

/*
 * Emergency teardown: iterate through the global registry and call
 * sp_mem_destroy() on every registered allocation. Call from signal handlers
 * or atexit() to prevent ION FD leaks across test runs.
 *
 * `unregister_fn` is passed to sp_mem_destroy for QNN deregistration.
 * Pass NULL only if QNN contexts are already destroyed.
 */
void sp_global_cleanup(sp_mem_qnn_unregister_fn unregister_fn);

/* ── Diagnostics ────────────────────────────────────────────────────────── */

/*
 * Print /sys/kernel/debug/dma_buf/bufinfo to stderr.
 * Shows which processes hold DMA-BUF references. Useful for detecting
 * orphaned buffers after a failed teardown. Requires root on production.
 */
void sp_mem_dump_bufinfo(void);

/*
 * Set up DMA heap / ION permissions for non-root access:
 *   chmod 666 /dev/dma_heap/system
 *   chmod 666 /dev/ion  (if present)
 *   setprop grp.adsprpc.unauth_enable 1
 *
 * Only effective if running with sufficient privileges (root or adb shell).
 * Call once before any sp_mem_alloc() when running in a non-system process.
 * Returns 0 if at least /dev/dma_heap/system is accessible after the call.
 */
int sp_mem_setup_permissions(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SP_MEM_H */
