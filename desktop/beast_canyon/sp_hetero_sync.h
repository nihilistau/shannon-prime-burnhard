// Shannon-Prime Beast Canyon: Heterogeneous GPU Synchronization
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Cross-GPU barrier for dual-dispatch (CUDA + Vulkan/Level-Zero).
// The CPU orchestrator fires both GPUs and then waits at a barrier
// until BOTH return results. Pre-shredding of next expert happens
// during the wait window.
//
// Design:
//   - CUDA: cudaEvent_t recorded after kernel launch
//   - Vulkan: VkFence signaled after queue submit
//   - Level Zero: ze_event_handle_t (future, when L0 is added)
//   - CPU polling: yield-based with configurable spin count
//
// The barrier is the single point where the system can desync.
// If one GPU finishes before the other, the CPU uses that dead time
// for pre-shredding the next expert's weights.

#ifndef SP_HETERO_SYNC_H
#define SP_HETERO_SYNC_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// GPU Backend Types
// ============================================================================

typedef enum {
    SP_GPU_NONE     = 0,
    SP_GPU_CUDA     = 1,   // NVIDIA via CUDA runtime
    SP_GPU_VULKAN   = 2,   // Any Vulkan device (Intel Xe, AMD, etc.)
    SP_GPU_LEVEL_ZERO = 3, // Intel Level Zero (raw metal, future)
    SP_GPU_CPU_ONLY = 4,   // CPU fallback (no GPU)
} sp_gpu_type_t;

// ============================================================================
// GPU Device Descriptor
// ============================================================================

typedef struct {
    sp_gpu_type_t type;
    int           device_id;       // CUDA device or Vulkan physical device index
    char          name[128];       // Human-readable device name
    uint64_t      vram_bytes;      // Total VRAM
    uint64_t      vram_free;       // Available VRAM (at init time)

    // Backend-specific handles (opaque — cast at usage site)
    void         *stream;          // CUDA stream or VkQueue
    void         *event;           // CUDA event or VkFence or ze_event
    void         *command_list;    // Level Zero immediate command list (or NULL)
    void         *context;         // Backend context (CUDA context, VkDevice, ze_context)

    // Staging buffer for this GPU's results
    void         *result_buf;      // Device pointer to MLP output vector
    size_t        result_size;     // Size in bytes

    // Performance tracking
    uint64_t      total_dispatch_us;
    uint64_t      total_wait_us;
    uint64_t      dispatch_count;
} sp_gpu_device_t;

// ============================================================================
// Heterogeneous Barrier
// ============================================================================

typedef struct {
    // GPU devices (up to 2 for the dual-dispatch system)
    sp_gpu_device_t  gpu[2];
    int              n_gpus;

    // Barrier configuration
    int              spin_count;     // Spins before yielding (default 1000)
    bool             use_blocking;   // Use blocking wait (save power) vs spin

    // Barrier state
    bool             gpu_done[2];    // Per-GPU completion flag
    uint64_t         last_barrier_us; // Time of last barrier wait

    // Pre-shredder callback: called during barrier wait window
    // to overlap next expert's dequantization with GPU compute.
    void           (*prefetch_callback)(void *user_data, int next_expert_id);
    void            *prefetch_user_data;
    int              next_expert_hint; // Set by orchestrator before barrier

    // Diagnostics
    uint64_t         total_barrier_us;
    uint64_t         total_barriers;
    uint64_t         max_barrier_us;
    uint64_t         min_barrier_us;
    uint64_t         total_prefetch_us;  // Time spent pre-shredding during wait
} sp_hetero_barrier_t;

// ============================================================================
// Public API
// ============================================================================

// Initialize the barrier with detected GPU devices.
int sp_hetero_barrier_init(sp_hetero_barrier_t *barrier);
void sp_hetero_barrier_free(sp_hetero_barrier_t *barrier);

// --- GPU Discovery ---

// Detect available GPUs and populate barrier->gpu[].
// Returns number of GPUs found (0 = CPU-only mode).
int sp_hetero_detect_gpus(sp_hetero_barrier_t *barrier);

// Manually add a GPU device (for testing or custom configs).
int sp_hetero_add_gpu(sp_hetero_barrier_t *barrier,
                      sp_gpu_type_t type, int device_id);

// --- Dispatch ---

// Signal that GPU[idx] has been launched (kernel submitted).
// The barrier tracks timing from this point.
void sp_hetero_mark_dispatched(sp_hetero_barrier_t *barrier, int gpu_idx);

// Signal that GPU[idx] has completed (event/fence signaled).
void sp_hetero_mark_done(sp_hetero_barrier_t *barrier, int gpu_idx);

// --- The Barrier ---

// Wait for ALL dispatched GPUs to complete.
// During the wait, calls prefetch_callback if set (pre-shredding).
// Returns total barrier wait time in microseconds.
uint64_t sp_hetero_barrier_wait(sp_hetero_barrier_t *barrier);

// Poll without blocking — returns true if all GPUs are done.
bool sp_hetero_barrier_poll(sp_hetero_barrier_t *barrier);

// --- Backend-specific event operations ---

// CUDA: record event on stream, synchronize event.
#ifdef SP_WITH_CUDA
int sp_hetero_cuda_record_event(sp_gpu_device_t *gpu);
int sp_hetero_cuda_sync_event(sp_gpu_device_t *gpu);
#endif

// Vulkan: submit fence, wait for fence.
#ifdef SP_WITH_VULKAN
int sp_hetero_vulkan_submit_fence(sp_gpu_device_t *gpu);
int sp_hetero_vulkan_wait_fence(sp_gpu_device_t *gpu);
#endif

// --- Diagnostics ---

void sp_hetero_barrier_print_status(const sp_hetero_barrier_t *barrier);

// Get average barrier wait time in microseconds.
static inline double sp_hetero_avg_barrier_us(const sp_hetero_barrier_t *barrier) {
    return barrier->total_barriers > 0
        ? (double)barrier->total_barrier_us / barrier->total_barriers
        : 0.0;
}

#ifdef __cplusplus
}
#endif

#endif // SP_HETERO_SYNC_H
