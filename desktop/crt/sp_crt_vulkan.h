// Shannon-Prime CRT: Vulkan compute dispatch for Ring M2
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
// Licensed under AGPLv3.
//
// Provides Vulkan compute shader dispatch for the second CRT ring
// (M2 = 2^31 - 19) on Intel UHD / any Vulkan-capable GPU.
// Ring M1 (Mersenne) runs on CUDA (RTX 2060).

#ifndef SP_CRT_VULKAN_H
#define SP_CRT_VULKAN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque Vulkan compute context.
// Holds VkDevice, pipeline, descriptor sets, command buffers, etc.
typedef struct sp_crt_vulkan_ctx sp_crt_vulkan_ctx_t;

// Initialize Vulkan compute context for CRT Ring M2 dispatch.
// physical_device_index: index from vkEnumeratePhysicalDevices (from sp_hetero_detect_gpus)
// vk_instance: existing VkInstance handle (from sp_hetero_sync)
// max_elements: max M*K or K*N or M*N elements (for buffer sizing)
// Returns opaque context, or NULL on failure.
sp_crt_vulkan_ctx_t* sp_crt_vulkan_init(void *vk_instance,
                                         int physical_device_index,
                                         int max_elements);

// Free all Vulkan resources.
void sp_crt_vulkan_free(sp_crt_vulkan_ctx_t *ctx);

// Allocate a device-local buffer. Returns opaque handle.
// The handle is an index into the context's buffer pool.
uint32_t* sp_crt_vulkan_alloc(sp_crt_vulkan_ctx_t *ctx, size_t bytes);

// Free a device buffer.
void sp_crt_vulkan_buf_free(sp_crt_vulkan_ctx_t *ctx, uint32_t *buf);

// Copy host → device (staging buffer path).
int sp_crt_vulkan_memcpy_h2d(sp_crt_vulkan_ctx_t *ctx,
                              uint32_t *dst, const uint32_t *src,
                              size_t bytes);

// Copy device → host.
int sp_crt_vulkan_memcpy_d2h(sp_crt_vulkan_ctx_t *ctx,
                              uint32_t *dst, const uint32_t *src,
                              size_t bytes);

// Dispatch modular matmul: C = (A × B) mod modulus
// A: [M×K], B: [K×N], C: [M×N], all uint32 in device memory.
int sp_crt_vulkan_matmul_mod_dispatch(sp_crt_vulkan_ctx_t *ctx,
                                      const uint32_t *d_A,
                                      const uint32_t *d_B,
                                      uint32_t *d_C,
                                      int M, int N, int K,
                                      uint32_t modulus);

// Wait for all submitted work to complete.
int sp_crt_vulkan_sync(sp_crt_vulkan_ctx_t *ctx);

// Get the VkQueue handle (for sp_hetero_sync integration).
void* sp_crt_vulkan_get_queue(sp_crt_vulkan_ctx_t *ctx);

// Get the VkFence handle (for barrier polling).
void* sp_crt_vulkan_get_fence(sp_crt_vulkan_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif // SP_CRT_VULKAN_H
