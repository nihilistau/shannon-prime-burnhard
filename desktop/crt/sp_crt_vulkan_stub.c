// Shannon-Prime: CRT Vulkan stubs for non-Vulkan builds
// These provide link-time symbols that return error/NULL when Vulkan is not available.
// Mirrors the pattern of sp_crt_cuda_stub.c.

#include "sp_crt_vulkan.h"
#include <stddef.h>

sp_crt_vulkan_ctx_t* sp_crt_vulkan_init(void *vk_instance,
                                         int physical_device_index,
                                         int max_elements) {
    (void)vk_instance; (void)physical_device_index; (void)max_elements;
    return NULL;
}

void sp_crt_vulkan_free(sp_crt_vulkan_ctx_t *ctx) {
    (void)ctx;
}

uint32_t* sp_crt_vulkan_alloc(sp_crt_vulkan_ctx_t *ctx, size_t bytes) {
    (void)ctx; (void)bytes;
    return NULL;
}

void sp_crt_vulkan_buf_free(sp_crt_vulkan_ctx_t *ctx, uint32_t *buf) {
    (void)ctx; (void)buf;
}

int sp_crt_vulkan_memcpy_h2d(sp_crt_vulkan_ctx_t *ctx,
                              uint32_t *dst, const uint32_t *src,
                              size_t bytes) {
    (void)ctx; (void)dst; (void)src; (void)bytes;
    return -1;
}

int sp_crt_vulkan_memcpy_d2h(sp_crt_vulkan_ctx_t *ctx,
                              uint32_t *dst, const uint32_t *src,
                              size_t bytes) {
    (void)ctx; (void)dst; (void)src; (void)bytes;
    return -1;
}

int sp_crt_vulkan_matmul_mod_dispatch(sp_crt_vulkan_ctx_t *ctx,
                                      const uint32_t *d_A,
                                      const uint32_t *d_B,
                                      uint32_t *d_C,
                                      int M, int N, int K,
                                      uint32_t modulus) {
    (void)ctx; (void)d_A; (void)d_B; (void)d_C;
    (void)M; (void)N; (void)K; (void)modulus;
    return -1;
}

int sp_crt_vulkan_sync(sp_crt_vulkan_ctx_t *ctx) {
    (void)ctx;
    return -1;
}

void* sp_crt_vulkan_get_queue(sp_crt_vulkan_ctx_t *ctx) {
    (void)ctx;
    return NULL;
}

void* sp_crt_vulkan_get_fence(sp_crt_vulkan_ctx_t *ctx) {
    (void)ctx;
    return NULL;
}
