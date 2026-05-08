// Shannon-Prime: CRT CUDA stubs for non-CUDA builds
// These provide link-time symbols that return error/NULL when CUDA is not available.

#include <stdint.h>
#include <stddef.h>

void sp_crt_cuda_matmul_mersenne(const uint32_t *d_A, const uint32_t *d_B,
                                  uint32_t *d_C,
                                  int M, int N, int K,
                                  void *stream) {
    (void)d_A; (void)d_B; (void)d_C; (void)M; (void)N; (void)K; (void)stream;
}

void sp_crt_cuda_matmul_mod(const uint32_t *d_A, const uint32_t *d_B,
                             uint32_t *d_C,
                             int M, int N, int K,
                             uint64_t modulus,
                             void *stream) {
    (void)d_A; (void)d_B; (void)d_C; (void)M; (void)N; (void)K; (void)modulus; (void)stream;
}

void sp_crt_cuda_quantize(const float *d_input, uint32_t *d_output,
                           int n, double scale, int64_t zero_point,
                           uint64_t modulus, void *stream) {
    (void)d_input; (void)d_output; (void)n; (void)scale; (void)zero_point;
    (void)modulus; (void)stream;
}

void sp_crt_cuda_quantize_symmetric(const float *d_input, uint32_t *d_output,
                                     int n, double scale, uint64_t modulus,
                                     void *stream) {
    (void)d_input; (void)d_output; (void)n; (void)scale; (void)modulus; (void)stream;
}

void* sp_crt_cuda_alloc(size_t bytes) {
    (void)bytes;
    return (void*)0;
}

void sp_crt_cuda_free(void *ptr) {
    (void)ptr;
}

int sp_crt_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes, void *stream) {
    (void)dst; (void)src; (void)bytes; (void)stream;
    return -1;
}

int sp_crt_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes, void *stream) {
    (void)dst; (void)src; (void)bytes; (void)stream;
    return -1;
}

int sp_crt_cuda_stream_sync(void *stream) {
    (void)stream;
    return -1;
}

void* sp_crt_cuda_stream_create(void) {
    return (void*)0;
}

void sp_crt_cuda_stream_destroy(void *stream) {
    (void)stream;
}

// Vulkan stub
void sp_crt_vulkan_matmul_mod(const uint32_t *d_A, const uint32_t *d_B,
                               uint32_t *d_C,
                               int M, int N, int K,
                               uint64_t modulus,
                               void *vk_queue) {
    (void)d_A; (void)d_B; (void)d_C; (void)M; (void)N; (void)K; (void)modulus; (void)vk_queue;
}
