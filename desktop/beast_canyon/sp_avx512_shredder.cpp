// Shannon-Prime BurnHard — AVX-512 Optane Shredder
// Streams 8-bit quantized bands from Optane mmap directly into 24MB SVM pointer
// bcast scaling into fp32, staged directly to LLC for iGPU consumption.

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

extern "C" void* sp_shredder_get_svm_ptr();

extern "C" int sp_avx512_shred_optane_to_svm(const void* optane_mmap_ptr, size_t band_bytes, float global_scale) {
    if (!optane_mmap_ptr) return -1;
    
    float* svm_ptr = (float*)sp_shredder_get_svm_ptr();
    if (!svm_ptr) {
        fprintf(stderr, "[sp_avx512_shredder] SVM pointer missing. Call sp_shredder_svm_init first.\n");
        return -1;
    }

    const uint8_t* in_ptr = (const uint8_t*)optane_mmap_ptr;
    size_t num_vectors = band_bytes / 16; 

    // AVX-512: Broadcast the float scale across all 16 lanes
    __m512 v_scale = _mm512_set1_ps(global_scale);

    // Unroll manually to process 16 bytes per loop iteration (one ZMM float vector)
    for (size_t i = 0; i < num_vectors; i++) {
        // Load 128-bits (16 x 8-bit ints)
        __m128i v_in_8 = _mm_loadu_si128((const __m128i*)(in_ptr + i * 16));
        
        // Zero-extend 8-bit to 32-bit integers
        __m512i v_in_32 = _mm512_cvtepu8_epi32(v_in_8);
        
        // Convert to 32-bit floats
        __m512 v_f32 = _mm512_cvtepi32_ps(v_in_32);
        
        // Apply band dequantization scale
        __m512 v_out = _mm512_mul_ps(v_f32, v_scale);
        
        // Stream into SVM (LLC/iGPU shared).
        // _mm512_store_ps stages nicely into LLC for immediate iGPU pickup.
        _mm512_store_ps(svm_ptr + i * 16, v_out);
    }

    return 0;
}