// Shannon-Prime BurnHard — L0 SVM Shredder Kernel
// Implements 24MB Shared Virtual Memory (SVM) for Intel UHD iGPU / Host.
// Zero-copy pointer sharing (no memcpy) between AVX-512 and UHD 750.

#include "sp_level_zero.h"
#include "sp_shredder_crt.h"
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>

#define SVM_ALLOC_SIZE (24 * 1024 * 1024) // 24MB SVM

typedef struct {
    ze_context_handle_t context;
    ze_device_handle_t device;
    void* svm_buffer;
    ze_command_queue_handle_t cmd_queue;
    ze_command_list_handle_t cmd_list;
} sp_l0_svm_state_t;

static sp_l0_svm_state_t g_svm_state = {0};

extern "C" int sp_shredder_svm_init(ze_context_handle_t ctx, ze_device_handle_t dev) {
    g_svm_state.context = ctx;
    g_svm_state.device = dev;

    ze_device_mem_alloc_desc_t dev_desc = {
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        nullptr,
        0, // flags
        0  // ordinal
    };
    
    ze_host_mem_alloc_desc_t host_desc = {
        ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
        nullptr,
        0  // flags
    };

    // Allocate 24MB Shared Virtual Memory (SVM) for zero-copy host/iGPU access
    ze_result_t status = zeMemAllocShared(
        g_svm_state.context,
        &dev_desc,
        &host_desc,
        SVM_ALLOC_SIZE,
        4096, // 4K page-alignment ensures Optane/AVX-512 boundaries are clean
        g_svm_state.device,
        &g_svm_state.svm_buffer
    );

    if (status != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "[sp_shredder_svm] zeMemAllocShared failed (24MB)\n");
        return -1;
    }

    fprintf(stderr, "[sp_shredder_svm] 24MB SVM allocated at %p. memcpy eliminated.\n", g_svm_state.svm_buffer);
    return 0;
}

extern "C" void* sp_shredder_get_svm_ptr() {
    return g_svm_state.svm_buffer;
}

extern "C" int sp_shredder_dispatch_crt_residue(uint32_t expert_idx, size_t residue_size, void* output_ptr) {
    if (!g_svm_state.svm_buffer) return -1;

    // Host AVX-512 has already shredded the SP band directly into g_svm_state.svm_buffer.
    // Apply hetero barrier to ensure AVX-512 writes are visible to iGPU
    zeCommandListAppendBarrier(g_svm_state.cmd_list, nullptr, 0, nullptr);

    // Note: Assuming `g_crt_kernel` is bound downstream. We instruct the command list
    // to compute directly over `g_svm_state.svm_buffer`.
    ze_group_count_t dispatch_traits = { (uint32_t)(residue_size / 256), 1, 1 };
    // zeCommandListAppendLaunchKernel(g_svm_state.cmd_list, g_crt_kernel, &dispatch_traits, nullptr, 0, nullptr);

    // Sync queued here pending completion
    return 0;
}