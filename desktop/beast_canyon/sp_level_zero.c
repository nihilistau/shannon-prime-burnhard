/*
 * sp_level_zero.c — Shannon-Prime Beast Canyon: Level Zero Dynamic Loader
 *
 * Runtime loading of ze_loader.dll (Windows) or libze_loader.so (Linux).
 * Discovers Intel UHD integrated GPU, creates context + immediate command
 * list for zero-copy Ring Bus dispatch.
 *
 * NO compile-time dependency on Intel oneAPI SDK.
 *
 * Copyright (c) 2026 Ray Daniels
 * Licensed under the GNU Affero General Public License v3.0
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include "sp_level_zero.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  define SP_L0_LIB_NAME  "ze_loader.dll"
#else
#  include <dlfcn.h>
#  define SP_L0_LIB_NAME  "libze_loader.so.1"
#endif


/* ====================================================================
 * Dynamic library helpers
 * ==================================================================== */

static void* sp_l0_load_library(const char *name) {
#ifdef _WIN32
    return (void*)LoadLibraryA(name);
#else
    return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
#endif
}

static void sp_l0_unload_library(void *handle) {
    if (!handle) return;
#ifdef _WIN32
    FreeLibrary((HMODULE)handle);
#else
    dlclose(handle);
#endif
}

static void* sp_l0_get_proc(void *handle, const char *name) {
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)handle, name);
#else
    return dlsym(handle, name);
#endif
}


/* ====================================================================
 * Load all function pointers
 * ==================================================================== */

#define SP_L0_LOAD(fn) \
    l0->fn = (pfn_##fn)sp_l0_get_proc(l0->dll_handle, #fn); \
    if (!l0->fn) { \
        fprintf(stderr, "[sp-l0] WARN: missing symbol: %s\n", #fn); \
        missing++; \
    }

static int sp_l0_load_symbols(sp_l0_t *l0) {
    int missing = 0;

    SP_L0_LOAD(zeInit);
    SP_L0_LOAD(zeDriverGet);
    SP_L0_LOAD(zeDeviceGet);
    SP_L0_LOAD(zeDeviceGetProperties);
    SP_L0_LOAD(zeContextCreate);
    SP_L0_LOAD(zeContextDestroy);
    SP_L0_LOAD(zeCommandQueueCreate);
    SP_L0_LOAD(zeCommandQueueDestroy);
    SP_L0_LOAD(zeCommandQueueSynchronize);
    SP_L0_LOAD(zeCommandListCreateImmediate);
    SP_L0_LOAD(zeCommandListDestroy);
    SP_L0_LOAD(zeCommandListReset);
    SP_L0_LOAD(zeCommandListAppendMemoryCopy);
    SP_L0_LOAD(zeCommandListAppendBarrier);
    SP_L0_LOAD(zeCommandListAppendLaunchKernel);
    SP_L0_LOAD(zeMemAllocShared);
    SP_L0_LOAD(zeMemFree);
    SP_L0_LOAD(zeEventPoolCreate);
    SP_L0_LOAD(zeEventPoolDestroy);
    SP_L0_LOAD(zeEventCreate);
    SP_L0_LOAD(zeEventDestroy);
    SP_L0_LOAD(zeEventHostSynchronize);
    SP_L0_LOAD(zeEventHostReset);
    SP_L0_LOAD(zeEventQueryStatus);
    SP_L0_LOAD(zeModuleCreate);
    SP_L0_LOAD(zeModuleDestroy);
    SP_L0_LOAD(zeKernelCreate);
    SP_L0_LOAD(zeKernelDestroy);
    SP_L0_LOAD(zeKernelSetGroupSize);
    SP_L0_LOAD(zeKernelSetArgumentValue);

    /* Core functions are required; module/kernel functions are optional
     * (we may not have SPIR-V kernels yet during bringup). */
    int critical_missing = 0;
    if (!l0->zeInit) critical_missing++;
    if (!l0->zeDriverGet) critical_missing++;
    if (!l0->zeDeviceGet) critical_missing++;
    if (!l0->zeDeviceGetProperties) critical_missing++;
    if (!l0->zeContextCreate) critical_missing++;
    if (!l0->zeCommandListCreateImmediate) critical_missing++;
    if (!l0->zeMemAllocShared) critical_missing++;
    if (!l0->zeMemFree) critical_missing++;

    if (critical_missing > 0) {
        fprintf(stderr, "[sp-l0] ERROR: %d critical L0 symbols missing\n", critical_missing);
        return -1;
    }

    l0->loaded = true;
    return 0;
}

#undef SP_L0_LOAD


/* ====================================================================
 * sp_l0_init — Full initialisation sequence
 * ==================================================================== */

int sp_l0_init(sp_l0_t *l0) {
    if (!l0) return -1;
    memset(l0, 0, sizeof(*l0));

    /* ── Step 1: Load ze_loader.dll ──────────────────────────────── */
    l0->dll_handle = sp_l0_load_library(SP_L0_LIB_NAME);
    if (!l0->dll_handle) {
        fprintf(stderr, "[sp-l0] %s not found — Level Zero not available\n", SP_L0_LIB_NAME);
        return -1;
    }
    fprintf(stderr, "[sp-l0] Loaded %s\n", SP_L0_LIB_NAME);

    /* ── Step 2: Resolve function pointers ───────────────────────── */
    if (sp_l0_load_symbols(l0) != 0) {
        sp_l0_free(l0);
        return -1;
    }

    /* ── Step 3: zeInit ──────────────────────────────────────────── */
    ze_result_t r = l0->zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (r != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "[sp-l0] zeInit failed: 0x%08x\n", (unsigned)r);
        sp_l0_free(l0);
        return -3;
    }

    /* ── Step 4: Get first driver ────────────────────────────────── */
    uint32_t driver_count = 1;
    r = l0->zeDriverGet(&driver_count, &l0->driver);
    if (r != ZE_RESULT_SUCCESS || driver_count == 0 || !l0->driver) {
        fprintf(stderr, "[sp-l0] No L0 drivers found\n");
        sp_l0_free(l0);
        return -2;
    }

    /* ── Step 5: Enumerate devices, find integrated Intel GPU ────── */
    uint32_t dev_count = 0;
    r = l0->zeDeviceGet(l0->driver, &dev_count, NULL);
    if (r != ZE_RESULT_SUCCESS || dev_count == 0) {
        fprintf(stderr, "[sp-l0] No L0 devices on driver\n");
        sp_l0_free(l0);
        return -2;
    }

    ze_device_handle_t *devices = (ze_device_handle_t*)calloc(dev_count, sizeof(ze_device_handle_t));
    if (!devices) {
        sp_l0_free(l0);
        return -3;
    }
    l0->zeDeviceGet(l0->driver, &dev_count, devices);

    l0->device_found = false;
    for (uint32_t i = 0; i < dev_count; i++) {
        ze_device_properties_t props;
        memset(&props, 0, sizeof(props));
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

        r = l0->zeDeviceGetProperties(devices[i], &props);
        if (r != ZE_RESULT_SUCCESS) continue;

        fprintf(stderr, "[sp-l0] Device[%u]: %s (vendor=0x%04x type=%d flags=0x%x)\n",
                i, props.name, props.vendorId, props.type, props.flags);

        /* We want: GPU + integrated (Intel UHD on Ring Bus) */
        if (props.type == ZE_DEVICE_TYPE_GPU &&
            (props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED))
        {
            l0->device = devices[i];
            l0->props  = props;
            l0->device_found = true;
            fprintf(stderr, "[sp-l0] ✓ Selected: %s (%u EUs, %u slices × %u subslices)\n",
                    props.name,
                    props.numSlices * props.numSubslicesPerSlice * props.numEUsPerSubslice,
                    props.numSlices, props.numSubslicesPerSlice);
            break;
        }
    }
    free(devices);

    if (!l0->device_found) {
        fprintf(stderr, "[sp-l0] No integrated Intel GPU found via Level Zero\n");
        sp_l0_free(l0);
        return -2;
    }

    /* ── Step 6: Create context ──────────────────────────────────── */
    ze_context_desc_t ctx_desc;
    memset(&ctx_desc, 0, sizeof(ctx_desc));
    ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;

    r = l0->zeContextCreate(l0->driver, &ctx_desc, &l0->context);
    if (r != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "[sp-l0] zeContextCreate failed: 0x%08x\n", (unsigned)r);
        sp_l0_free(l0);
        return -3;
    }

    /* ── Step 7: Create immediate command list ───────────────────── */
    /*
     * Immediate command lists execute commands as they are appended —
     * no explicit execute/submit step.  This matches the shadow-steal
     * use case where we want to fire kernels with minimal latency.
     *
     * We use ordinal 0 (first command queue group, typically compute).
     */
    ze_command_queue_desc_t cq_desc;
    memset(&cq_desc, 0, sizeof(cq_desc));
    cq_desc.stype    = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    cq_desc.ordinal  = 0;
    cq_desc.index    = 0;
    cq_desc.mode     = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    cq_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    r = l0->zeCommandListCreateImmediate(l0->context, l0->device,
                                          &cq_desc, &l0->cmd_list);
    if (r != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "[sp-l0] zeCommandListCreateImmediate failed: 0x%08x\n", (unsigned)r);
        sp_l0_free(l0);
        return -3;
    }

    /* ── Step 8: Create event pool (8 events for shadow-steal) ──── */
    ze_event_pool_desc_t ep_desc;
    memset(&ep_desc, 0, sizeof(ep_desc));
    ep_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    ep_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    ep_desc.count = 8;

    r = l0->zeEventPoolCreate(l0->context, &ep_desc, 1, &l0->device, &l0->event_pool);
    if (r != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "[sp-l0] zeEventPoolCreate failed: 0x%08x\n", (unsigned)r);
        /* Non-fatal — we can still dispatch, just can't async-query completion */
        l0->event_pool = NULL;
    }

    fprintf(stderr, "[sp-l0] Level Zero READY — %s on Ring Bus\n", l0->props.name);
    return 0;
}


/* ====================================================================
 * sp_l0_free
 * ==================================================================== */

void sp_l0_free(sp_l0_t *l0) {
    if (!l0) return;

    /* Free tracked USM allocations */
    for (int i = 0; i < l0->n_usm_allocs; i++) {
        if (l0->usm_allocs[i] && l0->zeMemFree && l0->context) {
            l0->zeMemFree(l0->context, l0->usm_allocs[i]);
        }
    }

    /* Destroy event pool */
    if (l0->event_pool && l0->zeEventPoolDestroy) {
        l0->zeEventPoolDestroy(l0->event_pool);
    }

    /* Destroy command list */
    if (l0->cmd_list && l0->zeCommandListDestroy) {
        l0->zeCommandListDestroy(l0->cmd_list);
    }

    /* Destroy command queue (if we created one separately) */
    if (l0->cmd_queue && l0->zeCommandQueueDestroy) {
        l0->zeCommandQueueDestroy(l0->cmd_queue);
    }

    /* Destroy context */
    if (l0->context && l0->zeContextDestroy) {
        l0->zeContextDestroy(l0->context);
    }

    /* Unload DLL */
    if (l0->dll_handle) {
        sp_l0_unload_library(l0->dll_handle);
    }

    memset(l0, 0, sizeof(*l0));
}


/* ====================================================================
 * USM Shared Memory
 * ==================================================================== */

void* sp_l0_alloc_shared(sp_l0_t *l0, size_t bytes, size_t alignment) {
    if (!l0 || !l0->loaded || !l0->context || !l0->device) return NULL;
    if (l0->n_usm_allocs >= 16) {
        fprintf(stderr, "[sp-l0] ERROR: USM alloc tracking full (max 16)\n");
        return NULL;
    }

    ze_device_mem_alloc_desc_t dev_desc;
    memset(&dev_desc, 0, sizeof(dev_desc));
    dev_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;

    ze_host_mem_alloc_desc_t host_desc;
    memset(&host_desc, 0, sizeof(host_desc));
    host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;

    void *ptr = NULL;
    ze_result_t r = l0->zeMemAllocShared(l0->context, &dev_desc, &host_desc,
                                          bytes, alignment, l0->device, &ptr);
    if (r != ZE_RESULT_SUCCESS || !ptr) {
        fprintf(stderr, "[sp-l0] zeMemAllocShared(%zu bytes) failed: 0x%08x\n",
                bytes, (unsigned)r);
        return NULL;
    }

    l0->usm_allocs[l0->n_usm_allocs++] = ptr;
    return ptr;
}

void sp_l0_free_mem(sp_l0_t *l0, void *ptr) {
    if (!l0 || !ptr || !l0->zeMemFree || !l0->context) return;

    l0->zeMemFree(l0->context, ptr);

    /* Remove from tracking */
    for (int i = 0; i < l0->n_usm_allocs; i++) {
        if (l0->usm_allocs[i] == ptr) {
            l0->usm_allocs[i] = l0->usm_allocs[--l0->n_usm_allocs];
            break;
        }
    }
}


/* ====================================================================
 * Events
 * ==================================================================== */

int sp_l0_create_event(sp_l0_t *l0, ze_event_handle_t *out_event, uint32_t index) {
    if (!l0 || !out_event || !l0->event_pool || !l0->zeEventCreate) return -1;

    ze_event_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.stype  = ZE_STRUCTURE_TYPE_EVENT_DESC;
    desc.index  = index;
    desc.signal = 0x01;  /* ZE_EVENT_SCOPE_FLAG_HOST */
    desc.wait   = 0x01;  /* ZE_EVENT_SCOPE_FLAG_HOST */

    ze_result_t r = l0->zeEventCreate(l0->event_pool, &desc, out_event);
    return (r == ZE_RESULT_SUCCESS) ? 0 : -1;
}

int sp_l0_sync_event(sp_l0_t *l0, ze_event_handle_t event, uint64_t timeout_ns) {
    if (!l0 || !event || !l0->zeEventHostSynchronize) return -1;

    ze_result_t r = l0->zeEventHostSynchronize(event, timeout_ns);
    if (r == ZE_RESULT_SUCCESS)  return 0;
    if (r == ZE_RESULT_NOT_READY) return 1;
    return -1;
}

int sp_l0_query_event(sp_l0_t *l0, ze_event_handle_t event) {
    if (!l0 || !event || !l0->zeEventQueryStatus) return -1;

    ze_result_t r = l0->zeEventQueryStatus(event);
    if (r == ZE_RESULT_SUCCESS)  return 0;
    if (r == ZE_RESULT_NOT_READY) return 1;
    return -1;
}

void sp_l0_reset_event(sp_l0_t *l0, ze_event_handle_t event) {
    if (!l0 || !event || !l0->zeEventHostReset) return;
    l0->zeEventHostReset(event);
}


/* ====================================================================
 * Command list operations
 * ==================================================================== */

int sp_l0_reset_cmd_list(sp_l0_t *l0) {
    if (!l0 || !l0->cmd_list || !l0->zeCommandListReset) return -1;
    ze_result_t r = l0->zeCommandListReset(l0->cmd_list);
    return (r == ZE_RESULT_SUCCESS) ? 0 : -1;
}

int sp_l0_append_copy(sp_l0_t *l0, void *dst, const void *src,
                      size_t bytes, ze_event_handle_t signal_event) {
    if (!l0 || !l0->cmd_list || !l0->zeCommandListAppendMemoryCopy) return -1;
    ze_result_t r = l0->zeCommandListAppendMemoryCopy(
        l0->cmd_list, dst, src, bytes, signal_event, 0, NULL);
    return (r == ZE_RESULT_SUCCESS) ? 0 : -1;
}

int sp_l0_append_barrier(sp_l0_t *l0, ze_event_handle_t signal_event) {
    if (!l0 || !l0->cmd_list || !l0->zeCommandListAppendBarrier) return -1;
    ze_result_t r = l0->zeCommandListAppendBarrier(
        l0->cmd_list, signal_event, 0, NULL);
    return (r == ZE_RESULT_SUCCESS) ? 0 : -1;
}


/* ====================================================================
 * Module + Kernel
 * ==================================================================== */

int sp_l0_build_module_spirv(sp_l0_t *l0, const uint8_t *spirv_data,
                             size_t spirv_size, ze_module_handle_t *out_module) {
    if (!l0 || !spirv_data || !out_module || !l0->zeModuleCreate) return -1;

    ze_module_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.stype       = ZE_STRUCTURE_TYPE_MODULE_DESC;
    desc.format      = ZE_MODULE_FORMAT_IL_SPIRV;
    desc.inputSize   = spirv_size;
    desc.pInputModule = spirv_data;
    desc.pBuildFlags  = "-ze-opt-level=2";

    ze_result_t r = l0->zeModuleCreate(l0->context, l0->device,
                                        &desc, out_module, NULL);
    if (r != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "[sp-l0] zeModuleCreate failed: 0x%08x\n", (unsigned)r);
        return -1;
    }
    return 0;
}

int sp_l0_create_kernel(sp_l0_t *l0, ze_module_handle_t module,
                        const char *kernel_name, ze_kernel_handle_t *out_kernel) {
    if (!l0 || !module || !kernel_name || !out_kernel || !l0->zeKernelCreate) return -1;

    ze_kernel_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.stype       = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    desc.pKernelName = kernel_name;

    ze_result_t r = l0->zeKernelCreate(module, &desc, out_kernel);
    if (r != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "[sp-l0] zeKernelCreate('%s') failed: 0x%08x\n",
                kernel_name, (unsigned)r);
        return -1;
    }
    return 0;
}

int sp_l0_launch_kernel(sp_l0_t *l0, ze_kernel_handle_t kernel,
                        uint32_t group_count_x, uint32_t group_count_y,
                        uint32_t group_count_z,
                        ze_event_handle_t signal_event) {
    if (!l0 || !kernel || !l0->cmd_list || !l0->zeCommandListAppendLaunchKernel) return -1;

    ze_group_count_t gc;
    gc.groupCountX = group_count_x;
    gc.groupCountY = group_count_y;
    gc.groupCountZ = group_count_z;

    ze_result_t r = l0->zeCommandListAppendLaunchKernel(
        l0->cmd_list, kernel, &gc, signal_event, 0, NULL);
    if (r != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "[sp-l0] zeCommandListAppendLaunchKernel failed: 0x%08x\n",
                (unsigned)r);
        return -1;
    }
    return 0;
}
