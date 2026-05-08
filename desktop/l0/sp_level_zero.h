/*
 * sp_level_zero.h — Shannon-Prime Beast Canyon: Level Zero Dynamic Loader
 *
 * Minimal Level Zero type definitions and dynamic function loader.
 * Loads ze_loader.dll at runtime via LoadLibrary/GetProcAddress —
 * NO Intel oneAPI SDK installation required for compilation.
 *
 * We only define the subset of the L0 API that Beast Canyon uses:
 *   - Device discovery (zeInit, zeDriverGet, zeDeviceGet, properties)
 *   - Context + command queue/list creation
 *   - USM shared allocation (Ring Bus zero-copy for UHD iGPU)
 *   - Event pool + events (sync primitives)
 *   - Module + kernel creation (for the M2 residue matmul kernel)
 *   - Kernel launch + barrier
 *
 * Copyright (c) 2026 Ray Daniels
 * Licensed under the GNU Affero General Public License v3.0
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef SP_LEVEL_ZERO_H
#define SP_LEVEL_ZERO_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ====================================================================
 * Minimal Level Zero Type Definitions
 * ====================================================================
 * These mirror the official ze_api.h types but only what we use.
 * Handles are opaque pointers.  Enums are uint32_t-compatible.
 */

/* --- Result codes --- */
typedef enum {
    ZE_RESULT_SUCCESS                       = 0,
    ZE_RESULT_NOT_READY                     = 1,
    ZE_RESULT_ERROR_DEVICE_LOST             = 0x70000001,
    ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY      = 0x70000002,
    ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY    = 0x70000003,
    ZE_RESULT_ERROR_MODULE_BUILD_FAILURE    = 0x70000004,
    ZE_RESULT_ERROR_UNINITIALIZED           = 0x78000001,
    ZE_RESULT_ERROR_UNSUPPORTED_VERSION     = 0x78000002,
    ZE_RESULT_ERROR_UNSUPPORTED_FEATURE     = 0x78000003,
    ZE_RESULT_ERROR_INVALID_ARGUMENT        = 0x78000004,
    ZE_RESULT_ERROR_INVALID_NULL_HANDLE     = 0x78000005,
    ZE_RESULT_ERROR_INVALID_NULL_POINTER    = 0x78000006,
    ZE_RESULT_ERROR_INVALID_SIZE            = 0x78000007,
    ZE_RESULT_ERROR_UNKNOWN                 = 0x7ffffffe,
} ze_result_t;

/* --- Handles (opaque pointers) --- */
typedef struct _ze_driver_handle_t*        ze_driver_handle_t;
typedef struct _ze_device_handle_t*        ze_device_handle_t;
typedef struct _ze_context_handle_t*       ze_context_handle_t;
typedef struct _ze_command_queue_handle_t*  ze_command_queue_handle_t;
typedef struct _ze_command_list_handle_t*   ze_command_list_handle_t;
typedef struct _ze_event_pool_handle_t*    ze_event_pool_handle_t;
typedef struct _ze_event_handle_t*         ze_event_handle_t;
typedef struct _ze_module_handle_t*        ze_module_handle_t;
typedef struct _ze_kernel_handle_t*        ze_kernel_handle_t;

/* --- Init flags --- */
typedef uint32_t ze_init_flags_t;
#define ZE_INIT_FLAG_GPU_ONLY  0x01

/* --- Device type --- */
typedef enum {
    ZE_DEVICE_TYPE_GPU  = 1,
    ZE_DEVICE_TYPE_CPU  = 2,
    ZE_DEVICE_TYPE_FPGA = 3,
    ZE_DEVICE_TYPE_MCA  = 4,
    ZE_DEVICE_TYPE_VPU  = 5,
} ze_device_type_t;

/* --- Device property flags --- */
typedef uint32_t ze_device_property_flags_t;
#define ZE_DEVICE_PROPERTY_FLAG_INTEGRATED  0x01
#define ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE   0x02
#define ZE_DEVICE_PROPERTY_FLAG_ECC         0x04
#define ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING 0x08

/* --- Device properties (partial — only what we query) --- */
#define ZE_MAX_DEVICE_NAME  256

typedef struct {
    uint32_t                    stype;          /* ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES */
    void*                       pNext;
    ze_device_type_t            type;
    uint32_t                    vendorId;
    uint32_t                    deviceId;
    ze_device_property_flags_t  flags;
    uint32_t                    subdeviceId;
    uint32_t                    coreClockRate;
    uint64_t                    maxMemAllocSize;
    uint32_t                    maxHardwareContexts;
    uint32_t                    maxCommandQueuePriority;
    uint32_t                    numThreadsPerEU;
    uint32_t                    physicalEUSimdWidth;
    uint32_t                    numEUsPerSubslice;
    uint32_t                    numSubslicesPerSlice;
    uint32_t                    numSlices;
    uint64_t                    timerResolution;
    uint32_t                    timestampValidBits;
    uint32_t                    kernelTimestampValidBits;
    uint8_t                     uuid[16];
    char                        name[ZE_MAX_DEVICE_NAME];
} ze_device_properties_t;

#define ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES         3
#define ZE_STRUCTURE_TYPE_CONTEXT_DESC              0x12
#define ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC         0x14
#define ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC          0x15
#define ZE_STRUCTURE_TYPE_EVENT_POOL_DESC            0x16
#define ZE_STRUCTURE_TYPE_EVENT_DESC                 0x17
#define ZE_STRUCTURE_TYPE_MODULE_DESC                0x19
#define ZE_STRUCTURE_TYPE_KERNEL_DESC                0x1c
#define ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC      0x1f
#define ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC         0x20

/* --- Context descriptor --- */
typedef struct {
    uint32_t  stype;    /* ZE_STRUCTURE_TYPE_CONTEXT_DESC */
    void*     pNext;
    uint32_t  flags;
} ze_context_desc_t;

/* --- Command queue descriptor --- */
typedef enum {
    ZE_COMMAND_QUEUE_MODE_DEFAULT  = 0,
    ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1,
    ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS = 2,
} ze_command_queue_mode_t;

typedef enum {
    ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0,
    ZE_COMMAND_QUEUE_PRIORITY_LOW    = 1,
    ZE_COMMAND_QUEUE_PRIORITY_HIGH   = 2,
} ze_command_queue_priority_t;

typedef struct {
    uint32_t                     stype;     /* ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC */
    void*                        pNext;
    uint32_t                     ordinal;
    uint32_t                     index;
    uint32_t                     flags;
    ze_command_queue_mode_t       mode;
    ze_command_queue_priority_t   priority;
} ze_command_queue_desc_t;

/* --- Command list descriptor --- */
typedef struct {
    uint32_t  stype;    /* ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC */
    void*     pNext;
    uint32_t  commandQueueGroupOrdinal;
    uint32_t  flags;
} ze_command_list_desc_t;

/* --- Event pool descriptor --- */
typedef uint32_t ze_event_pool_flags_t;
#define ZE_EVENT_POOL_FLAG_HOST_VISIBLE  0x01
#define ZE_EVENT_POOL_FLAG_IPC           0x02
#define ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP 0x04

typedef struct {
    uint32_t              stype;    /* ZE_STRUCTURE_TYPE_EVENT_POOL_DESC */
    void*                 pNext;
    ze_event_pool_flags_t flags;
    uint32_t              count;
} ze_event_pool_desc_t;

/* --- Event descriptor --- */
typedef struct {
    uint32_t  stype;    /* ZE_STRUCTURE_TYPE_EVENT_DESC */
    void*     pNext;
    uint32_t  index;
    uint32_t  signal;
    uint32_t  wait;
} ze_event_desc_t;

/* --- Module descriptor --- */
typedef enum {
    ZE_MODULE_FORMAT_IL_SPIRV   = 0,
    ZE_MODULE_FORMAT_NATIVE     = 1,
} ze_module_format_t;

typedef struct {
    uint32_t             stype;     /* ZE_STRUCTURE_TYPE_MODULE_DESC */
    void*                pNext;
    ze_module_format_t   format;
    size_t               inputSize;
    const uint8_t*       pInputModule;
    const char*          pBuildFlags;
    void*                pConstants;
} ze_module_desc_t;

/* --- Kernel descriptor --- */
typedef struct {
    uint32_t     stype;    /* ZE_STRUCTURE_TYPE_KERNEL_DESC */
    void*        pNext;
    uint32_t     flags;
    const char*  pKernelName;
} ze_kernel_desc_t;

/* --- Memory allocation descriptors --- */
typedef struct {
    uint32_t  stype;    /* ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC */
    void*     pNext;
    uint32_t  flags;
    uint32_t  ordinal;
} ze_device_mem_alloc_desc_t;

typedef struct {
    uint32_t  stype;    /* ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC */
    void*     pNext;
    uint32_t  flags;
} ze_host_mem_alloc_desc_t;

/* --- Group count for kernel launch --- */
typedef struct {
    uint32_t groupCountX;
    uint32_t groupCountY;
    uint32_t groupCountZ;
} ze_group_count_t;


/* ====================================================================
 * Function Pointer Typedefs
 * ==================================================================== */

typedef ze_result_t (*pfn_zeInit)(ze_init_flags_t flags);
typedef ze_result_t (*pfn_zeDriverGet)(uint32_t *pCount, ze_driver_handle_t *phDrivers);
typedef ze_result_t (*pfn_zeDeviceGet)(ze_driver_handle_t hDriver, uint32_t *pCount, ze_device_handle_t *phDevices);
typedef ze_result_t (*pfn_zeDeviceGetProperties)(ze_device_handle_t hDevice, ze_device_properties_t *pDeviceProperties);
typedef ze_result_t (*pfn_zeContextCreate)(ze_driver_handle_t hDriver, const ze_context_desc_t *desc, ze_context_handle_t *phContext);
typedef ze_result_t (*pfn_zeContextDestroy)(ze_context_handle_t hContext);
typedef ze_result_t (*pfn_zeCommandQueueCreate)(ze_context_handle_t hContext, ze_device_handle_t hDevice, const ze_command_queue_desc_t *desc, ze_command_queue_handle_t *phCommandQueue);
typedef ze_result_t (*pfn_zeCommandQueueDestroy)(ze_command_queue_handle_t hCommandQueue);
typedef ze_result_t (*pfn_zeCommandQueueSynchronize)(ze_command_queue_handle_t hCommandQueue, uint64_t timeout);
typedef ze_result_t (*pfn_zeCommandListCreateImmediate)(ze_context_handle_t hContext, ze_device_handle_t hDevice, const ze_command_queue_desc_t *altdesc, ze_command_list_handle_t *phCommandList);
typedef ze_result_t (*pfn_zeCommandListDestroy)(ze_command_list_handle_t hCommandList);
typedef ze_result_t (*pfn_zeCommandListReset)(ze_command_list_handle_t hCommandList);
typedef ze_result_t (*pfn_zeCommandListAppendMemoryCopy)(ze_command_list_handle_t hCommandList, void *dstptr, const void *srcptr, size_t size, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);
typedef ze_result_t (*pfn_zeCommandListAppendBarrier)(ze_command_list_handle_t hCommandList, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);
typedef ze_result_t (*pfn_zeCommandListAppendLaunchKernel)(ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel, const ze_group_count_t *pLaunchFuncArgs, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);
typedef ze_result_t (*pfn_zeMemAllocShared)(ze_context_handle_t hContext, const ze_device_mem_alloc_desc_t *device_desc, const ze_host_mem_alloc_desc_t *host_desc, size_t size, size_t alignment, ze_device_handle_t hDevice, void **pptr);
typedef ze_result_t (*pfn_zeMemFree)(ze_context_handle_t hContext, void *ptr);
typedef ze_result_t (*pfn_zeEventPoolCreate)(ze_context_handle_t hContext, const ze_event_pool_desc_t *desc, uint32_t numDevices, ze_device_handle_t *phDevices, ze_event_pool_handle_t *phEventPool);
typedef ze_result_t (*pfn_zeEventPoolDestroy)(ze_event_pool_handle_t hEventPool);
typedef ze_result_t (*pfn_zeEventCreate)(ze_event_pool_handle_t hEventPool, const ze_event_desc_t *desc, ze_event_handle_t *phEvent);
typedef ze_result_t (*pfn_zeEventDestroy)(ze_event_handle_t hEvent);
typedef ze_result_t (*pfn_zeEventHostSynchronize)(ze_event_handle_t hEvent, uint64_t timeout);
typedef ze_result_t (*pfn_zeEventHostReset)(ze_event_handle_t hEvent);
typedef ze_result_t (*pfn_zeEventQueryStatus)(ze_event_handle_t hEvent);
typedef ze_result_t (*pfn_zeModuleCreate)(ze_context_handle_t hContext, ze_device_handle_t hDevice, const ze_module_desc_t *desc, ze_module_handle_t *phModule, void *phBuildLog);
typedef ze_result_t (*pfn_zeModuleDestroy)(ze_module_handle_t hModule);
typedef ze_result_t (*pfn_zeKernelCreate)(ze_module_handle_t hModule, const ze_kernel_desc_t *desc, ze_kernel_handle_t *phKernel);
typedef ze_result_t (*pfn_zeKernelDestroy)(ze_kernel_handle_t hKernel);
typedef ze_result_t (*pfn_zeKernelSetGroupSize)(ze_kernel_handle_t hKernel, uint32_t groupSizeX, uint32_t groupSizeY, uint32_t groupSizeZ);
typedef ze_result_t (*pfn_zeKernelSetArgumentValue)(ze_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize, const void *pArgValue);


/* ====================================================================
 * sp_l0_t — Dynamic loader + device state for one Intel UHD iGPU
 * ==================================================================== */

typedef struct {
    /* Dynamic library handle */
    void*   dll_handle;       /* HMODULE on Windows, dlopen handle on Linux */
    bool    loaded;           /* true if all function pointers resolved */

    /* --- Function pointer table --- */
    pfn_zeInit                          zeInit;
    pfn_zeDriverGet                     zeDriverGet;
    pfn_zeDeviceGet                     zeDeviceGet;
    pfn_zeDeviceGetProperties           zeDeviceGetProperties;
    pfn_zeContextCreate                 zeContextCreate;
    pfn_zeContextDestroy                zeContextDestroy;
    pfn_zeCommandQueueCreate            zeCommandQueueCreate;
    pfn_zeCommandQueueDestroy           zeCommandQueueDestroy;
    pfn_zeCommandQueueSynchronize       zeCommandQueueSynchronize;
    pfn_zeCommandListCreateImmediate    zeCommandListCreateImmediate;
    pfn_zeCommandListDestroy            zeCommandListDestroy;
    pfn_zeCommandListReset              zeCommandListReset;
    pfn_zeCommandListAppendMemoryCopy   zeCommandListAppendMemoryCopy;
    pfn_zeCommandListAppendBarrier      zeCommandListAppendBarrier;
    pfn_zeCommandListAppendLaunchKernel zeCommandListAppendLaunchKernel;
    pfn_zeMemAllocShared                zeMemAllocShared;
    pfn_zeMemFree                       zeMemFree;
    pfn_zeEventPoolCreate               zeEventPoolCreate;
    pfn_zeEventPoolDestroy              zeEventPoolDestroy;
    pfn_zeEventCreate                   zeEventCreate;
    pfn_zeEventDestroy                  zeEventDestroy;
    pfn_zeEventHostSynchronize          zeEventHostSynchronize;
    pfn_zeEventHostReset                zeEventHostReset;
    pfn_zeEventQueryStatus              zeEventQueryStatus;
    pfn_zeModuleCreate                  zeModuleCreate;
    pfn_zeModuleDestroy                 zeModuleDestroy;
    pfn_zeKernelCreate                  zeKernelCreate;
    pfn_zeKernelDestroy                 zeKernelDestroy;
    pfn_zeKernelSetGroupSize            zeKernelSetGroupSize;
    pfn_zeKernelSetArgumentValue        zeKernelSetArgumentValue;

    /* --- Discovered device state --- */
    ze_driver_handle_t          driver;
    ze_device_handle_t          device;       /* The Intel UHD iGPU */
    ze_context_handle_t         context;
    ze_command_queue_handle_t   cmd_queue;
    ze_command_list_handle_t    cmd_list;     /* Immediate command list */
    ze_event_pool_handle_t      event_pool;

    /* Device info */
    ze_device_properties_t      props;
    bool                        device_found; /* true if an integrated Intel GPU was found */

    /* USM allocations (tracked for cleanup) */
    void*   usm_allocs[16];
    int     n_usm_allocs;
} sp_l0_t;


/* ====================================================================
 * Public API
 * ==================================================================== */

/*
 * sp_l0_init — Load ze_loader.dll, discover Intel UHD iGPU, create
 * context + immediate command list.
 *
 * Returns 0 on success (UHD ready for dispatch).
 * Returns -1 if ze_loader.dll not found (no L0 runtime).
 * Returns -2 if no integrated Intel GPU found.
 * Returns -3 on L0 API error during init.
 *
 * On any failure, the context is safe to pass to sp_l0_free().
 */
int sp_l0_init(sp_l0_t *l0);

/*
 * sp_l0_free — Release all L0 resources and unload DLL.
 * Safe to call on partially-initialised context.
 */
void sp_l0_free(sp_l0_t *l0);

/*
 * sp_l0_alloc_shared — Allocate USM shared memory accessible from
 * both CPU and Intel UHD via Ring Bus.  The CPU can write directly
 * and the iGPU sees updates through LLC coherence.
 *
 * Returns pointer on success, NULL on failure.
 * The allocation is tracked and freed by sp_l0_free().
 */
void* sp_l0_alloc_shared(sp_l0_t *l0, size_t bytes, size_t alignment);

/*
 * sp_l0_free_mem — Free a specific USM allocation.
 */
void sp_l0_free_mem(sp_l0_t *l0, void *ptr);

/*
 * sp_l0_create_event — Create a host-visible event for sync.
 * Returns 0 on success, -1 on error.
 */
int sp_l0_create_event(sp_l0_t *l0, ze_event_handle_t *out_event, uint32_t index);

/*
 * sp_l0_sync_event — Wait for an event to be signaled (host-side).
 * timeout_ns: UINT64_MAX for infinite wait.
 * Returns 0 if signaled, 1 if timeout, -1 on error.
 */
int sp_l0_sync_event(sp_l0_t *l0, ze_event_handle_t event, uint64_t timeout_ns);

/*
 * sp_l0_query_event — Non-blocking event query.
 * Returns 0 if signaled, 1 if not ready, -1 on error.
 */
int sp_l0_query_event(sp_l0_t *l0, ze_event_handle_t event);

/*
 * sp_l0_reset_event — Reset event for reuse.
 */
void sp_l0_reset_event(sp_l0_t *l0, ze_event_handle_t event);

/*
 * sp_l0_reset_cmd_list — Reset the immediate command list (abort
 * in-flight work).  Cost: ~2µs on Ring Bus.
 */
int sp_l0_reset_cmd_list(sp_l0_t *l0);

/*
 * sp_l0_append_copy — Append a memory copy to the immediate cmd list.
 * For USM shared memory, this is often unnecessary (CPU writes are
 * LLC-coherent), but useful for device-local transfers.
 */
int sp_l0_append_copy(sp_l0_t *l0, void *dst, const void *src,
                      size_t bytes, ze_event_handle_t signal_event);

/*
 * sp_l0_append_barrier — Append an execution barrier.
 */
int sp_l0_append_barrier(sp_l0_t *l0, ze_event_handle_t signal_event);

/*
 * sp_l0_build_module_spirv — Build a SPIR-V module for the UHD.
 * spirv_data/spirv_size: pre-compiled SPIR-V binary.
 * Returns 0 on success, -1 on build failure.
 */
int sp_l0_build_module_spirv(sp_l0_t *l0, const uint8_t *spirv_data,
                             size_t spirv_size, ze_module_handle_t *out_module);

/*
 * sp_l0_create_kernel — Create a kernel from a built module.
 */
int sp_l0_create_kernel(sp_l0_t *l0, ze_module_handle_t module,
                        const char *kernel_name, ze_kernel_handle_t *out_kernel);

/*
 * sp_l0_launch_kernel — Append a kernel launch to the immediate cmd list.
 * The kernel's arguments must already be set via zeKernelSetArgumentValue.
 */
int sp_l0_launch_kernel(sp_l0_t *l0, ze_kernel_handle_t kernel,
                        uint32_t group_count_x, uint32_t group_count_y,
                        uint32_t group_count_z,
                        ze_event_handle_t signal_event);

#ifdef __cplusplus
}
#endif

#endif /* SP_LEVEL_ZERO_H */
