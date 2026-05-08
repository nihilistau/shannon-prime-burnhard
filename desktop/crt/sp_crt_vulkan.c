// Shannon-Prime CRT: Vulkan compute dispatch for Ring M2 — Implementation
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
// Licensed under AGPLv3.
//
// Full Vulkan compute pipeline for modular matmul on Intel UHD (or any Vulkan GPU).
// The SPIR-V shader is embedded at compile time from sp_crt_matmul_mod_spv.h.

#include "sp_crt_vulkan.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Embedded SPIR-V bytecode
#include "sp_crt_matmul_mod_spv.h"

#define TILE_DIM 16
#define MAX_BUFS 8

// ============================================================================
// Internal context structure
// ============================================================================

typedef struct {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    size_t         size;
    int            in_use;
} sp_vk_buf_t;

struct sp_crt_vulkan_ctx {
    VkInstance       instance;        // Borrowed from sp_hetero_sync (not owned)
    VkPhysicalDevice phys_device;
    VkDevice         device;
    VkQueue          compute_queue;
    uint32_t         queue_family;

    // Pipeline
    VkShaderModule       shader;
    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout     pipe_layout;
    VkPipeline           pipeline;

    // Descriptors
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet  descriptor_set;

    // Command buffer
    VkCommandPool   cmd_pool;
    VkCommandBuffer cmd_buf;
    VkFence         fence;

    // Device buffers (A, B, C for the matmul + staging)
    sp_vk_buf_t device_bufs[MAX_BUFS];
    int n_bufs;

    // Staging buffer (host-visible, for H2D and D2H)
    VkBuffer        staging_buf;
    VkDeviceMemory  staging_mem;
    void           *staging_mapped;
    size_t          staging_size;

    // Memory type indices
    uint32_t mem_type_device;   // DEVICE_LOCAL
    uint32_t mem_type_host;     // HOST_VISIBLE | HOST_COHERENT

    int max_elements;
};

// ============================================================================
// Helpers
// ============================================================================

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_filter,
                                  VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

static int find_compute_queue_family(VkPhysicalDevice phys) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, NULL);
    VkQueueFamilyProperties *families = (VkQueueFamilyProperties*)malloc(count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, families);

    int best = -1;
    for (uint32_t i = 0; i < count; i++) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            best = (int)i;
            // Prefer compute-only (not graphics) for async compute
            if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                free(families);
                return (int)i;
            }
        }
    }
    free(families);
    return best;
}

// ============================================================================
// Push constant layout — must match the shader
// ============================================================================

typedef struct {
    int      M;
    int      N;
    int      K;
    uint32_t modulus;
} sp_vk_push_constants_t;

// ============================================================================
// Init
// ============================================================================

sp_crt_vulkan_ctx_t* sp_crt_vulkan_init(void *vk_instance,
                                         int physical_device_index,
                                         int max_elements) {
    sp_crt_vulkan_ctx_t *ctx = (sp_crt_vulkan_ctx_t*)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    ctx->instance = (VkInstance)vk_instance;
    ctx->max_elements = max_elements;

    // --- Get physical device ---
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &dev_count, NULL);
    VkPhysicalDevice *devs = (VkPhysicalDevice*)malloc(dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx->instance, &dev_count, devs);

    if (physical_device_index < 0 || physical_device_index >= (int)dev_count) {
        fprintf(stderr, "[sp-crt-vk] Invalid physical device index %d (have %u)\n",
                physical_device_index, dev_count);
        free(devs); free(ctx);
        return NULL;
    }
    ctx->phys_device = devs[physical_device_index];
    free(devs);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx->phys_device, &props);
    fprintf(stderr, "[sp-crt-vk] Using: %s (vendor 0x%04x)\n",
            props.deviceName, props.vendorID);

    // --- Find compute queue family ---
    int qf = find_compute_queue_family(ctx->phys_device);
    if (qf < 0) {
        fprintf(stderr, "[sp-crt-vk] No compute queue family found\n");
        free(ctx);
        return NULL;
    }
    ctx->queue_family = (uint32_t)qf;

    // --- Create logical device ---
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci = {0};
    queue_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = ctx->queue_family;
    queue_ci.queueCount = 1;
    queue_ci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dev_ci = {0};
    dev_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_ci.queueCreateInfoCount = 1;
    dev_ci.pQueueCreateInfos = &queue_ci;

    VkResult vr = vkCreateDevice(ctx->phys_device, &dev_ci, NULL, &ctx->device);
    if (vr != VK_SUCCESS) {
        fprintf(stderr, "[sp-crt-vk] vkCreateDevice failed: %d\n", (int)vr);
        free(ctx);
        return NULL;
    }
    vkGetDeviceQueue(ctx->device, ctx->queue_family, 0, &ctx->compute_queue);

    // --- Find memory types ---
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(ctx->phys_device, &mem_props);

    // Device-local (for compute buffers)
    ctx->mem_type_device = find_memory_type(ctx->phys_device, 0xFFFFFFFF,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // Host-visible + coherent (for staging)
    ctx->mem_type_host = find_memory_type(ctx->phys_device, 0xFFFFFFFF,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (ctx->mem_type_device == UINT32_MAX || ctx->mem_type_host == UINT32_MAX) {
        fprintf(stderr, "[sp-crt-vk] Could not find required memory types\n");
        vkDestroyDevice(ctx->device, NULL);
        free(ctx);
        return NULL;
    }

    // --- Create shader module from embedded SPIR-V ---
    VkShaderModuleCreateInfo shader_ci = {0};
    shader_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_ci.codeSize = sp_crt_matmul_mod_spv_size;
    shader_ci.pCode = (const uint32_t*)sp_crt_matmul_mod_spv;

    vr = vkCreateShaderModule(ctx->device, &shader_ci, NULL, &ctx->shader);
    if (vr != VK_SUCCESS) {
        fprintf(stderr, "[sp-crt-vk] vkCreateShaderModule failed: %d\n", (int)vr);
        vkDestroyDevice(ctx->device, NULL);
        free(ctx);
        return NULL;
    }

    // --- Descriptor set layout: 3 storage buffers ---
    VkDescriptorSetLayoutBinding bindings[3];
    for (int i = 0; i < 3; i++) {
        memset(&bindings[i], 0, sizeof(bindings[i]));
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo desc_layout_ci = {0};
    desc_layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    desc_layout_ci.bindingCount = 3;
    desc_layout_ci.pBindings = bindings;
    vr = vkCreateDescriptorSetLayout(ctx->device, &desc_layout_ci, NULL, &ctx->desc_layout);
    if (vr != VK_SUCCESS) goto fail_shader;

    // --- Pipeline layout with push constants ---
    VkPushConstantRange push_range = {0};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(sp_vk_push_constants_t);

    VkPipelineLayoutCreateInfo pipe_layout_ci = {0};
    pipe_layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipe_layout_ci.setLayoutCount = 1;
    pipe_layout_ci.pSetLayouts = &ctx->desc_layout;
    pipe_layout_ci.pushConstantRangeCount = 1;
    pipe_layout_ci.pPushConstantRanges = &push_range;
    vr = vkCreatePipelineLayout(ctx->device, &pipe_layout_ci, NULL, &ctx->pipe_layout);
    if (vr != VK_SUCCESS) goto fail_desc;

    // --- Compute pipeline ---
    VkComputePipelineCreateInfo pipe_ci = {0};
    pipe_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipe_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipe_ci.stage.module = ctx->shader;
    pipe_ci.stage.pName = "main";
    pipe_ci.layout = ctx->pipe_layout;
    vr = vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipe_ci, NULL, &ctx->pipeline);
    if (vr != VK_SUCCESS) goto fail_pipe_layout;

    // --- Descriptor pool ---
    VkDescriptorPoolSize pool_size = {0};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 3;

    VkDescriptorPoolCreateInfo pool_ci = {0};
    pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets = 1;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes = &pool_size;
    vr = vkCreateDescriptorPool(ctx->device, &pool_ci, NULL, &ctx->descriptor_pool);
    if (vr != VK_SUCCESS) goto fail_pipeline;

    // --- Allocate descriptor set ---
    VkDescriptorSetAllocateInfo desc_alloc = {0};
    desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    desc_alloc.descriptorPool = ctx->descriptor_pool;
    desc_alloc.descriptorSetCount = 1;
    desc_alloc.pSetLayouts = &ctx->desc_layout;
    vr = vkAllocateDescriptorSets(ctx->device, &desc_alloc, &ctx->descriptor_set);
    if (vr != VK_SUCCESS) goto fail_pool;

    // --- Command pool + buffer ---
    VkCommandPoolCreateInfo cmd_pool_ci = {0};
    cmd_pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_ci.queueFamilyIndex = ctx->queue_family;
    cmd_pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vr = vkCreateCommandPool(ctx->device, &cmd_pool_ci, NULL, &ctx->cmd_pool);
    if (vr != VK_SUCCESS) goto fail_pool;

    VkCommandBufferAllocateInfo cmd_alloc = {0};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = ctx->cmd_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(ctx->device, &cmd_alloc, &ctx->cmd_buf);

    // --- Fence ---
    VkFenceCreateInfo fence_ci = {0};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vr = vkCreateFence(ctx->device, &fence_ci, NULL, &ctx->fence);
    if (vr != VK_SUCCESS) goto fail_cmd;

    // --- Staging buffer (max_elements * sizeof(uint32_t)) ---
    ctx->staging_size = (size_t)max_elements * sizeof(uint32_t);

    VkBufferCreateInfo staging_buf_ci = {0};
    staging_buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    staging_buf_ci.size = ctx->staging_size;
    staging_buf_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    staging_buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vr = vkCreateBuffer(ctx->device, &staging_buf_ci, NULL, &ctx->staging_buf);
    if (vr != VK_SUCCESS) goto fail_fence;

    VkMemoryRequirements staging_reqs;
    vkGetBufferMemoryRequirements(ctx->device, ctx->staging_buf, &staging_reqs);

    uint32_t staging_mem_type = find_memory_type(ctx->phys_device, staging_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkMemoryAllocateInfo staging_alloc_ci = {0};
    staging_alloc_ci.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    staging_alloc_ci.allocationSize = staging_reqs.size;
    staging_alloc_ci.memoryTypeIndex = staging_mem_type;
    vr = vkAllocateMemory(ctx->device, &staging_alloc_ci, NULL, &ctx->staging_mem);
    if (vr != VK_SUCCESS) goto fail_staging_buf;

    vkBindBufferMemory(ctx->device, ctx->staging_buf, ctx->staging_mem, 0);
    vkMapMemory(ctx->device, ctx->staging_mem, 0, ctx->staging_size, 0, &ctx->staging_mapped);

    fprintf(stderr, "[sp-crt-vk] Vulkan compute ready: pipeline + %zu KB staging\n",
            ctx->staging_size / 1024);
    return ctx;

    // --- Cleanup on failure ---
fail_staging_buf:
    vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
fail_fence:
    vkDestroyFence(ctx->device, ctx->fence, NULL);
fail_cmd:
    vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
fail_pool:
    vkDestroyDescriptorPool(ctx->device, ctx->descriptor_pool, NULL);
fail_pipeline:
    vkDestroyPipeline(ctx->device, ctx->pipeline, NULL);
fail_pipe_layout:
    vkDestroyPipelineLayout(ctx->device, ctx->pipe_layout, NULL);
fail_desc:
    vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layout, NULL);
fail_shader:
    vkDestroyShaderModule(ctx->device, ctx->shader, NULL);
    vkDestroyDevice(ctx->device, NULL);
    free(ctx);
    return NULL;
}

void sp_crt_vulkan_free(sp_crt_vulkan_ctx_t *ctx) {
    if (!ctx) return;

    vkDeviceWaitIdle(ctx->device);

    // Free device buffers
    for (int i = 0; i < ctx->n_bufs; i++) {
        if (ctx->device_bufs[i].in_use) {
            vkDestroyBuffer(ctx->device, ctx->device_bufs[i].buffer, NULL);
            vkFreeMemory(ctx->device, ctx->device_bufs[i].memory, NULL);
        }
    }

    // Staging
    vkUnmapMemory(ctx->device, ctx->staging_mem);
    vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
    vkFreeMemory(ctx->device, ctx->staging_mem, NULL);

    // Pipeline
    vkDestroyFence(ctx->device, ctx->fence, NULL);
    vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
    vkDestroyDescriptorPool(ctx->device, ctx->descriptor_pool, NULL);
    vkDestroyPipeline(ctx->device, ctx->pipeline, NULL);
    vkDestroyPipelineLayout(ctx->device, ctx->pipe_layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layout, NULL);
    vkDestroyShaderModule(ctx->device, ctx->shader, NULL);

    // Device (we own it)
    vkDestroyDevice(ctx->device, NULL);
    // Instance is borrowed — don't destroy

    free(ctx);
}

// ============================================================================
// Buffer allocation
// ============================================================================

uint32_t* sp_crt_vulkan_alloc(sp_crt_vulkan_ctx_t *ctx, size_t bytes) {
    if (!ctx || ctx->n_bufs >= MAX_BUFS) return NULL;

    int idx = ctx->n_bufs;
    sp_vk_buf_t *b = &ctx->device_bufs[idx];

    VkBufferCreateInfo buf_ci = {0};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = bytes;
    buf_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult vr = vkCreateBuffer(ctx->device, &buf_ci, NULL, &b->buffer);
    if (vr != VK_SUCCESS) return NULL;

    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(ctx->device, b->buffer, &reqs);

    uint32_t mem_type = find_memory_type(ctx->phys_device, reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mem_type == UINT32_MAX) {
        // Fallback: try host-visible (iGPU shares system memory)
        mem_type = find_memory_type(ctx->phys_device, reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    VkMemoryAllocateInfo alloc_ci = {0};
    alloc_ci.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_ci.allocationSize = reqs.size;
    alloc_ci.memoryTypeIndex = mem_type;

    vr = vkAllocateMemory(ctx->device, &alloc_ci, NULL, &b->memory);
    if (vr != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, b->buffer, NULL);
        return NULL;
    }

    vkBindBufferMemory(ctx->device, b->buffer, b->memory, 0);
    b->size = bytes;
    b->in_use = 1;
    ctx->n_bufs++;

    // Return the index encoded as a pointer (we decode it in dispatch)
    // This is a sentinel — the actual VkBuffer is in device_bufs[idx].
    return (uint32_t*)(uintptr_t)(idx + 1);
}

void sp_crt_vulkan_buf_free(sp_crt_vulkan_ctx_t *ctx, uint32_t *buf) {
    if (!ctx || !buf) return;
    int idx = (int)((uintptr_t)buf) - 1;
    if (idx < 0 || idx >= ctx->n_bufs) return;

    sp_vk_buf_t *b = &ctx->device_bufs[idx];
    if (b->in_use) {
        vkDestroyBuffer(ctx->device, b->buffer, NULL);
        vkFreeMemory(ctx->device, b->memory, NULL);
        b->in_use = 0;
    }
}

// Get the actual VkBuffer from our encoded pointer
static VkBuffer get_vk_buffer(sp_crt_vulkan_ctx_t *ctx, const uint32_t *encoded) {
    int idx = (int)((uintptr_t)encoded) - 1;
    if (idx < 0 || idx >= ctx->n_bufs) return VK_NULL_HANDLE;
    return ctx->device_bufs[idx].buffer;
}

static size_t get_vk_buffer_size(sp_crt_vulkan_ctx_t *ctx, const uint32_t *encoded) {
    int idx = (int)((uintptr_t)encoded) - 1;
    if (idx < 0 || idx >= ctx->n_bufs) return 0;
    return ctx->device_bufs[idx].size;
}

// ============================================================================
// Memory transfers via staging buffer
// ============================================================================

static int submit_and_wait(sp_crt_vulkan_ctx_t *ctx) {
    vkEndCommandBuffer(ctx->cmd_buf);

    VkSubmitInfo submit = {0};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &ctx->cmd_buf;

    vkResetFences(ctx->device, 1, &ctx->fence);
    VkResult vr = vkQueueSubmit(ctx->compute_queue, 1, &submit, ctx->fence);
    if (vr != VK_SUCCESS) return -1;

    vr = vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, 5000000000ULL); // 5s
    return (vr == VK_SUCCESS) ? 0 : -1;
}

int sp_crt_vulkan_memcpy_h2d(sp_crt_vulkan_ctx_t *ctx,
                              uint32_t *dst, const uint32_t *src,
                              size_t bytes) {
    if (!ctx || bytes > ctx->staging_size) return -1;

    VkBuffer dst_buf = get_vk_buffer(ctx, dst);
    if (dst_buf == VK_NULL_HANDLE) return -1;

    // Copy to staging
    memcpy(ctx->staging_mapped, src, bytes);

    // Record copy command
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(ctx->cmd_buf, 0);
    vkBeginCommandBuffer(ctx->cmd_buf, &begin);

    VkBufferCopy region = {0};
    region.size = bytes;
    vkCmdCopyBuffer(ctx->cmd_buf, ctx->staging_buf, dst_buf, 1, &region);

    return submit_and_wait(ctx);
}

int sp_crt_vulkan_memcpy_d2h(sp_crt_vulkan_ctx_t *ctx,
                              uint32_t *dst, const uint32_t *src,
                              size_t bytes) {
    if (!ctx || bytes > ctx->staging_size) return -1;

    VkBuffer src_buf = get_vk_buffer(ctx, src);
    if (src_buf == VK_NULL_HANDLE) return -1;

    // Record copy command
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(ctx->cmd_buf, 0);
    vkBeginCommandBuffer(ctx->cmd_buf, &begin);

    VkBufferCopy region = {0};
    region.size = bytes;
    vkCmdCopyBuffer(ctx->cmd_buf, src_buf, ctx->staging_buf, 1, &region);

    int rc = submit_and_wait(ctx);
    if (rc != 0) return rc;

    // Copy from staging to host
    memcpy(dst, ctx->staging_mapped, bytes);
    return 0;
}

// ============================================================================
// Compute dispatch
// ============================================================================

int sp_crt_vulkan_matmul_mod_dispatch(sp_crt_vulkan_ctx_t *ctx,
                                      const uint32_t *d_A,
                                      const uint32_t *d_B,
                                      uint32_t *d_C,
                                      int M, int N, int K,
                                      uint32_t modulus) {
    if (!ctx) return -1;

    VkBuffer buf_a = get_vk_buffer(ctx, d_A);
    VkBuffer buf_b = get_vk_buffer(ctx, d_B);
    VkBuffer buf_c = get_vk_buffer(ctx, d_C);
    if (buf_a == VK_NULL_HANDLE || buf_b == VK_NULL_HANDLE || buf_c == VK_NULL_HANDLE)
        return -1;

    // Update descriptor set to point at our buffers
    VkDescriptorBufferInfo buf_infos[3];
    buf_infos[0].buffer = buf_a;
    buf_infos[0].offset = 0;
    buf_infos[0].range  = (VkDeviceSize)M * K * sizeof(uint32_t);

    buf_infos[1].buffer = buf_b;
    buf_infos[1].offset = 0;
    buf_infos[1].range  = (VkDeviceSize)K * N * sizeof(uint32_t);

    buf_infos[2].buffer = buf_c;
    buf_infos[2].offset = 0;
    buf_infos[2].range  = (VkDeviceSize)M * N * sizeof(uint32_t);

    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        memset(&writes[i], 0, sizeof(writes[i]));
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ctx->descriptor_set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);

    // Record compute dispatch
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(ctx->cmd_buf, 0);
    vkBeginCommandBuffer(ctx->cmd_buf, &begin);

    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipe_layout, 0, 1, &ctx->descriptor_set, 0, NULL);

    // Push constants
    sp_vk_push_constants_t pc;
    pc.M = M;
    pc.N = N;
    pc.K = K;
    pc.modulus = modulus;
    vkCmdPushConstants(ctx->cmd_buf, ctx->pipe_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    // Dispatch: workgroup grid covers [N/TILE, M/TILE, 1]
    uint32_t gx = ((uint32_t)N + TILE_DIM - 1) / TILE_DIM;
    uint32_t gy = ((uint32_t)M + TILE_DIM - 1) / TILE_DIM;
    vkCmdDispatch(ctx->cmd_buf, gx, gy, 1);

    return submit_and_wait(ctx);
}

int sp_crt_vulkan_sync(sp_crt_vulkan_ctx_t *ctx) {
    if (!ctx) return -1;
    vkDeviceWaitIdle(ctx->device);
    return 0;
}

void* sp_crt_vulkan_get_queue(sp_crt_vulkan_ctx_t *ctx) {
    return ctx ? (void*)ctx->compute_queue : NULL;
}

void* sp_crt_vulkan_get_fence(sp_crt_vulkan_ctx_t *ctx) {
    return ctx ? (void*)ctx->fence : NULL;
}
