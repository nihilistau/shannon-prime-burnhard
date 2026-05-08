# Shannon-Prime: Vulkan Backend

## Overview

The Vulkan backend runs VHT2 compression via compute shaders, targeting cross-platform GPU acceleration. It works on NVIDIA, AMD, Intel, and Qualcomm Adreno GPUs — any device with Vulkan 1.1+ compute support. When no Vulkan device is available, it falls back to the C core implementation transparently.

**Current runtime status (post-v1.04):** The default shadow-cache path stays on the CPU staged VHT2 inside `vk_compress_one` / `vk_decompress_one`. Two blockers keep the GPU path off by default:

1. **vilenkin.comp semantics drift.** The v1.02 `VK_ERROR_DEVICE_LOST` hang on RTX 2060 no longer reproduces (v1.03 validation pass), but `SHANNON_PRIME_VULKAN_FORCE_GPU=1` produces K correlation ≈ 0.62 vs the CPU reference — tracked at `logs/vulkan_diagnostic_v1.03.json`. Needs RenderDoc + per-stage parity.
2. **Shipped-shader domain is narrow.** `band_{quantize,dequantize}.comp` only expose `band_bits_0..band_bits_3` in push constants and iterate `b * band_size + i` with no last-band remainder, and `mobius_reorder.comp` uses `local_size_x = 128`. So the GPU path only matches the CPU reference when `n_bands ≤ 4`, `head_dim % n_bands == 0`, and `head_dim ≤ 128`. The v1.03 10-band sqfree configs, pad_dim=154, and hd=256 models fall outside. `vk_gpu_dispatch_is_supported()` enforces this at runtime — `FORCE_GPU=1` on an out-of-domain config logs a one-shot fallback notice and falls through to CPU instead of silently corrupting output.

Pipelines are still created so `test_vulkan` exercises the full init path and the GPU path can be hand-driven via `SHANNON_PRIME_VULKAN_FORCE_GPU=1` for whatever shader-debugging session comes next. The wording below describes the shader pipeline as it is wired up; the domain limits are called out on each shader bullet.

## Compute Shaders

The following GLSL compute shaders live in `backends/vulkan/shaders/`:

**`vilenkin.comp`:** In-place VHT2 butterfly (self-inverse, 1/√p per stage, no 1/N on reconstruction). One workgroup per vector, shared memory for butterfly passes. For hd=128 at p=2: 7 barrier-synchronized passes in shared memory, then writeback. The legacy `wht.comp` shader has been removed; `vilenkin.comp` is the single transform. **Domain:** output parity with the CPU reference is not yet achieved (K corr ≈ 0.62); treat as under-repair.

**`mobius_reorder.comp`:** Gather-based Möbius permutation. Each thread reads from `order[tid]` and writes to position `tid`. The permutation table is a storage buffer bound at descriptor set 0, binding 2. **Domain:** `local_size_x = 128`, so only `head_dim ≤ 128` is written correctly; hd=256 would leave positions 128..255 uninitialised. A strided loop (or `local_size_x = 256`) is needed before we ship hd=256.

**`band_quantize.comp`:** Per-band abs-max reduction in shared memory, then sequential bit packing by thread 0. Push constants carry the band configuration. **Domain:** push constants expose only `band_bits_0..band_bits_3` and the kernel indexes `b * band_size + i` with no remainder absorption, so shader-correct output requires `n_bands ≤ 4` and `head_dim % n_bands == 0`. The GLSL `bit_buffer >>= 32u` flush is also UB on exact-32-bit alignment — deferred along with the push-constant widening.

**`band_dequantize.comp`:** Inverse of `band_quantize.comp`. Reads the packed payload + per-band fp16 scale, unpacks to signed integers, and scales back to fp32. Mirrors the CPU core's non-finite-scale guard: if an fp16 scale round-trips to Inf/NaN the whole band decodes as zero instead of poisoning the inverse VHT2. Same domain caveats as `band_quantize.comp`.

**`knight_predict.comp`:** Sqfree+spinor Knight CSR predict shader used by the aggressive path. Currently CPU-staged until the vilenkin parity issue above is resolved.

## Pipeline

```
Vulkan Compute Pipeline
════════════════════════

  Input Buffer (raw K/V, fp32)
       │
       ▼
  ┌─────────────┐    Dispatch 1: vilenkin.comp
  │   VHT2      │    workgroup = (head_dim,), groups = n_vectors
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐    Dispatch 2: mobius_reorder.comp
  │   Möbius     │    (K only — V skips this step)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐    Dispatch 3: band_quantize.comp
  │   Quantize   │    Output to compressed cache buffer
  └──────┬──────┘
         │
         ▼
  Compressed Cache Buffer
```

Read path reverses: dequantize → unreorder → VHT2 (reusing the same `vilenkin.comp` shader since VHT2 is self-inverse; no separate inverse shader, no 1/N on reconstruction).

## Building

Compile shaders to SPIR-V:

```bash
glslangValidator -V backends/vulkan/shaders/vilenkin.comp -o backends/vulkan/shaders/vilenkin.spv
glslangValidator -V backends/vulkan/shaders/mobius_reorder.comp -o backends/vulkan/shaders/mobius_reorder.spv
glslangValidator -V backends/vulkan/shaders/band_quantize.comp -o backends/vulkan/shaders/band_quantize.spv
glslangValidator -V backends/vulkan/shaders/band_dequantize.comp -o backends/vulkan/shaders/band_dequantize.spv
glslangValidator -V backends/vulkan/shaders/knight_predict.comp -o backends/vulkan/shaders/knight_predict.spv
```

Compile the host code (requires Vulkan SDK for full GPU path):

```bash
gcc -O2 -DVK_USE_PLATFORM -o build/test_vulkan \
    tests/test_vulkan.c \
    backends/vulkan/shannon_prime_vulkan.c \
    core/shannon_prime.c \
    -lvulkan -lm
```

Without Vulkan SDK, the host code compiles with CPU fallback (all 4 tests pass):

```bash
gcc -O2 -o build/test_vulkan \
    tests/test_vulkan.c \
    backends/vulkan/shannon_prime_vulkan.c \
    core/shannon_prime.c \
    -lm
```

## Integration Modes

**Standalone:** Creates its own VkDevice and VkQueue. For testing and benchmarking.

```c
sp_vulkan_cache_t *cc;
sp_vulkan_cache_init(&cc, &cfg, max_seq_len, NULL, NULL, 0);  // gpu_index 0
```

**Multi-GPU (standalone):** Open a specific physical device by index. Enumerate with `vulkaninfo --summary` to see available GPUs.

```c
sp_vulkan_cache_t *hot, *cold;
sp_vulkan_cache_init(&hot,  &cfg, max_seq_len, NULL, NULL, 0);  // discrete GPU
sp_vulkan_cache_init(&cold, &cfg, max_seq_len, NULL, NULL, 1);  // iGPU / second GPU
```

**Shared:** Uses the existing Vulkan device from llama.cpp's Vulkan backend or another inference engine. Zero overhead from device creation. `gpu_index` is ignored when a device is provided.

```c
sp_vulkan_cache_init(&cc, &cfg, max_seq_len, existing_vk_device, existing_vk_queue, 0);
```

## Buffer API

For zero-copy integration when K/V are already in Vulkan buffers:

```c
// Write from existing VkBuffer (no CPU→GPU transfer)
sp_vulkan_write_k_buffer(cc, layer, head, pos, vk_buffer, byte_offset);

// Read into existing VkBuffer (stays on GPU)
sp_vulkan_read_k_buffer(cc, layer, head, pos, vk_buffer, byte_offset);
```

## Adreno-Specific Notes

On Qualcomm Adreno GPUs (Snapdragon 8 Gen 1 / Adreno 730):

- Shared memory per workgroup: 32 KB (sufficient for hd=256)
- Max workgroup size: 1024 threads (sufficient for any head_dim)
- Wave64 execution: 64 threads per warp (vs 32 on NVIDIA)
- OpenCL is also available but Vulkan compute is preferred for llama.cpp integration

## Environment variables (diagnostic / debug)

| Variable | Default | Description |
|----------|---------|-------------|
| `SHANNON_PRIME_VULKAN_VALIDATE` | 0 | Set to 1 to load `VK_LAYER_KHRONOS_validation` when the instance is created. Without a `VK_EXT_debug_utils` callback the layer's messages go to the debug channel — pipe through `vulkaninfo --debug-utils` or attach RenderDoc to actually see them. |
| `SHANNON_PRIME_VULKAN_FORCE_GPU` | 0 | Set to 1 to route `vk_compress_one` / `vk_decompress_one` through the GPU pipeline instead of the CPU staged fallback. `vk_gpu_dispatch_is_supported()` gates this on (`n_bands ≤ 4`, `head_dim % n_bands == 0`, `head_dim ≤ 128`); configs outside the supported domain log a one-shot warning and take the CPU path. Expect K corr ≈ 0.62 vs CPU today — use for shader parity debugging, not production. |
| `SHANNON_PRIME_VULKAN_DEBUG` | 0 | Prints per-dispatch push-constant values once at init — useful for diffing against the CPU reference during shader bring-up. |

The Vulkan backend is the GPU path for mobile inference. The Adreno backend handles the CPU/NEON writeback path for when the GPU is busy with inference.
