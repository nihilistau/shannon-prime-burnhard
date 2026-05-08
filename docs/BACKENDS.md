# Shannon-Prime Backend Reference

Shannon-Prime ships 9 backend implementations spanning phone to server hardware. Every backend implements the same VHT2 + banded quantization pipeline; they differ only in instruction set, memory model, and dispatch mechanism.

The core math is defined in `core/shannon_prime.h` (1133-line public API). Backends live under `backends/` and implement the transform, quantization, and cache operations using platform-native instructions.

---

## 1. CPU (C Reference)

**Location:** `core/shannon_prime.c`, `core/shannon_prime_sqfree.c`
**Target:** Any x86-64 or ARM64 CPU
**Status:** Production — all features implemented

The reference implementation. Portable C with no mandatory SIMD dependencies. Implements the complete feature set: ship path, sqfree + spinor, hierarchical Vilenkin predictor, variance-ranked calibration, partial band reads, disk serialization, Cauchy reset, and PrimePE.

All other backends are validated against CPU output. If the CPU backend produces different results from a GPU backend on the same input, the GPU backend has the bug.

**Performance:** On ARM Cortex-A78 (Snapdragon 8 Gen 1), the fused decompress-matmul path achieves 1.79× the throughput of vanilla fp32 matmul. The bandwidth crossover materializes before any DSP dispatch is attempted — the CPU alone benefits from compressed storage.

**Build:**
```bash
make                    # CPU-only, all features
make test-all           # 187/188 tests
```

---

## 2. CUDA

**Location:** `backends/cuda/`
**Target:** NVIDIA GPUs (Compute Capability 6.0+)
**Status:** Production — ship path + sqfree GPU-resident cache

The CUDA backend runs the entire VHT2 pipeline on-GPU. Compressed KV blocks live in VRAM; compress and decompress execute as CUDA kernels with no host round-trip on the write or read path. The compressed cache never leaves the GPU — no PCIe transfers needed for KV storage.

**Key files:**
- `shannon_prime_cuda.cu` — VHT2 forward kernel, band quantize/dequantize kernels
- `shannon_prime_sqfree.cu` — Sqfree path kernels (Knight gather, Möbius predict, residual quant)
- `shannon_prime_fp8.cu` — FP8 (E4M3FN) banded quantization (compile-time gated via `SP_FP8`)

**GPU-resident cache API:**
```cpp
auto cache = KvCache::create_gpu(n_layer, n_head_kv, head_dim, max_seq, cfg, stream);
cache->write_gpu(layer, pos_offset, n_tokens, d_K_flat, d_V_flat);
cache->read_gpu(layer, kv_len, d_K_out, d_V_out);
```

**Performance:** Ship GPU cache on Qwen3-8B Q8: 1m28s for full PPL evaluation (vs 23 min with host fallback — 15.6× faster).

**FP8 path:** V cache benefits from fp8's higher dynamic range on smooth distributions. Gated behind `SP_FP8=1` compile flag. FP4 (MXFP4) for Blackwell tensor cores requires `SP_FP4=1` + sm_120+ + CUDA 12.8.

**Build:**
```bash
make SP_WITH_CUDA=1
# Or via CMake
cmake -B build -DSP_WITH_CUDA=ON
```

**Multi-GPU:** Layer L → GPU[L × n_gpus / n_layer]. Cross-GPU copies inserted automatically. Validated on RTX 2060 + Intel UHD with cross-device correlation 1.0000.

---

## 3. Vulkan

**Location:** `backends/vulkan/`
**Target:** Any GPU with Vulkan 1.1+ compute support (NVIDIA, AMD, Intel, Qualcomm Adreno)
**Status:** Production

The Vulkan backend implements VHT2 via compute shaders. Cross-platform by design — the same shader code runs on discrete NVIDIA/AMD GPUs, Intel integrated graphics, and Qualcomm Adreno mobile GPUs.

**Capabilities:**
- VHT2 forward/inverse via compute shader
- Banded quantize/dequantize (ship path)
- Dual-GPU support (Beast Canyon: one CUDA device + one Vulkan device)

**Validated configurations:**
- RTX 2060 (Vulkan): K=0.9920, V=0.9730
- Intel UHD 770 (Vulkan): Identical fidelity to CUDA
- Qualcomm Adreno 730 (Vulkan compute): Mobile path

When no Vulkan device is available, transparent fallback to the C core implementation.

**Build:**
```bash
make SP_WITH_VULKAN=1
# Requires Vulkan SDK installed
```

---

## 4. Adreno (Mobile GPU)

**Location:** `backends/adreno/`
**Target:** Qualcomm Snapdragon 8 Gen 1+ (Adreno 730+)
**Status:** Production

The Adreno backend targets the Qualcomm Snapdragon compute stack via two paths: OpenCL 2.0 for older devices and Vulkan compute for newer ones. On Snapdragon SoCs, Shannon-Prime can use four of the five compute units: CPU (Cortex-A78/A510), GPU (Adreno), DSP (Hexagon), and NPU (HTP via QNN).

The Adreno GPU is particularly effective for the matmul phase of attention — the compressed K/V cache sits in shared GPU memory and the Adreno's tiled architecture maps well to the banded dequantize + dot-product pattern.

**Build:**
```bash
# Via Android NDK
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DSP_WITH_ADRENO=ON
```

---

## 5. Hexagon DSP

**Location:** `backends/hexagon/`
**Target:** Qualcomm Hexagon V69+ with HVX (Snapdragon 8 Gen 1+)
**Status:** Production

The Hexagon backend runs VHT2 compression on the Qualcomm DSP via FastRPC. The DSP operates on its own power/clock domain — it doesn't compete with the CPU for thermal budget. On the Samsung Galaxy S22 Ultra (primary validation device), the FastRPC dispatch ceiling is 577 calls/sec.

**Key insight:** 4-split (4 FastRPC calls per token) is the correct granularity. Per-layer (28 calls per token) is DOA at 38.7 t/s due to FastRPC overhead. HTP residency is an mmap problem, not a granularity problem.

**Architecture:**
- FastRPC for CPU ↔ DSP communication
- rpcmem for shared memory registration (the buffer must be rpcmem-allocated for zero-copy)
- HVX vector instructions for VHT2 butterfly (128-byte vector width)
- HTP V69 for neural-network-style tensor ops via QNN

**Validated result:** 81 µs MIN per AI-Hub .bin split on S22U V69 HTP, matching Qualcomm's lab numbers exactly.

**Build requirements:** Qualcomm Hexagon SDK 5.x with toolv87 and DSP_ARCH=v69.

---

## 6. QNN AI Hub

**Location:** `backends/qnn_aihub/`
**Target:** Qualcomm NPU/HTP via QNN runtime
**Status:** Production

The QNN backend executes pre-compiled `.bin` files on the Hexagon Tensor Processor. These .bins are exported via Qualcomm AI Hub's cloud compilation pipeline from ONNX models, then executed locally on-device via the QNN C API.

**Key discovery:** The `.bin` outputs use `QNN_DATATYPE_UFIXED_POINT_16` (dtype=1046), not fp16 (dtype=534). Both are uint16 in memory, but ufixed requires per-tensor scale+offset dequantization. This resolved the "NaN output" bug that blocked the pipeline for weeks — the .bins were always functional; we were misinterpreting the output type.

**Runtime graph construction:** `QnnGraph_addNode` + `APP_WRITE` weight injection builds MatMul graphs at runtime without the AOT .bin compilation step. Measured at 238 µs for 256×256 fp32 on V69 — this is the Mode C dispatch primitive.

**Integration:** `sp_qnn.c` loads 4 pre-compiled .bin splits, executes them in sequence (load splits 1+2 → exec → destroy → load splits 3+4 → exec → destroy), and chains residual outputs. Steady-state: 85 ms/split, projecting to 376 tok/sec for prefill.

---

## 7. Halide

**Location:** `backends/halide/`
**Target:** Cross-platform (CPU, GPU, DSP via Halide scheduling)
**Status:** Scaffold — build driver working, blocked on Qualcomm Compute add-on

The Halide backend uses the Halide scheduling language for auto-tuned VHT2 kernels. Halide separates the algorithm (what to compute) from the schedule (how to compute it), enabling automatic optimization for different hardware targets from a single source.

**Build driver:** `backends/halide/build-example.ps1` builds any Qualcomm Halide example on Windows.

**Status:** The build infrastructure works. The Halide kernels for VHT2 are written. Blocked on Qualcomm Compute add-on installation for Hexagon target scheduling.

---

## 8. Torch (PyTorch)

**Location:** `backends/torch/`
**Target:** Any device PyTorch supports (CPU, CUDA, MPS)
**Status:** Production

The PyTorch backend is the reference implementation for Python-based inference engines. It runs on any device PyTorch supports — CPU, CUDA, MPS — with no compiled extensions required. It is used by the ComfyUI integration and for notebook-based experimentation.

**Integration path:** The ComfyUI nodes import the torch backend directly. The VHT2 forward pass, Möbius reorder, and banded quantization are implemented as pure PyTorch operations (tensor math, no custom CUDA kernels). This makes it the easiest backend to test against and the fastest to iterate on.

**Performance:** Slower than the native C/CUDA/Vulkan backends due to Python overhead and PyTorch dispatch. Use for prototyping, validation, and Python-native workflows. For production inference, use the engine or the llama.cpp integration.

---

## 9. Beast Canyon (Heterogeneous Desktop)

**Location:** `backends/beast_canyon/`
**Target:** Desktop heterogeneous systems (Optane + AVX-512 + dual-GPU + optional phone sidecar)
**Status:** New — scaffold complete, compiles clean

Beast Canyon is the desktop counterpart to the phone-side Hexagon engine. It orchestrates inference across fundamentally different compute units:

**Components:**

| Module | File | Purpose |
|---|---|---|
| Optane Reservoir | `sp_optane.h/c` | DAX-mapped GGUF with expert pointer table. GGUF v2/v3 parser. |
| AVX-512 Shredder | `sp_avx512_shredder.h/c` | Dequant Q4_0/Q4_1/Q8_0/Q4_K/Q6_K → fp16. CPUID feature detection. |
| Hetero Sync | `sp_hetero_sync.h/c` | Cross-GPU barrier (CUDA events + Vulkan fences). Pre-shred callback. |
| Engine | `sp_beast_canyon.h/c` | Full orchestrator. 6-stage boot, MoE forward, ping-pong staging. |
| Diagnostics | `sp_diagnostics_bc.h/c` | Pulse monitor, Day Zero audit, topology report, ASCII dashboard. |
| Test Harness | `test_beast_canyon.c` | 5-test validation suite. `--audit-only` for Optane validation. |

**AVX-512 Shredder kernels:**
- **Q4_0:** Load fp16 scale → F16C broadcast → unpack nibbles → zero-extend → subtract zero-point → scale → fp16 output. One cache line (64 bytes) per iteration.
- **Q8_0:** Sign-extend int8→int32 → scale → fp16 output.
- **Q4_1:** FMA path: `_mm512_fmadd_ps(q*d + m)`, no zero-point subtraction.
- **Q4_K:** Scalar superblock (256 elements, 144 bytes) with 6-bit scale unpacking.
- **Q6_K:** Low 4-bit + high 2-bit reconstruction, 16 sub-blocks.
- **F16:** Prefetch in 4KB strides + memcpy passthrough.

All kernels have scalar fallback paths for non-AVX-512 CPUs.

**Build:**
```bash
cd backends/beast_canyon
cmake -B build -DSP_BEAST_STANDALONE=ON [-DSP_WITH_CUDA=ON] [-DSP_WITH_VULKAN=ON]
cmake --build build
./build/sp_beast_test /path/to/model.gguf [--audit-only]
```

**Graceful degradation:** Dual-GPU → Single GPU → CPU-only. Optane → filesystem mmap. Sidecar → local PrimePE. Hardware detection at boot.

---

## Backend Selection Guide

| Scenario | Recommended Backend | Why |
|---|---|---|
| Desktop NVIDIA GPU | CUDA | GPU-resident cache, no host round-trip |
| Desktop AMD/Intel GPU | Vulkan | Cross-vendor compute shaders |
| Desktop heterogeneous | Beast Canyon | Optane + AVX-512 + dual-GPU pipeline |
| Phone (Snapdragon 8 Gen 1+) | Hexagon + Adreno | DSP for compression, GPU for matmul |
| Phone (QNN pre-compiled) | QNN AI Hub | Pre-compiled .bins, maximum throughput |
| Python / ComfyUI | Torch | No compilation needed, PyTorch-native |
| Laptop (no discrete GPU) | CPU | Portable, all features, no dependencies |
| Research / prototyping | Torch or CPU | Easy to modify, validate against |
