# Shannon-Prime: Adreno / Mobile Backend

## Overview

The mobile backend targets the Qualcomm Snapdragon compute stack — all of it. Snapdragon SoCs have five compute units, and Shannon-Prime can use four of them:

```
Snapdragon 8 Gen 1 (Samsung S22 Ultra)
═══════════════════════════════════════

  ┌────────────────────────────────────────────────────────────┐
  │  CPU Cluster                                               │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
  │  │ A510 ×4  │  │ A710 ×3  │  │          │  │  X2 ×1   │  │
  │  │ silver   │  │  gold    │  │          │  │  prime   │  │
  │  │ 1.8 GHz  │  │ 2.5 GHz  │  │          │  │ 3.0 GHz  │  │
  │  │          │  │          │  │          │  │          │  │
  │  │ Background│  │ Parallel │  │          │  │ Decode   │  │
  │  │ work     │  │ layers   │  │          │  │ writeback│  │
  │  └──────────┘  └──────────┘  │          │  └──────────┘  │
  ├────────────────────────────────┤          ├────────────────┤
  │  Adreno 730 GPU               │          │ Hexagon V69    │
  │  8 SPs, 900 MHz               │          │ HVX 1024-bit   │
  │  Vulkan 1.1                   │          │ (no HMX)       │
  │                               │          │                │
  │  Batch VHT2 on prefill        │          │ VHT2 butterfly │
  │  (via Vulkan compute shaders) │          │ (future path)  │
  └───────────────────────────────┘          └────────────────┘
```

## NEON SIMD Tiers

The backend auto-detects ARM features at runtime and uses the highest available tier:

### Tier 1: `float32x4_t` (Universal ARMv8+)

4 float32 elements per NEON operation. Works on every ARMv8 device ever made.

VHT2 at p=2 butterfly (via `sp_neon_vht2_*`): `vaddq_f32` / `vsubq_f32` with 4 elements per cycle, each pass scaled by 1/√2 for self-inverse orthonormality. For hd=128: 32 iterations per butterfly pass, 7 passes = 224 NEON operations total.

Absmax reduction: `vabsq_f32` + `vmaxq_f32` with pairwise horizontal reduction via `vpmax_f32`.

### Tier 2: `float16x8_t` (ARMv8.2 FEAT_FP16)

8 float16 elements per NEON operation. Available on Cortex-A76 and all newer cores — including your S22 Ultra's X2 and A710 cores.

VHT2 at p=2 butterfly directly on `__fp16` arrays: `vaddq_f16` / `vsubq_f16`. **Halves memory bandwidth** and **doubles elements per cycle** compared to Tier 1. For hd=128 in fp16: 256 bytes total = two 128-bit NEON loads for the entire vector.

The `sp_adreno_write_k_f16()` path runs VHT2 entirely in fp16, converting to f32 only for the banded quantization step. When the model already runs in fp16 (common on mobile), this avoids any fp16→f32 conversion for the VHT2.

### Tier 3: dotprod / i8mm (ARMv8.2+ / ARMv8.6)

`sdot` (signed dot product) and `smmla` (signed matrix multiply-accumulate) for integer operations. Available on your S22 Ultra's X2 core. Infrastructure is in place for accelerating the quantize/dequantize inner loops — the scale-multiply-round-clamp step maps to integer dot products.

### Feature Detection

```c
sp_mobile_caps_t caps;
sp_mobile_detect_caps(&caps);
sp_mobile_print_caps(&caps);
```

Output on Snapdragon 8 Gen 1 would show:

```
[Shannon-Prime Mobile] Hardware capabilities:
  NEON:         yes
  FP16 arith:   yes (Tier 2)
  Dot product:  yes
  I8MM:         yes
  SVE/SVE2:     no/no
  CPU cores:    4 big + 4 little (prime=#7)
  Adreno:       yes (model 730)
  Hexagon:      yes (V69, HVX=1024-bit)
```

## big.LITTLE Thread Affinity

Snapdragon 8 Gen 1 has an asymmetric CPU:

| Cores | Type | Clock | Shannon-Prime Role |
|-------|------|-------|--------------------|
| 0–3 | Cortex-A510 (silver) | 1.8 GHz | Background cache maintenance |
| 4–6 | Cortex-A710 (gold) | 2.5 GHz | Parallel per-layer compression |
| 7 | Cortex-X2 (prime) | 3.0 GHz | Decode writeback (latency-critical) |

```c
sp_set_thread_affinity(SP_AFFINITY_PRIME, &caps);  // Pin to core 7
sp_set_thread_affinity(SP_AFFINITY_GOLD, &caps);   // Pin to cores 4-6
sp_set_thread_affinity(SP_AFFINITY_SILVER, &caps);  // Pin to cores 0-3
sp_set_thread_affinity(SP_AFFINITY_ANY, &caps);     // OS scheduler decides
```

Uses `sched_setaffinity()` on Linux/Android. Returns -1 on non-Linux platforms (non-fatal).

## Hexagon DSP

### What's Available on Each Snapdragon

| SoC | Hexagon | HVX | HMX | Shannon-Prime Path |
|-----|---------|-----|-----|-------------------|
| 8 Gen 1 (your S22) | V69 | 1024-bit | No | HVX VHT2 butterfly (stub `sp_hvx_vht2_*`) |
| 8 Gen 2 | V73 | 1024-bit | FP16 | HVX VHT2 + HMX matrix ops |
| 8 Gen 3 | V75 | 1024-bit | Improved | Full HVX/HMX acceleration |
| 8 Elite | V79 | 1024-bit | Enhanced | Best performance |

### Why HVX is a Perfect Fit for VHT2 at p=2

HVX registers are 1024 bits = 128 bytes. Consider head_dim=128:

- 128 × 4 bytes (f32) = 512 bytes = **half an HVX register**
- 128 × 2 bytes (f16) = 256 bytes = **quarter HVX register**

Two complete head vectors fit in one HVX register. The VHT2 at p=2 butterfly (add/sub on vector halves, then 1/√2 scale) maps directly to `Q6_V_vadd_VV` / `Q6_V_vsub_VV`, with `Q6_W_vshuff_VVR` for stride permutation between passes.

### Current Status

HVX function signatures and pseudocode are implemented. The actual HVX intrinsics require the Hexagon SDK (version 6.x) and signed skel libraries for non-rooted devices. Enable with `-DSP_HEXAGON_ENABLED` at compile time.

On your S22 Ultra without Hexagon SDK: the code compiles and runs the NEON Tier 2 path, which is already fast. Hexagon would primarily help by freeing the CPU cores for other work during VHT2 compression.

## Building for Android

Cross-compile with Android NDK:

```bash
# Set up NDK
export NDK=/path/to/android-ndk-r28b
export TOOLCHAIN=$NDK/toolchains/llvm/prebuilt/linux-x86_64

# Compile for arm64
$TOOLCHAIN/bin/aarch64-linux-android33-clang \
    -O2 -march=armv8.2-a+fp16+dotprod \
    -o build/test_adreno_arm64 \
    tests/test_adreno.c \
    backends/adreno/shannon_prime_adreno.c \
    core/shannon_prime.c \
    -lm

# Push and run on device
adb push build/test_adreno_arm64 /data/local/tmp/
adb shell /data/local/tmp/test_adreno_arm64
```

The `-march=armv8.2-a+fp16+dotprod` flag enables Tier 2 and Tier 3 NEON paths. The compiler will use `vaddq_f16`, `vsubq_f16`, `sdot`, etc.

For Hexagon:

```bash
# Requires Hexagon SDK 6.x
source $HEXAGON_SDK_ROOT/setup_sdk_env.sh

$TOOLCHAIN/bin/aarch64-linux-android33-clang \
    -O2 -march=armv8.2-a+fp16+dotprod \
    -DSP_HEXAGON_ENABLED \
    -I$HEXAGON_SDK_ROOT/incs \
    -o build/test_adreno_hvx \
    tests/test_adreno.c \
    backends/adreno/shannon_prime_adreno.c \
    core/shannon_prime.c \
    -lm -lcdsprpc
```

## Performance Expectations

On Samsung S22 Ultra (Snapdragon 8 Gen 1):

| Operation | Estimated (NEON Tier 2) | Paper (measured) |
|-----------|------------------------|------------------|
| VHT2 forward (hd=128, fp16) | ~2 μs | — |
| VHT2 forward (hd=128, f32) | ~4 μs | — |
| Full writeback (16L × 4H) | ~1 ms | 37-42 ms* |
| K correlation | 0.9972 | 0.9972 |

*Paper measurement includes Vulkan dispatch overhead for the full pipeline. NEON-only writeback is much faster.
