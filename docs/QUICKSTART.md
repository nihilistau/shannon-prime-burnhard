# Shannon-Prime Quick Start

Get Shannon-Prime running on your hardware. This guide covers building from source, first-run validation, configuration, and common pitfalls.

---

## Prerequisites

**All platforms:**
- C11-compliant compiler (GCC 9+, Clang 11+, MSVC 2019+)
- CMake 3.18+ (for Beast Canyon and engine builds)
- A GGUF model file (any architecture — Llama, Qwen, Mistral, etc.)

**Optional:**
- CUDA Toolkit 12.1+ (NVIDIA GPU backend)
- Vulkan SDK 1.1+ (cross-platform GPU backend)
- Python 3.10+ with PyTorch (torch backend, ComfyUI)
- Qualcomm Hexagon SDK 5.x (phone DSP backend)
- Android NDK r25+ (phone builds)

---

## 1. Building the Core Library

### Linux / macOS

```bash
cd shannon-prime
make                    # Builds libshannon_prime.a (CPU backend)
make test-all           # Runs 187/188 tests across 8 suites

# With GPU backends
make SP_WITH_CUDA=1     # NVIDIA CUDA
make SP_WITH_VULKAN=1   # Vulkan (any GPU vendor)
make SP_WITH_CUDA=1 SP_WITH_VULKAN=1   # Both
```

### Windows (MSVC)

```powershell
# From Developer Command Prompt for VS 2019
cd shannon-prime
cmake -B build -G "Visual Studio 16 2019" -A x64
cmake --build build --config Release

# With CUDA (requires CUDA 12.1, NOT 13.2)
cmake -B build -G "Visual Studio 16 2019" -A x64 -DSP_WITH_CUDA=ON
cmake --build build --config Release
```

**Windows CUDA note:** Use CUDA 12.1. CUDA 13.2 has known compatibility issues with our MSVC 2019 build path. The build script is at `scripts/build_cuda_ext.ps1`.

### Android (ARM64)

```bash
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DSP_WITH_ADRENO=ON
cmake --build build-android
```

---

## 2. Building the Beast Canyon Engine

Beast Canyon is the heterogeneous desktop backend (Optane + AVX-512 + dual-GPU).

```bash
cd backends/beast_canyon
cmake -B build -DSP_BEAST_STANDALONE=ON
cmake --build build

# With GPU support
cmake -B build -DSP_BEAST_STANDALONE=ON -DSP_WITH_CUDA=ON -DSP_WITH_VULKAN=ON
cmake --build build
```

### First Run: Optane Audit

When your Optane drive arrives, validate it before loading a model:

```bash
./build/sp_beast_test /optane/path/to/model.gguf --audit-only
```

This runs four tests:
1. Sequential 4KB stride latency (target: < 15 µs)
2. Random 4KB page access latency (target: < 20 µs)
3. Sustained sequential read bandwidth (target: > 1 GB/s)
4. DAX status check

### Full Validation Suite

```bash
./build/sp_beast_test /optane/path/to/model.gguf
```

Runs reservoir mapping, Optane audit, AVX-512 shredder benchmark, MoE expert table dump, and full engine boot/shutdown.

---

## 3. Building the Standalone Engine

The Shannon-Prime Engine is the reference inference implementation.

```bash
cd shannon-prime-engine
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run inference
./build/sp-engine run --model /path/to/model.gguf --prompt "Hello, world"

# With compression
./build/sp-engine run --model /path/to/model.gguf \
  --sp-compression ship \
  --sp-k-bits 5,5,4,3 \
  --prompt "Hello, world"
```

---

## 4. Configuration

### Environment Variables

Shannon-Prime reads configuration from environment variables. These work across all integration points (engine, llama.cpp, ComfyUI).

```bash
# Core switches
SHANNON_PRIME=1                      # Master enable
SHANNON_PRIME_K_BITS=5,5,4,3        # K band bit allocation
SHANNON_PRIME_V_BITS=3              # V band allocation
SHANNON_PRIME_MOBIUS=1              # Möbius squarefree-first reorder

# Aggressive paths (pick one)
SHANNON_PRIME_SQFREE=1              # Sqfree + spinor path
SHANNON_PRIME_HIERARCHICAL=1        # Hierarchical Vilenkin predictor

# Quality control
SHANNON_PRIME_CAUCHY=2              # Dynamic Cauchy reset (0=off, 1=fixed, 2=dynamic)
SHANNON_PRIME_CAUCHY_N=512          # Fixed reset interval (mode 1 only)

# Positional encoding
SHANNON_PRIME_PRIME_PE=1            # PrimePE frequency injection
SHANNON_PRIME_PRIME_PE_ALPHA=0.17   # Blend ratio (0.0-1.0)

# Model preset
SHANNON_PRIME_PRESET=auto           # Resolve from GGUF arch ("auto", "off", or preset name)

# Role (for speculative decoding)
SHANNON_PRIME_ROLE=target           # "target" or "draft"

# Debug
SHANNON_PRIME_VERBOSE=1             # Per-vector correlation logging
```

### Recommended Configurations

**Desktop (8B+ model, NVIDIA GPU):**
```bash
SHANNON_PRIME=1 SHANNON_PRIME_PRIME_PE=1
# Ship defaults are correct. No tuning needed.
```

**Desktop (MoE, Beast Canyon with Optane):**
```bash
SHANNON_PRIME=1 SHANNON_PRIME_PRIME_PE=1 SHANNON_PRIME_CAUCHY=2
# Beast Canyon handles MoE routing automatically.
```

**Phone (3B model, Snapdragon 8 Gen 1):**
```bash
SHANNON_PRIME=1 SHANNON_PRIME_SQFREE=1 SHANNON_PRIME_CAUCHY=2
# Sqfree gives better compression for memory-constrained devices.
```

**Speculative decoding:**
```bash
# Target model
SHANNON_PRIME=1 SHANNON_PRIME_ROLE=target SHANNON_PRIME_K_BITS=5,5,4,3

# Draft model (more aggressive — errors are recoverable)
SHANNON_PRIME=1 SHANNON_PRIME_ROLE=draft SHANNON_PRIME_K_BITS=4,4,3,3
```

---

## 5. ComfyUI Setup

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nihilistau/shannon-prime-comfyui
cd shannon-prime-comfyui
pip install -e .
```

Restart ComfyUI. The nodes appear under the "shannon-prime" category:

1. **ShannonPrimeWanCache** — Cross-attention K/V caching for Wan 2.x
2. **ShannonPrimeWanBlockSkip** — Block-level skip caching (the main performance node)
3. **ShannonPrimeWanCacheFlush** — Flush caches before VAE decode

See `workflows/` for ready-to-use workflow JSON files.

---

## 6. LM Studio Integration

```powershell
cd shannon-prime-llama
# Apply the full-engine patch to llama.cpp b8861
git apply patches/llama-cpp-b8861-full-engine.patch

# Build LM Studio runtime DLLs
cd lmstudio
.\build.bat    # Produces llama.dll + ggml.dll

# Drop into LM Studio's runtime directory
copy llama.dll "C:\Users\<you>\.cache\lm-studio\runtimes\..."
copy ggml.dll  "C:\Users\<you>\.cache\lm-studio\runtimes\..."
```

Set environment variables before launching LM Studio:
```powershell
$env:SHANNON_PRIME=1
$env:SHANNON_PRIME_PRIME_PE=1
& "C:\...\LM Studio.exe"
```

---

## 7. Things to Watch Out For

### Quantization Floor

**Never use 2-bit on any band.** The 3-bit floor is load-bearing. 2-bit quantization on any single band causes catastrophic quality degradation regardless of what the other bands are set to. The minimum safe configuration is 3/3/3/3.

### IQ2 + Code Models

**Do not use IQ2 quantization for Qwen2.5-Coder-3B correctness benchmarks.** IQ2 of Qwen-Coder produces gibberish. Use Q5_K_M (`qwen-target.gguf`) for correctness validation.

### V Quantization

**Use flat allocation for V vectors.** Banded quantization helps K vectors (which have strong spectral structure from RoPE). V vectors have a smoother distribution where banded allocation provides no benefit. Single-band flat quantization at 3 bits is the validated default.

### Spinor Requires Q8+

**The spinor sheet bit is a Q8+ feature.** On Q3 models, the same K-corr loss costs ~7× more PPL, washing out the 1-bit correction. The scaling law explains this: the super-linear bits exponent means low-precision weights amplify compression errors.

### CUDA Version

**Use CUDA 12.1 on Windows.** Not 13.2. The MSVC 2019 template instantiation patterns in the ggml CUDA backend produce errors on CUDA 13.2 that don't occur on 12.1.

### DLL Import Pattern (Windows)

When building CUDA extensions on Windows, use `os.add_dll_directory()` before importing the extension module. Without this, Python's DLL loader can't find the CUDA runtime. See `reference_build_env` in the project notes.

### Edit Tool Truncation (Development)

When editing large files with AI coding assistants, the Edit tool can silently truncate large `new_string` blocks. Always verify the file's last lines after a large edit. Use bash `tail` or `wc -l` to confirm the file is complete.

### Harness Before Code

**If defaults seem broken, suspect the harness first.** Shannon-Prime's invariants are extensively validated (187/188 tests). If a new integration produces unexpected results, verify: Is the harness feeding the right data? Is the compression config what you think it is? Is the environment variable actually set?

---

## 8. Running Tests

```bash
# All test suites
make test-all

# Individual suites
make test-vht2              # VHT2 transform round-trip
make test-mobius             # Möbius reorder/unreorder
make test-band              # Banded quantization
make test-shadow             # Shadow cache write/read
make test-sqfree             # Sqfree + spinor path
make test-hier               # Hierarchical predictor
make test-cauchy             # Cauchy reset system
make test-scaling            # Scaling law predictions
```

Expected: 187/188 passing. The one failing test is a synthetic-K flake in the round-trip suite (synthetic K vectors don't have the spectral structure that real RoPE'd K vectors have, so the correlation target is occasionally missed by a few ULP).

---

## 9. Performance Benchmarking

### Perplexity Evaluation

```bash
# Engine
./sp-engine ppl --model /path/to/model.gguf --dataset wikitext-2 --chunks 3

# llama.cpp
./llama-perplexity -m /path/to/model.gguf -f wiki.test.raw --chunks 3
```

### Throughput

```bash
# Engine
./sp-engine bench --model /path/to/model.gguf --n-tokens 512

# Beast Canyon
./sp_beast_test /optane/path/to/model.gguf
```

### Speculative Decoding

```bash
./llama-cli -m target.gguf -md draft.gguf \
  --draft-max 8 --draft-min 2 \
  -p "Write a function that sorts an array"
```

See `docs/BENCH-SPEC-DECODE.md` for the benchmarking harness and result interpretation.
