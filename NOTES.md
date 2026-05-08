# Shannon-Prime BurnHard — Development Notes

## Build Environment

### Windows Build Setup (Beast Canyon i9)

**Compilers:**
- MSVC via VS2019 BuildTools: `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat`
- VS18 BuildTools (newer): `D:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat`
- MinGW GCC 15.2.0 at `C:\ProgramData\mingw64\mingw64\bin` — **DO NOT USE with CUDA**, cmake picks it up by accident via PATH

**CUDA:**
- CUDA 13.2.51 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`
- Target arch: SM75 (RTX 2060)
- nvcc requires MSVC host compiler, NOT MinGW

**CMake:**
- cmake 4.2 from pip: `C:\Users\Knack\AppData\Local\Programs\Python\Python313\Scripts\cmake.exe`
- **GOTCHA**: cmake from pip has broken rc.exe/mt.exe detection when using Ninja with MSVC. Must call `vcvarsall.bat x64` BEFORE cmake to get rc.exe on PATH.
- **GOTCHA**: Must pass `-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl` when using Ninja to prevent cmake from finding MinGW gcc first.

**Build pattern (NO CUDA — fast iteration):**
```bat
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d C:\Projects\shannon-prime-burnhard\build_nocuda
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DSP_WITH_CUDA=OFF -DSP_WITH_BEAST=ON -DSP_BEAST_CUDA=OFF -DSP_BEAST_LEVEL_ZERO=ON
cmake --build . --config Release --target sp-engine -j 8
```

**Build pattern (WITH CUDA — full Beast Canyon):**
```bat
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d C:\Projects\shannon-prime-burnhard\build_beast
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DSP_WITH_CUDA=ON -DSP_WITH_BEAST=ON -DSP_BEAST_CUDA=ON -DSP_BEAST_LEVEL_ZERO=ON
cmake --build . --config Release --target sp-engine -j 8
```

**MSVC Stack size:**
- sp-engine linked with `/STACK:33554432` (32MB) to handle deep ggml graph builds

### Key Paths

| What | Path |
|------|------|
| Burnhard repo | `C:\Projects\shannon-prime-burnhard` |
| Bernhard repo | `C:\Projects\shannon-prime-bernhard` |
| Legacy repos | `D:\F\shannon-prime-repos\` |
| Custom llama.cpp | `https://github.com/nihilistau/shannon-prime-llama` |
| Models | `D:\Files\Models\lmstudio-community\` |
| Qwen3.6-35B-A3B | `D:\Files\Models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\` |
| Qwen3.6-27B | `D:\Files\Models\lmstudio-community\Qwen3.6-27B-GGUF\` |
| Gemma4-31B | `D:\Files\Models\lmstudio-community\gemma-4-31B-it-GGUF\` |
| Bench corpus | `C:\Projects\shannon-prime-burnhard\bench\test_corpus.txt` |

### Model Architectures Tested

| Model | Arch | Status |
|-------|------|--------|
| Qwen2.5-Coder-0.5B | qwen2 | PASS — logits verified |
| Phi-3.1-mini | phi3 | PASS |
| Qwen3.6-35B-A3B | qwen35moe | PARTIAL — 10/40 MOE_ATTN layers OK, 30 MOE_GDN layers need fused op |

## Fixes & Gotchas

### 2026-05-09: ggml 0.9.11 → 0.11.0 update
- **Problem**: GDN (Gated-DeltaNet) layers crash in the chunked prefill path. `ggml_solve_tri` and `ggml_set_inplace` cause ACCESS_VIOLATION.
- **Root cause**: The chunked prefill path manually builds solve_tri + set_inplace per chunk in a for-loop. The ggml graph allocator and compute scheduler handle this incorrectly.
- **Fix**: ggml 0.11.0 has a fused `ggml_gated_delta_net(q,k,v,g,beta,state)` op that handles n>1 correctly. Refactor forward.cpp to use the fused op for ALL token counts, not just n==1.
- **Source**: Pulled ggml 0.11.0 from `ggml-org/llama.cpp` master (b8763, 2026-05-08). The standalone ggml repo is only at v0.9.5.

### 2026-05-08: vcvarsall + Ninja + CUDA
- cmake from pip can't find rc.exe without vcvarsall sourced first
- Must explicitly set `-DCMAKE_C_COMPILER=cl` or MinGW gcc gets picked up
- CUDA compilation is SLOW (~2min+ for .cu files). Use build_nocuda for iteration, build_beast for final.

### 2026-05-08: GDN state tensor allocation
- If GDN layers are skipped, must set `have_gdn_shapes = false` in forward.cpp
- Otherwise gallocr asserts on unreferenced input tensors (they get marked as input but no graph node consumes them)

### 2026-05-08: bench/test_corpus.txt
- Was 0 bytes in burnhard (broken copy). Replaced with 198KB corpus from engine repo.

## Custom LM Studio / llama.cpp

- Custom llama.cpp lives at: https://github.com/nihilistau/shannon-prime-llama
- This includes the LM Studio build, expert splitting, SP integration
- Burnhard engine is the CUSTOM engine — not a llama.cpp wrapper
- Expert splitting was already working in the LM Studio package

## Hardware

### Beast Canyon (Desktop)
- CPU: Intel i9 (AVX-512 F+BW+VNNI)
- GPU0: NVIDIA RTX 2060 12GB (SM75, CUDA)
- GPU1: Intel UHD 750 (32 EUs, Level Zero)
- Storage: Optane drive (current), 32GB NVMe (arriving), P4800X 375GB Optane U.2 (arriving)
- RAM: sufficient for 35B mmap'd GGUF

### S22 Ultra (Mobile Sidecar)
- SoC: Snapdragon 8 Gen 1 (SM8450)
- CPU: 1×X2 + 3×A710 + 4×A510
- GPU: Adreno 730 (Vulkan)
- DSP: Hexagon V69 (HVX, NO HMX)
- Connection: USB-C / ADB bridge
