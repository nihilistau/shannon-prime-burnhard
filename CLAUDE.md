# CLAUDE.md — Shannon-Prime Engine: Agent Instructions

> ### 2026-05-13 update — two paths, one repo
>
> This repo now hosts **two parallel inference paths** that share the
> Shannon-Prime math core but have different deployment models. Read
> the path-specific section before working anywhere in the tree.
>
> | Path | Status | Target hardware | Entry point |
> |---|---|---|---|
> | **Desktop** | **MCP + LM Studio** is the API | RTX 2060 + i9-11900KB + Optane | `C:\Files\MCP\lmstudio-server\server.py` (private repo `shannon-prime-lmstudio-server`) |
> | **Mobile** | sp-engine standalone (this repo) | S22U / Hexagon V69 HTP | `build-android/bin/sp-engine` |
>
> The desktop architecture diverged from the original "standalone
> sp-engine binary" plan: validation showed that **LM Studio +
> shannon-prime-llama (modded `llama.dll`)** already does on-GPU
> inference better than any custom CUDA we wrote — see the regression
> diagnoses in `desktop/research/README.md`. The work in
> `desktop/research/` and `desktop/policy/` is kept as research baseline
> + future integration target for Shannon-Prime-QuickBorn's diffusion-
> draft work. The mobile QNN path below is unchanged.

## Desktop Path (LM Studio + MCP)

### What it is

Inference runs in **LM Studio** with the
[shannon-prime-llama](https://github.com/nihilistau/shannon-prime-llama)
patched `llama.dll` as the active runtime (see
`lms runtime ls` — the entry with `-shannon-prime@` is the one that
applies VHT2 KV compression + PrimePE on top of stock llama.cpp).

The **public API** is a FastMCP server at
`C:\Files\MCP\lmstudio-server\` (private GitHub repo
`shannon-prime-lmstudio-server`). It exposes 100+ tools split:

  - **79 system-utility tools** (filesystem, shell, archives, network,
    text processing — original implementation)
  - **24 inference tools** (`lmstudio_*`, `sp_engine_*`, `inference_*`)
    that orchestrate the LM Studio API + sp-engine fallback + telemetry
    + cross-runtime bench

### Verified numbers on this box (2026-05-13)

```
Hardware:          RTX 2060 12GB + i9-11900KB + 32GB DDR4-2666 + Optane
Runtime:           llama.cpp-win-x86_64-nvidia-cuda12-avx2-shannon-prime@2.14.0
Model:             qwen/qwen3.6-35b-a3b  Q4_K_M  ctx=8192
Loaded with:       num_experts=4, flash_attention=true,
                   offload_kv_cache_to_gpu=true, parallel=2

Warm decode (reasoning=off):     17.0 tok/s  (decode-only: 17.1 tok/s)
TTFT (25 prompt tokens):         0.82 s
Integrations chat through MCP:   22.5 tok/s
                                  (model called inference_telemetry
                                   tool, synthesized result coherently)

Storage tiers (DiskSpd 4K random QD=1):
  F: Optane 32GB (gen 4x4)    18 us avg latency  <-- best for mmap
  D: NVMe                     22 us
  E: Optane 16GB              21 us  (saturates at QD=32)
  C: NVMe                     28 us

sp_pointer_view hash lookup:     50 ns/call vs 1100 ns linear scan (22x)
sp_optane_find_tensor (wired):   65536 buckets / 31453 tensors in 3 ms
```

### Critical auth facts (LM Studio 0.4+)

LM Studio has **TWO independent local auth systems**:

1. **REST API tokens** — Bearer string in `Authorization` header,
   minted via UI "Manage Tokens". Works for `/v1/`, `/api/v0/`,
   `/api/v1/`. The shannon-prime-lmstudio-server MCP reads this from
   `LM_API_TOKEN` env var (or call `inference_set_token` at runtime).
   Local token format: `sk-lm-<id>:<passkey>` joined by `:`.

2. **SDK WebSocket pairing** — separate `clientIdentifier +
   clientPasskey` pair, stored as SHA-512 hashes in
   `~/.cache/lm-studio/.internal/permissions-store.json`. Created
   interactively when an SDK app first connects (GUI shows
   "Allow API access?" dialog). REQUIRED for SDK-exposed features:
   **speculative decode (`draftModel`), model load/unload via SDK**.

   On this box: 13 paired clients exist (SILLYTAVERN, Beast, claude,
   cosysim2, etc.) but the lmstudio-server MCP has not been paired
   yet — must be done with LM Studio GUI running.

3. **Model loading guardrails** at
   `~/.cache/lm-studio/settings.json -> modelLoadingGuardrails`
   block fresh model loads when free RAM is tight. `integrations` on
   a 22 GB model needed `mode: "off"` (or unload-and-reload) to work.

### Spec decode status (2026-05-13)

- REST `/api/v1/chat` and `/api/v1/models/load` both **reject** every
  draft-related field name we probed (`draft_model`, `draftModel`,
  `speculative`, `speculative_decoding`, `spec_decode`,
  `draft_model_key`). HTTP 400 `unrecognized_keys`.
- Spec decode is **SDK-only** on this LM Studio build.
- SPflash drafts visible in `lms ls`:
  `qwen3.6-35b-a3b-draft@q4_k_m` (291 MB, arch `dflash-draft`)
  `qwen3.6-35b-a3b-draft@q8_0`   (515 MB)
  Both ready to use as soon as the SDK pairing is done.

### The MCP tool surface

(See `C:\Files\MCP\lmstudio-server\README.md` for full table; quick
index by area:)

  - `lmstudio_daemon_*`     — `lms server start/stop/status`
  - `lmstudio_status`       — health-probe REST + SDK
  - `lmstudio_list_models`  — SDK preferred, REST fallback
  - `lmstudio_loaded_ps`    — `lms ps`
  - `lmstudio_v1_active_config` — actual applied load config
  - `lmstudio_load_model` / `lmstudio_v1_load` — load with full config
  - `lmstudio_rest_chat_v1` — native `/api/v1/chat` (text + images +
                                integrations + allowed_tools + streaming)
  - `lmstudio_quick_bench`  — one-shot warm tok/s
  - `lmstudio_chat` / `_stream_chat` / `_spec_chat` — SDK path (auth pending)
  - `lmstudio_auth_help`    — step-by-step for both auth systems
  - `sp_engine_chat` / `_bench` / `_set_env` — fallback to burnhard
                                                 sp-engine.exe
  - `inference_recommend_load_config` — port of
                                          `desktop/policy/sp_layer_split_policy.cpp`
                                          made dynamic
  - `inference_bench_compare` — side-by-side runtimes
  - `inference_telemetry`   — nvidia-smi + psutil snapshot
  - `inference_recent_runs` — ring buffer of recent runs

The model is **registered in `~/.cache/lm-studio/mcp.json` as
`system-controller`** — chat sessions can invoke any of our tools by
referencing `integrations=["mcp/system-controller"]` + an
`allowed_tools` allowlist.

### Desktop repo layout (post 2026-05 reorg)

```
core/                                 — Shannon-Prime math + new pointer_view
  include/sp_pointer_view.h           — mmap'd GGUF, O(1) hashed lookup
  src/sp_pointer_view.c               — 330 LOC, no deps, Win + POSIX
  src/sp_pointer_view_test.c          — standalone bench
  (existing core/* — VHT2, PrimePE, sqfree, etc. unchanged)

desktop/beast_canyon/                 — engine + active orchestration
  sp_beast_canyon.{c,h}               — engine (SSM path)
  sp_optane.{c,h}                     — reservoir + hash index
  sp_hetero_sync.{c,h}                — GPU enumeration
  sp_level_zero.{c,h}                 — Intel L0 dispatch
  sp_diagnostics_bc, sp_shadow_steal,
  sp_thermal_throttle,
  sp_prefetch_telemetry, sp_sidecar_proto.h

desktop/research/                     — research baselines (still builds
                                         into sp_beast_canyon via
                                         ../research/ in CMakeLists)
  sp_avx512_shredder.*                — CPU AVX-512 dequant
  sp_shredder_crt.*                   — CRT dual-ring matmul
  sp_shredder_svm.cpp                 — L0 SVM Shredder variant
  sp_beast_gpu.{cpp,h}                — cuBLAS gemv wrapper (regression
                                         diagnosis — Win WDDM sync floor)
  sp_beast_gpu_q4k.cu                 — Q4_K-resident CUDA kernel
  README.md                           — full rationale

desktop/policy/                       — orchestration policy
  sp_layer_split_policy.cpp           — 20-RTX + top-K=4 + CRT policy
                                         (orphan code; values baked into
                                         the live LM Studio load config)
  README.md

bench/                                — benches + analysis
  optane_bench_2026-05-13_summary.md  — Optane vs NVMe tiers
  ...

bridge/, mobile/                      — sidecar + Hexagon (unchanged)
```

### Rules for desktop work

DO:
- Use the MCP (`shannon-prime-lmstudio-server`) for any inference call
  the agent needs to make on the developer's box.
- Lean on LM Studio's `/api/v1/chat` (rich stats + SSE streaming +
  integrations) over the OpenAI-compat `/v1/`.
- Pass `reasoning="off"` for Qwen3.x reasoning models unless you want
  thinking tokens.
- Re-run `lmstudio_v1_active_config` before benchmarking — confirms
  the loaded config matches the bench premise.
- Treat `sp-engine.exe` as **research/fallback** — the `--beast`
  short-circuit path produces coherent output on Qwen3.6-A3B; the
  ForwardContext path produces gibberish on hybrid-MoE arches and is
  not the target of further fixing.

DO NOT:
- Try to make the `desktop/research/` GPU code beat the LM Studio path
  on this hardware tier. Win WDDM sync floor is the wall; that work
  belongs on Linux + TCC driver, or as batched-multi-token decode
  alongside the SP-Flash QuickBorn work.
- Edit `desktop/engine/src/dflash_weights.cpp` or
  `sp_flash_oracle_native.cpp` — that work moved to
  `C:\Projects\Shannon-Prime-QuickBorn` (separate GitHub repo). Future
  port-back is planned.
- Hand-write CLI clients for the inference path. The MCP IS the API.

---

## Mobile Path (Hexagon QNN HTP — unchanged below)

## What This Is

Shannon-Prime Engine (`sp-engine`) is a **custom heterogeneous inference runtime**
for Qualcomm Snapdragon SoCs, specifically targeting the Samsung Galaxy S22 Ultra
(SM8450 / Waipio / Hexagon V69 HTP). It runs Qwen3-4B (w4a16) at **500+ tok/s
prefill** and **50+ tok/s decode** using the full silicon: HTP tensor cores, HVX
vector extensions, Halide-orchestrated DMA, and ARM NEON — simultaneously.

This is NOT a wrapper around llama.cpp or ggml for inference. The engine owns its
own GGUF loader, tokenizer, forward pass, KV cache, and HTTP server. It uses ggml
only as a tensor-math library for the CPU path. The QNN path (`qnn_bin_driver`)
bypasses all of that and dispatches whole-model graphs directly to the V69 HTP.

**Everything described here is implemented, tested, and validated on-device.**

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│  sp-engine CLI  (main.cpp)                                 │
│    verbs: generate | serve | ppl | qnn_bin_run | qnn_serve │
├──────────┬──────────┬──────────┬───────────────────────────┤
│  Engine  │ HttpServer│ Tokenizer│  QnnBinDriver             │
│ (engine.h)│(http_server.h)│(tokenizer.h)│(qnn_bin_driver.h) │
├──────────┴──────────┴──────────┴───────────────────────────┤
│  ForwardContext / ForwardNativeContext                      │
│  KvCache (ship/sqfree/hier/dual)                           │
│  LlamaWeights / GgufLoader                                 │
├────────────────────────────────────────────────────────────┤
│  lib/shannon-prime (submodule)                             │
│    core/    — VHT2, Möbius, banded quant, sqfree, spinor   │
│    backends/cuda/   — NVIDIA GPU kernels                   │
│    backends/adreno/ — ARM NEON tiers                       │
│    backends/hexagon/— Hexagon HVX + FastRPC stubs          │
│    backends/qnn_aihub/ — sp_qnn.c: QNN HTP shim           │
│    backends/halide/ — DMA orchestration (Mode C/D)         │
│    backends/freethedsp/ — unsigned PD shim                 │
├────────────────────────────────────────────────────────────┤
│  vendor/ggml      — tensor math (CPU/CUDA/Vulkan)          │
│  vendor/cpp-httplib — HTTP server                          │
├────────────────────────────────────────────────────────────┤
│  frontend/        — glassmorphic web UI (served on-device) │
│    index.html, app.js, style.css                           │
│    Connects to /v1/chat/completions (OpenAI-compat)        │
└────────────────────────────────────────────────────────────┘
```

## Proven State (as of 2026-05-04)

These are NOT targets or aspirations. They are validated on the S22 Ultra:

| Metric | Value | Evidence |
|--------|-------|----------|
| Prefill throughput | 423–500+ tok/s | 302ms for 128-token chunk, 4-split pipeline |
| HTP residency | 4-split concurrent | `shareResources=true`, 1.5GB myth debunked |
| Numerical coherence | nan=0 across all splits | UFIXED_16 bridge working |
| Teardown | Clean exit (code 0) | dlclose deferred to process exit |
| HTTP server | 0.0.0.0:8080 | OpenAI-compat API + static file server |
| Frontend | Glassmorphic dark-mode | Live status, hardware telemetry, quick prompts |

## Hardware Constants (NEVER CHANGE WITHOUT DEVICE EVIDENCE)

These are empirically discovered constants baked into the V69 HTP pipeline:

| Constant | Value | Why |
|----------|-------|-----|
| UFIXED_POINT_16 dtype | 1046 | AI Hub bakes quantized uint16, NOT fp16 (534) |
| Zero-point offset | 30833 | Embedding output: `fp32 ≈ 7.2e-6 × uint16 − 0.222` |
| Buffer alignment | 4096 bytes | ION/SMMU requires page-aligned; unaligned = DMA stutter |
| HTP context priority | QNN_PRIORITY_LOW | Reduces TCM pressure for 4-split co-residency |
| Cos/Sin scale | 3.051804378628731e-05 | Offset: -32768 |
| Attention mask scale | 0.0015259021893143654 | Offset: -65535 |
| KV cache per-layer | See qnn_bin_quant_table.h | 12 layers × {key,value} × {scale,offset} |

## Build

### Android (aarch64 — the production target)
```bash
# From the engine root, with NDK r26+ in PATH:
mkdir -p build-android && cd build-android
cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 \
  -DSP_ENGINE_WITH_QNN=ON -DSP_ENGINE_WITH_HEXAGON=ON \
  -DSP_ENGINE_HEXAGON_FASTRPC=ON \
  -DCMAKE_BUILD_TYPE=Release
ninja
# Output: build-android/bin/sp-engine
```

### Windows CUDA (desktop dev/bench)
```powershell
mkdir build-cuda; cd build-cuda
cmake .. -G Ninja -DSP_ENGINE_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release
ninja
# Output: build-cuda/bin/sp-engine.exe
```

## Deploy to Device

```bash
# Push engine binary + QNN runtime libs
adb push build-android/bin/sp-engine /data/local/tmp/sp-engine/
adb push build-android/bin/lib*.so /data/local/tmp/sp-engine/

# Push frontend
adb push frontend/ /data/local/tmp/sp-engine/www/

# Push model + QNN splits
adb push /path/to/qwen-target.gguf /data/local/tmp/sp22u/model.gguf
adb push /path/to/qwen3_4b_*.bin /data/local/tmp/sp22u/qnn/
```

## Run

### QNN HTP inference (production path)
```bash
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/sp-engine:/data/local/tmp/sp22u/qnn:\$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH='/data/local/tmp/sp-engine;/vendor/lib/rfsa/adsp' && \
           /data/local/tmp/sp-engine/sp-engine qnn_bin_run \
           --tokenizer /data/local/tmp/sp22u/model.gguf \
           --prompt 'The capital of France is' \
           /data/local/tmp/sp22u/qnn/qwen3_4b_1_of_4.bin \
           /data/local/tmp/sp22u/qnn/qwen3_4b_2_of_4.bin \
           /data/local/tmp/sp22u/qnn/qwen3_4b_3_of_4.bin \
           /data/local/tmp/sp22u/qnn/qwen3_4b_4_of_4.bin"
```

### Serve (HTTP + frontend)
```bash
adb shell "chmod +x /data/local/tmp/sp-engine/sp-engine && \
           export LD_LIBRARY_PATH=/data/local/tmp/sp-engine:/data/local/tmp/sp22u/qnn:\$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH='/data/local/tmp/sp-engine;/vendor/lib/rfsa/adsp' && \
           /data/local/tmp/sp-engine/sp-engine serve \
           --model /data/local/tmp/sp22u/model.gguf \
           --host 0.0.0.0 --port 8080 \
           --www /data/local/tmp/sp-engine/www"
```
Then open `http://<phone_ip>:8080` in any browser.

## Key Source Files

| File | What It Does |
|------|-------------|
| `src/cli/main.cpp` | CLI entry — verbs, backend selection, System 1/2 entropy gate |
| `src/engine.h` | Public API — Config struct with ALL compression switches |
| `src/engine.cpp` | Engine::load / generate / perplexity |
| `src/qnn_bin_driver.cpp` | 4-split QNN pipeline: load → exec → carry residual → sample |
| `src/qnn_bin_quant_table.h` | Recovered UFIXED_16 scales for all tensors (the Rosetta Stone) |
| `src/http_server.cpp` | OpenAI-compat /v1/chat/completions + static file server |
| `src/forward.cpp` | CPU/GPU forward pass (ggml graph) |
| `src/kv_cache.cpp` | KvCache with ship/sqfree/hier/dual modes |
| `lib/shannon-prime/backends/qnn_aihub/sp_qnn_runner/sp_qnn.c` | QNN HTP shim — dlopen, load, execute, shared resources |
| `lib/shannon-prime/backends/qnn_aihub/sp_qnn_runner/sp_qnn.h` | sp_qnn public C API |
| `frontend/` | On-device web UI (glassmorphic, connects to /v1/) |

## The QNN Pipeline (how inference actually works on-device)

1. `sp_qnn_init()` — dlopen libQnnHtp.so + libQnnSystem.so once
2. `sp_qnn_load_binary_list()` — load 4 splits as a GROUP with `shareResources=true`
   and `QNN_PRIORITY_LOW`. All 4 contexts co-reside in HTP memory.
3. For each token:
   - Build inputs: token IDs, position_ids (cos/sin via RoPE), attention_mask, KV cache
   - Encode all inputs to UFIXED_16 using scales from `qnn_bin_quant_table.h`
   - Execute splits 1→2→3→4, carrying residual stream between them
   - Decode logits from UFIXED_16 → fp32
   - Argmax → next token ID
4. `sp_qnn_destroy()` — deregister persistent ION tensors, free contexts

The residual stream between splits uses **host-side 4096-byte-aligned buffers**
allocated via `memalign`. The KV cache inputs are initialized with the **30833
zero-point** (not zero!) to represent "empty" in the UFIXED_16 domain.

## Heterogeneous Silicon Map (Current + Planned)

| Silicon Block | Current Use | Planned Use |
|---------------|-------------|-------------|
| HTP (Tensor Core) | All MatMuls, Attention (w4a16) | Same — this is the engine room |
| HVX (Vector Core) | Idle | **Phase 6**: Softmax, LayerNorm, Argmax on-DSP |
| Halide DMA | Reference only (probe built) | **Phase 7**: Async layer N+1 prefetch from UFS |
| ARM NEON | Tokenization, argmax, entropy | **Phase 8**: Speculative oracle, MoE expert routing |
| Spectra 680 ISP | Not used | **Phase 10 (Mode D)**: 18-bit spectral reconstruction |
| UFS 3.1 | Model storage | Weight streaming source for triple-buffered pipeline |
| Adreno 730 GPU | Desktop CUDA path only | Future: hybrid GPU+HTP for larger models |

## Rules for AI Agents

### DO:
- Read CODE before making claims about what works or doesn't
- Trust empirical device results over theoretical analysis
- Use `qnn_bin_quant_table.h` scales — they were recovered by probing, they're correct
- Build and test on Android (the production target) — Windows is dev-only
- Validate through the frontend on-device as the final acceptance test
- Use PowerShell MCP for local git/build operations (don't defer to user)

### DO NOT:
- Retreat to ggml/llama.cpp standard paths. We built the engine to NOT do that.
- Call proven things "hard" or "complex" — they're done, they work
- Declare the 1.5GB HTP limit real — it's debunked, 4-split co-residency proven
- Hallucinate that UFIXED_16 is fp16 — it's quantized uint16 with per-tensor scale+offset
- Feed zero-initialized KV cache buffers — use zero-point 30833
- Use unaligned buffers — 4096-byte alignment or the DMA stutters
- Suggest "wrapping up" or "calling it a night" — the user works when they want
- "Optimize" invariants without understanding them (read the 10 invariants in lib/shannon-prime/CLAUDE.md)
- Frankenpatch — flag build errors, don't invent fixes mixing code from different paths
- Treat engine bench numbers as SP quality metrics — they measure different things

### CRITICAL INVARIANTS:
1. dtype 1046 = UFIXED_POINT_16 (quantized uint16), NOT fp16
2. Zero-point 30833 for embedding output; -32768 for cos/sin; -65535 for mask
3. 4096-byte buffer alignment for ION/SMMU
4. QNN_PRIORITY_LOW for 4-split TCM co-residency
5. `shareResources=true` in `createFromBinaryListAsync` for shared kernel/workspace
6. The residual stream carries between splits on HOST buffers (not HTP-internal)
7. The quant scales in `qnn_bin_quant_table.h` are empirically recovered — don't guess new ones
8. Frontend at /www/ is the acceptance test — if it doesn't work in the browser, it's not done
9. VHT2 self-inverse: VHT2(VHT2(x)) = x — no 1/N on inverse
10. 3-bit floor on any VHT2 band — 2-bit is catastrophic

## Useful Paths

| What | Path |
|------|------|
| Engine source | `D:\F\Projects\Shannon-Prime-Refactor\shannon-prime-engine\` |
| Shannon-Prime math core | `lib\shannon-prime\` (submodule) |
| QNN shim | `lib\shannon-prime\backends\qnn_aihub\sp_qnn_runner\` |
| QNN splits on device | `/data/local/tmp/sp22u/qnn/qwen3_4b_*.bin` |
| Model GGUF on device | `/data/local/tmp/sp22u/model.gguf` |
| Engine binary on device | `/data/local/tmp/sp-engine/sp-engine` |
| Frontend on device | `/data/local/tmp/sp-engine/www/` |
| QNN SDK | `C:\Qualcomm\` (QAIRT 2.45.40.260406) |
| Qualcomm tools | `C:\Qualcomm\HALIDE_Tools\`, `C:\Qualcomm\Hexagon_IDE\` |
| Old workspace | `D:\F\shannon-prime-repos\` (reference, not active dev) |

## Papers and Theory

The math is in `lib/shannon-prime/`:
- `position_is_arithmetic_v8.md` — PE theory, ZetaZeroPredictor
- `kv_cache_is_a_view_v2.md` — VHT2, Vilenkin, Möbius, spinor, scaling law
- `multiplicative_lattice_combined.md` — unified synthesis

The scaling law: `log(PPL / PPL_base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)`

## Frontend Integration

The web UI at `frontend/` connects to the engine's HTTP server:
- `GET /v1/models` — returns model info (hardware, backend)
- `POST /v1/chat/completions` — OpenAI-compat chat endpoint
- Static files served from `--www` directory
- Status indicator polls engine health
- Displays: model name, hardware (Snapdragon 8 Gen 1), backend (QNN HTP V69)
- Quick-prompt buttons for common tasks

**Every phase of work must be validated through this frontend on-device.**
If the user can't chat coherently through the browser, the work isn't done.
