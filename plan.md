# Shannon-Prime: Full Project Plan

**Date**: 2026-05-09
**Mono-repo**: `C:\Projects\shannon-prime-burnhard\`
**Source repos**: `D:\F\shannon-prime-repos\shannon-prime\`, `shannon-prime-engine\`, `shannon-prime-llama\`

---

## 1. Source Repo Inventory — What Exists

### 1.1 shannon-prime (math core)

The foundational C library. Everything below is **implemented and tested**.

**Core math** (`core/`):
- `shannon_prime.c/.h` — VHT2 spectral transform (self-inverse butterfly), banded quantization (`sp_band_quantize`/`sp_band_dequantize`), Möbius reordering, sqfree/spinor compression, hierarchical predictor, shadow cache (ship-path), fp16 utilities. `SP_MAX_HEAD_DIM=256`, `SP_MAX_BANDS=16`. Default K bits `{5,5,4,3}`, V bits `{3}`.
- `shannon_prime_modelpack.c/.h` — per-arch compression preset registry. phi3 CALIBRATED (+2.44%), qwen3 edge-fail (+5.14%), 7 others PROVISIONAL (llama, qwen2, mistral3, granite, gemma3, gemma4, qwen35moe). `sp_modelpack_lookup()` returns band configs, sqfree flags, hier flags, suggested draft model.
- `shannon_prime_cauchy.c` — Cauchy reset system: Zeta Schedule (Layer 1), Mertens Oracle (Layer 2, proactive arithmetic), Ricci Sentinel (Layer 3, reactive drift). 3-layer causal stability stack.
- `shannon_prime_sqfree.c` — squarefree/Knight-mask compression path.
- `shannon_prime_pe.c` — PrimePE lattice-aligned RoPE frequency generation.

**Backends** (`backends/`):
- `hexagon/` — Full Hexagon V69 DSP pipeline: FastRPC scaffold, HVX VHT2 butterfly kernel (1.76× vs scalar), HVX band_quantize with int-bit-pattern amax, qf32 intermediate, batched compress_f32 IDL, V-specific IDL (`compress_f32_v`), `sp_hexagon_cache_t`, rpcmem zero-copy, ping-pong scratch, VTCM acquisition, cycle bench, logit argmax, DMA probe.
- `qnn_aihub/` — QNN HTP V69 runner (`sp_qnn.c`): runtime graph build via `QnnGraph_addNode`, APP_WRITE weights, ION-backed persistent K, `sp_llama_qnn_matmul_dispatch`, rate projections (577 calls/sec FastRPC ceiling), w4a16 quantize+compile pipeline. Phase 2.5 runtime matmul validated at 238 µs @ 256×256 on V69.
- `adreno/` — Adreno GPU backend (NEON dispatch).
- `vulkan/` — Vulkan compute backend (`shannon_prime_vulkan.c`).
- `cuda/` — CUDA backend (`shannon_prime_cuda.h`, `vht2_wan_ext.cpp` PyTorch bridge).
- `halide/` — Halide V69 HVX generators, Mode D ISP probe (Spectra 680 MFNR/Super-Res), DMA raw probe. **Result**: DMA unreachable in unsigned PD on production S22U — stay on rpcmem.
- `freethedsp/` — geohot's cDSP unblock shim, baked into harness via `SP_FREETHEDSP=1`.

**Docs** (`docs/`):
- 7 core architecture documents (rewritten at 216bb85).
- Per-backend docs (BACKEND-HEXAGON, BACKEND-CUDA, BACKEND-VULKAN, BACKEND-ADRENO, BACKEND-TORCH).
- Feature docs (COMPRESSION-FEATURES, MATHEMATICAL-FOUNDATIONS, CAUCHY-RESET, DISK-TIER-ARCHITECTURE).
- Integration docs (INTEGRATION-LLAMA, INTEGRATION-COMFYUI).
- Modality docs (MODALITY-AUDIO, MODALITY-IMAGE, MODALITY-VIDEO).
- MODEL-PACK-CALIBRATION ledger.

### 1.2 shannon-prime-engine (GGUF inference engine)

Standalone inference engine. Builds against ggml 0.11.0 (b8763). **This is THE reference implementation**.

**Engine core** (`src/`):
- `gguf_loader.cpp/.h` — GGUF metadata reader. Supports llama, qwen2, qwen3, mistral3, phi3, granite, gemma3, gemma4, qwen35moe. Gemma4 KV fold derivation (n_head_kv=16 from Q/K tensor sizes), head_dim=256 from Q tensor.
- `llama_weights.cpp/.h` — Weight materialisation from GGUF into ggml_context. Supports: standard attention, MoE (qwen35moe), Gated DeltaNet (GDN), phi3 fused QKV, gemma3 sandwich norms, gemma4 V-sharing (backfill from global-attention neighbors), optional biases (qwen2, granite), per-head attention norms (qwen3, qwen35moe). Multi-GPU layer distribution. `LlamaLayerKind` enum: STANDARD, MOE_ATTN, MOE_GDN.
- `forward.cpp/.h` — ggml graph builder for llama-family forward pass. `build_block` (prefill) + `build_block_decode` (single-token). Handles: RoPE (standard + mRoPE mode 8), GQA, fused QKV (phi3), sandwich norms (gemma3), MoE expert routing (qwen35moe), GDN recurrence (qwen35moe), CRT matmul dispatch, logit softcapping. **Gemma4 KV fold**: `n_rot > head_dim` signal, pre-fold K reshape for RoPE, post-fold to effective heads. **FFN activation**: conditional GELU (gemma) vs SiLU.
- `kv_cache.cpp/.h` — SP-compressed KV cache. Ship-path (shadow), sqfree, hierarchical modes. GPU-resident variant (CUDA). Calibration (variance-ranked). Cold storage (GPU↔CPU↔disk tiering). Disk serialization (VHT2 v2/v3 binary). Cauchy reset integration. **DualKvCache**: System 1↔2 entropy-routed dual cache.
- `forward_native.cpp/.h` — ggml-free forward pass. Straight-line CPU kernel walk (RMSNorm, SiLU, Softmax, RoPE, MatMul). QNN HTP dispatch at per-head KQ matmul. SP-banded KV cache via KvCache. Hexagon FastRPC dispatch. Persistent thread pool.
- `forward_native_context.cpp/.h` — Context management for native forward. QNN n_kv_total bucketing.
- `gdn_state.cpp/.h` — GDN recurrent state cache (conv_state + ssm_state per GDN layer).
- `engine.cpp/.h` — Top-level Engine class. `generate()` with prefill/decode t/s instrumentation, 4-thread CPU, armv8.2-a NEON dotprod/fp16. Config struct with all SP knobs (sqfree, spinor, mobius, hierarchical, cauchy, k/v bits, PrimePE, etc.).
- `sp_kernels_cpu.cpp/.h` — CPU kernel library: RMSNorm, SiLU, Softmax, RoPE, MatMul. NEON vectorized.
- `sp_quant.cpp/.h` — Q5_K bit-exact dequant (parity-tested vs ggml).
- `sp_tensor.cpp/.h` — Lightweight tensor + arena buffer management.
- `sp_threadpool.cpp/.h` — Persistent thread pool for native forward.
- `speculative_oracle.cpp/.h` — Speculative decoding oracle.
- `qnn_bin_driver.cpp/.h` — QNN .bin driver (loads AI-Hub-compiled .bin splits, executes on V69 HTP). 104 t/s validated on Qwen3-4B w4a16.
- `prime_pe.cpp/.h` — PrimePE integration for engine.
- `tokenizer.cpp/.h` + `vocab.cpp/.h` — GGUF tokenizer (BPE, SPM, WPM).
- `http_server.cpp/.h` — OpenAI-compatible HTTP wrapper (Phase 3 sp-server).
- `cli/main.cpp` — CLI with verbs: `info`, `logits`, `generate`, `chat`, `cache_ppl`, `bench`, `crt_smoke`, `crt_model`, `qnn_bin_serve`.

**Key engine milestones** (from git log):
- Phase 3: sp-server OpenAI HTTP wrapper.
- Phase 4.7: SP-banded KV cache in forward_native.
- Phase 4.8: KQ fused dispatch (per-head through sp_hexagon_cache_kq_matmul_fused).
- Phase 5.0-5.2: QNN .bin driver in sp-engine (104 t/s Qwen3-4B).
- Phase 6-8: HVX logit path, DMA probe, NEON oracle, speculative decode.
- Phase 9: Beast Canyon desktop engine shipped.
- Phase 15.1: hier default, Cauchy default, CRT multi-GPU, SVD entropy.
- Beast Canyon dual-GPU auto-detection (RTX 2060 + Intel UHD).
- MoE expert curriculum + Top-2 speculative prefetch with confidence gate.
- Ternary noise-tail + split K/V residual bits.
- GDN hybrid forward pass for Qwen3.6 (016c04a).

### 1.3 shannon-prime-llama (llama.cpp patch bridge)

Bolts Shannon-Prime into llama.cpp (b8861) via a canonical patch. **NOT the reference** — the engine is. But contains proven patterns for how SP hooks into an inference runtime.

**Engine hooks** (`src/engine/`):
- `llama_sp_fused_kq.cpp/.h` — **Fused decompress-matmul** ggml custom op (Phase 1.6 / Path A.2). Reads K from per-(layer,head) packed-byte buffer, decompresses on-the-fly via `sp_band_dequantize` + VHT2 self-inverse, dot-products against Q. 1.79× faster than vanilla scalar matmul on Snapdragon 8 Gen 1. Thread-split along n_kv. Falls back to on-the-fly compress when no archive pointer.
- `llama_sp_kcap.cpp/.h` — **K-capture custom op** (Phase 1.7). `ggml_map_custom2` op that runs DURING graph compute as a consumer of k_cur. Gathers k_cur values and dispatches `sp_llama_write_k_batch` to populate SP archive directly. Eliminates post_compute K loop. Enables cpy_k bypass → ~150 MB savings at n_ctx=4096.
- `kv_cache.cpp/.h` — Same KvCache/DualKvCache as engine (shared source).
- `gdn_state.cpp/.h` — Same GDN state cache as engine (shared source).

**Bridge hooks** (`src/tools/shannon_prime_llama.h`):
- Constructor init (sp_llama_init), destructor cleanup, post-compute hook.
- Per-model SP context for speculative decoding (role-aware init: draft vs target).
- SHANNON_PRIME_FAST_PATH env var: SP archive is sole K source, fp16 K cache pages stay zero-mapped.
- SHANNON_PRIME_FUSED_KQ env var: route attention through fused decompress-matmul.
- SHANNON_PRIME_K_TERNARY_BANDS, SHANNON_PRIME_FP8, SHANNON_PRIME_HIERARCHICAL env vars.
- SP_BACKEND_HEXAGON dispatch case for Snapdragon cDSP.
- Batched K+V dispatch in post-decode hook.
- Partial-read dispatch through llama_sp_post_compute.
- Draft model suggestion logging.
- PrimePE auto-injection in RoPE path.

**Key llama milestones** (from git log):
- Phase 1.6: Fast-path complete — 3.58× eval on Qwen2.5-Coder-3B IQ2 + 0.5B Q8 draft + spec-decode (05c405d). **NOTE**: Later revealed FUSED_KQ never engages because flash_attn intercepts; 3.58× is from spec-decode alone.
- Phase 1.7: kcap custom op (ab4c65f) — SP archive sourced from k_cur during graph compute.
- Phase 1.7: cpy_k bypass via SHANNON_PRIME_FAST_PATH (966e4eb).
- Per-model SP context for speculative decoding (370fd81).
- Patch-drift CI smoke test on every push.
- CI release pipeline (GitHub Actions, Win-CUDA).
- LM Studio v2.14.0-sp1 shipped.

### 1.4 shannon-prime-comfyui (not in scope for this plan)

ComfyUI integration. Voxtral TTS. Wan video. Not part of the burnhard engine port.

---

## 2. Burnhard Mono-Repo — What's Already Ported

**Location**: `C:\Projects\shannon-prime-burnhard\`

### 2.1 Core Math (`core/`)
- `shannon_prime.c/.h` — Full VHT2, banded quant, Möbius, sqfree, shadow cache. ✅
- `shannon_prime_modelpack.c/.h` — Per-arch presets. ✅
- `shannon_prime_cauchy.c` — Cauchy reset stack. ✅
- `shannon_prime_sqfree.c` — Sqfree compression. ✅
- `shannon_prime_pe.c` — PrimePE. ✅

### 2.2 Desktop Engine (`desktop/engine/src/`)
Ported from shannon-prime-engine. All files present:
- `gguf_loader.cpp/.h` — With Gemma4 KV fold derivation. ✅
- `llama_weights.cpp/.h` — With Gemma4 V-sharing backfill, qwen35moe GDN/MoE, phi3 fused QKV. ✅
- `forward.cpp/.h` — With Gemma4 KV fold, mRoPE mode 8, GDN, MoE, CRT dispatch, conditional GELU. ✅ (Gemma4 runtime crash in progress)
- `kv_cache.cpp/.h` — Full KvCache + DualKvCache + calibration + Cauchy + cold storage + disk I/O. ✅
- `gdn_state.cpp/.h` — GDN recurrent state. ✅
- `engine.cpp/.h` — Full Engine::generate. ✅
- `forward_native.cpp/.h` — ggml-free forward + QNN dispatch. ✅
- `forward_native_context.cpp/.h` — QNN bucketing. ✅
- `sp_kernels_cpu.cpp/.h` — CPU kernels. ✅
- `sp_quant.cpp/.h` — Q5_K dequant. ✅
- `sp_tensor.cpp/.h` — Tensor + arena. ✅
- `sp_threadpool.cpp/.h` — Thread pool. ✅
- `speculative_oracle.cpp/.h` — Spec-decode oracle. ✅
- `qnn_bin_driver.cpp/.h` — QNN .bin driver. ✅
- `prime_pe.cpp/.h` — PrimePE. ✅
- `tokenizer.cpp/.h` + `vocab.cpp/.h` — Tokenizer. ✅
- `http_server.cpp/.h` — sp-server HTTP. ✅
- `cli/main.cpp` — Full CLI. ✅

### 2.3 Beast Canyon Stack (`desktop/beast_canyon/`)
- `sp_beast_canyon.c/.h` — Main orchestrator, dual-GPU auto-detect. ✅
- `sp_shadow_steal.c/.h` — iGPU speculative expert pre-computation. ✅
- `sp_thermal_throttle.c/.h` — 3-zone homeostatic thermal regulator. ✅
- `sp_prefetch_telemetry.c/.h` — Prefetch telemetry. ✅
- `sp_shredder_crt.c/.h` — Fused Q8_0 dequant + CRT residue split. ✅
- `sp_optane.c/.h` — Intel Optane P4800X integration. ✅
- `sp_level_zero.c/.h` — Intel Level Zero backend. ✅

### 2.4 CRT Stack (`desktop/crt/`)
- `sp_crt.c/.h` — Chinese Remainder Theorem multi-GPU decomposition. ✅
- `sp_crt_dispatch.c/.h` — CRT dispatch logic. ✅
- `sp_crt_vulkan.c/.h` — Vulkan CRT compute. ✅
- `sp_crt_cuda.cu` — CUDA CRT kernels. ✅
- `sp_theoretical.c/.h` — Theoretical math module (quaternion, Poincaré, Ricci flow, free energy, stereographic projection, heat equation, Fano plane). ✅
- `sp_prefetch_engine.h` — Prefetch engine. ✅
- `sp_moe_curriculum.h` — MoE expert curriculum + Top-2 speculative prefetch. ✅
- `sp_crt_matmul_mod.comp/.spv` — Vulkan compute shaders. ✅

### 2.5 Vulkan Shaders (`desktop/vulkan/shaders/`)
- `vilenkin.comp`, `knight_predict.comp`, `band_quantize.comp`, `band_dequantize.comp`, `mobius_reorder.comp`. ✅

### 2.6 CUDA (`desktop/cuda/`)
- `shannon_prime_cuda.cu` — CUDA kernels. ✅
- `shannon_prime_fp8.cu` — FP8 CUDA kernels. ✅
- `vht2_wan_ext.cpp` + `setup_wan.py` — PyTorch CUDA bridge. ✅

### 2.7 Mobile (`mobile/`)
- `adreno/shannon_prime_adreno.c` — Adreno backend. ✅
- `hexagon/scaffold/` — Full Hexagon V69 scaffold (FastRPC, HVX kernels, DSP impl). ✅
- `qnn/` — QNN/AI-Hub Python workflows (v69_workflow, attention_block, w4a16 submit). ✅

### 2.8 Bridge (`bridge/`)
- `llama/shannon_prime_llama.c/.h` — llama.cpp bridge (C-level hooks). ✅
- `comfyui/` — ComfyUI nodes. ✅
- `tools/` — Python tools (sp_compress, sp_benchmark, extract_kv, sp_inject_freqs, sp_auto_bands). ✅

### 2.9 Torch (`desktop/torch/`)
- `shannon_prime_torch.py`, `shannon_prime_sqfree.py`, `vht2_cuda_bridge.py`. ✅

### 2.10 Tests (`tests/`)
- `core/test_modelpack.c`, `desktop/test_gguf_loader.cpp`, `desktop/test_engine_smoke.cpp`, `desktop/test_cuda.c`, `desktop/test_vulkan.c`, `desktop/test_vulkan_dualgpu.c`, `mobile/test_adreno.c`, `integration/test_integration.c`. ✅

### 2.11 Docs (`docs/`)
- Full doc set ported: ARCHITECTURE, BACKENDS (per-backend), COMPRESSION-FEATURES, MATHEMATICAL-FOUNDATIONS, CAUCHY-RESET, DISK-TIER-ARCHITECTURE, MODEL-PACK, MODEL-PACK-CALIBRATION, MODALITY-*, INTEGRATION-*, SPECULATIVE-DECODING, BENCH-SPEC-DECODE, TOOLS, TESTING, QUICKSTART, PRIME-ENGINE, MODELS-AND-HARDWARE. ✅

---

## 3. What's NOT Ported / Needs Work

### 3.1 Fused KQ Custom Op (from shannon-prime-llama Phase 1.6)

**Source**: `shannon-prime-llama/src/engine/llama_sp_fused_kq.cpp/.h`
**Status in burnhard**: NOT PORTED. `forward_native.cpp` has a `kq_fused_cpu` reference but uses a different code path (raw SP cache read). The ggml graph path (`forward.cpp`) has CRT matmul dispatch but does NOT have the fused decompress-matmul custom op.

**What it does**: Replaces `ggml_mul_mat(ctx, K, Q)` in attention with a fused op that reads K from the SP compressed archive, decompresses on-the-fly (sp_band_dequantize + VHT2 self-inverse), and dot-products against Q. Eliminates memory bandwidth bottleneck — compressed K is 10× smaller so it stays in L2 cache.

**What needs to happen**: Port `llama_sp_kq_compute` into burnhard as a ggml custom op wired into `build_block`/`build_block_decode` in `forward.cpp`, gated on config flag. The native forward path already has an analogous dispatch in `forward_native.cpp`.

### 3.2 K-Capture Custom Op (from shannon-prime-llama Phase 1.7)

**Source**: `shannon-prime-llama/src/engine/llama_sp_kcap.cpp/.h`
**Status in burnhard**: NOT PORTED. The engine writes K/V to KvCache via explicit `write()` calls in the decode loop, but does not have the ggml graph-level kcap op that captures k_cur during graph compute.

**What it does**: A `ggml_map_custom2` op that runs DURING graph compute as a consumer of k_cur. Gathers k_cur values and dispatches `sp_llama_write_k_batch` into the persistent SP archive. This enables the cpy_k bypass (Phase 1.7 fast-path) — fp16 K cache pages stay zero-mapped, recovering ~150 MB at n_ctx=4096.

**What needs to happen**: Port into burnhard's `forward.cpp` as a custom op in the attention build, gated on a fast-path config flag. The KvCache already supports the write interface; we just need the graph-level capture.

### 3.3 Fast-Path (cpy_k bypass, ~150 MB savings)

**Source**: `shannon-prime-llama` Phase 1.7 (966e4eb, 7432c9f)
**Status in burnhard**: NOT PORTED. No `SHANNON_PRIME_FAST_PATH` equivalent.

**What it does**: When active, SP compressed K archive is the sole source of truth. The fp16 ggml K cache is still allocated (graph build requires it) but the K_cur copy step is skipped. Zero pages stay uncommitted by the kernel → ~150 MB savings on a 3B model at n_ctx=4096.

**What needs to happen**: Wire as a config flag in Engine. Gate the cpy_k step in `forward.cpp` build_block. Depends on kcap (3.2) being ported first.

### 3.4 Gemma4 Forward Pass Runtime Crash (ACTIVE BUG)

**Status**: IN PROGRESS (Task #24). KV fold code written in both `build_block` and `build_block_decode`. Build compiles clean. Runtime crashes with `GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2)` at ggml.c:3648.

**Diagnostic fprintf added but not yet rebuilt/tested**. The kv_fold reshape logic looks correct on paper (Q→[256,32,n], K pre-fold→[512,8,n], V→[256,16,n], all element counts match). The crash may be from a different reshape elsewhere in the graph, or from the single-layer shape-inference path (~line 1777), or from build_block_decode.

**What needs to happen**: Rebuild with diagnostic, run, inspect which reshape actually fails. Fix the reshape. Test with `sp-engine.exe logits --model gemma-4-31B-it-Q4_K_M.gguf "The capital of France is"`.

### 3.5 Speculative Decoding Integration

**Source**: shannon-prime-llama (per-model SP context, role-aware init, draft-hint bridge)
**Status in burnhard**: `speculative_oracle.cpp/.h` exists with the oracle logic. CLI has `--draft` and spec-decode flags. But the **per-model SP context** (separate SP state for draft vs target model) and **role-aware init** are NOT ported.

**What needs to happen**: Port the per-model context concept so that when running spec-decode with draft+target, each model gets its own SP compression state with appropriate band configs (draft model may use lighter compression or no compression).

### 3.6 Hexagon Backend Integration in Engine Forward Path

**Source**: shannon-prime-engine Phase 4.8 (8bd7379, 5d950ad)
**Status in burnhard**: The Hexagon scaffold (`mobile/hexagon/`) is fully ported. The `forward_native.cpp` has QNN HTP dispatch references. But the **FastRPC dispatch wiring from the ggml graph path** (sp_hexagon_cache_kq_matmul_fused through `forward.cpp`) is NOT connected — it only exists in the native forward path.

**What needs to happen**: Wire Hexagon dispatch as an alternative backend for the fused KQ op (3.1) in the ggml graph path. The native forward path already has this; the ggml path needs it for the phone deployment target.

### 3.7 CI/CD Pipeline

**Source**: shannon-prime-llama has GitHub Actions release pipeline (6e54f98)
**Status in burnhard**: NO CI. No GitHub Actions, no patch-drift test, no release pipeline.

**What needs to happen**: Set up GitHub Actions for burnhard. At minimum: build (no-CUDA + CUDA), test_engine_smoke, test_gguf_loader. Bonus: release pipeline with artifact naming.

### 3.8 Model Validation & Calibration

**Status**: phi3 is the only CALIBRATED arch. 7 others are PROVISIONAL.
- qwen3 edge-fail at +5.14% (above 5% threshold).
- gemma3, gemma4, qwen35moe, mistral3, granite, llama all PROVISIONAL.

**What needs to happen**: Run cache_ppl benchmarks on each arch with the engine. Update MODEL-PACK-CALIBRATION.md ledger. Priority: qwen2 (most common), gemma4 (current work), qwen35moe (hybrid arch).

### 3.9 FP8/FP4 Integration Gap

**Source**: shannon-prime core has `shannon_prime_cuda.h` with fp8/fp4 kernels. shannon-prime-llama has `SHANNON_PRIME_FP8` env var bridge.
**Status in burnhard**: `desktop/cuda/shannon_prime_fp8.cu` exists. But the FP8 path is **PyTorch-only** — no integration into the ggml engine forward pass or the native forward pass.

**What needs to happen**: Wire FP8 as an alternative to the banded quantization path in KvCache. Gate on config flag. This is a research/future item — banded quant is the proven ship path.

### 3.10 Fisher Information Research

**Source**: Referenced in FUTURE-WORK.md.
**Status**: Not started. Theoretical framework for optimal bit allocation using Fisher information matrix of the VHT2 transform.

---

## 4. Current State Summary

| Component | Source | Burnhard | Status |
|-----------|--------|----------|--------|
| Core math (VHT2, band quant, Möbius, sqfree) | shannon-prime | ✅ | Complete |
| Modelpack presets | shannon-prime | ✅ | Complete |
| Cauchy reset (Zeta + Mertens + Ricci) | shannon-prime | ✅ | Complete |
| PrimePE | shannon-prime | ✅ | Complete |
| GGUF loader (all archs + gemma4 KV fold) | engine | ✅ | Complete |
| Weight binding (all archs + gemma4 V-share) | engine | ✅ | Complete |
| Forward pass (ggml graph, all archs) | engine | ✅ | Complete (gemma4 runtime bug) |
| Forward native (ggml-free, Qwen2) | engine | ✅ | Complete |
| KvCache + DualKvCache | engine | ✅ | Complete |
| GDN state cache | engine | ✅ | Complete |
| CPU kernels (RMSNorm, SiLU, etc.) | engine | ✅ | Complete |
| Q5_K dequant | engine | ✅ | Complete |
| QNN .bin driver | engine | ✅ | Complete |
| sp-server HTTP | engine | ✅ | Complete |
| Speculative oracle | engine | ✅ | Complete |
| Tokenizer/vocab | engine | ✅ | Complete |
| Beast Canyon (shadow steal, thermal, shredder, optane, L0) | engine | ✅ | Complete |
| CRT (decomposition, Vulkan, CUDA, dispatch) | engine | ✅ | Complete |
| Theoretical math (quaternion, Poincaré, Ricci, Fano) | engine | ✅ | Complete |
| MoE curriculum + Top-2 prefetch | engine | ✅ | Complete |
| Ternary noise-tail + split K/V bits | engine | ✅ | Complete |
| Hexagon scaffold (FastRPC, HVX, DSP) | shannon-prime | ✅ | Complete |
| QNN/AI-Hub workflows | shannon-prime | ✅ | Complete |
| Vulkan shaders | engine | ✅ | Complete |
| CUDA + FP8 kernels | engine | ✅ | Complete (FP8 not wired to engine) |
| **Fused KQ decompress-matmul** | **llama** | **❌** | **Not ported** |
| **K-capture (kcap) custom op** | **llama** | **❌** | **Not ported** |
| **Fast-path (cpy_k bypass)** | **llama** | **❌** | **Not ported** |
| **Per-model SP context (spec-decode)** | **llama** | **❌** | **Not ported** |
| **Hexagon dispatch in ggml graph path** | **engine** | **❌** | **Only in native forward** |
| **CI/CD pipeline** | **llama** | **❌** | **Not set up** |
| Gemma4 forward pass | new work | 🔧 | In progress (crash) |
| FP8 engine integration | gap | ❌ | PyTorch only |
| Model calibration (7 archs) | gap | ❌ | Only phi3 calibrated |

---

## 5. Priority Work Queue

### P0 — Immediate (fix what's broken)

1. **Fix Gemma4 forward pass crash** (Task #24)
   - Rebuild with diagnostic fprintf, run, find the failing reshape
   - Fix the assert, validate logits output
   - Test: `sp-engine.exe logits --model gemma-4-31B-it-Q4_K_M.gguf "The capital of France is"`

### P1 — Port proven llama features into engine

2. **Port Fused KQ decompress-matmul** (from llama_sp_fused_kq.cpp)
   - New files: `desktop/engine/src/sp_fused_kq.cpp/.h`
   - Wire as ggml_map_custom2 in build_block attention, gated on config.fused_kq
   - Port the decompress_one_row / compress_then_decompress inner loops
   - Port the thread-split-along-n_kv dispatch pattern

3. **Port K-capture custom op** (from llama_sp_kcap.cpp)
   - New files: `desktop/engine/src/sp_kcap.cpp/.h`
   - Wire as ggml_map_custom2 consuming k_cur in build_block, gated on config.fast_path
   - Port the per-head gather + sp_llama_write_k_batch dispatch

4. **Wire Fast-Path (cpy_k bypass)**
   - Add config.fast_path flag
   - Gate the K cache copy in forward.cpp build_block
   - Verify ~150 MB savings on 3B model at n_ctx=4096

5. **Port per-model SP context for spec-decode**
   - Extend Engine to carry separate SP state per loaded model
   - Draft model gets lighter compression or bypass
   - Target model gets full SP compression

### P2 — Integration & validation

6. **Wire Hexagon dispatch in ggml graph path**
   - Connect sp_hexagon_cache_kq_matmul_fused as alternative backend for fused KQ op
   - Gate on SP_BACKEND_HEXAGON build flag
   - Test on S22U via ADB push

7. **CRT expert-split Phase 2: GPU dispatch kernels** (Task #25)
   - CUDA kernel for CRT residue split → multi-GPU scatter → gather
   - Wire through Beast Canyon orchestrator

8. **Run calibration on priority archs**
   - qwen2 (most deployed), gemma4 (current work), qwen35moe (hybrid)
   - Run cache_ppl with SP compression, measure PPL delta
   - Update MODEL-PACK-CALIBRATION.md

### P3 — Production hardening

9. **CI/CD pipeline**
    - GitHub Actions: build (no-CUDA + CUDA), test_engine_smoke, test_gguf_loader
    - Release artifacts with version tags

10. **FP8 engine integration** (research)
    - Wire FP8 quantization as alternative to banded quant in KvCache
    - Gate on config flag, CUDA-only initially

11. **Fisher information research**
    - Optimal bit allocation via Fisher information matrix of VHT2 transform
    - Would replace the current heuristic band configs with information-theoretically optimal ones

---

## 6. Build Commands Reference

**No-CUDA build (fast, ~30s):**
```
cd C:\Projects\shannon-prime-burnhard\build_nocuda
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DSP_WITH_CUDA=OFF -DSP_WITH_BEAST=ON
ninja
```

**Beast Canyon build (with CUDA, ~3min):**
```
cd C:\Projects\shannon-prime-burnhard\build_beast
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DSP_WITH_CUDA=ON -DSP_WITH_BEAST=ON -DSP_BEAST_CUDA=ON -DSP_BEAST_LEVEL_ZERO=ON
ninja
```

**GOTCHAS:**
- Must pass `-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl` or MinGW gets picked up
- Must source vcvarsall BEFORE cmake so rc.exe is on PATH
- ggml version: 0.11.0 (b8763), has fused ggml_gated_delta_net op

**Binary output**: `build_*/bin/sp-engine.exe` (+ ggml.dll + ggml-cpu.dll)

**Test model**: `D:\Files\Models\lmstudio-community\gemma-4-31B-it-GGUF\gemma-4-31B-it-Q4_K_M.gguf`

---

## 7. Architecture Diagram

```
C:\Projects\shannon-prime-burnhard\
├── core/                          # SP math library (C)
│   ├── include/shannon_prime.h    #   VHT2, band quant, Möbius, sqfree, shadow
│   └── src/                       #   .c implementations
├── desktop/
│   ├── engine/src/                # THE engine (C++, ggml graph + native forward)
│   │   ├── forward.cpp            #   ggml graph builder (all archs)
│   │   ├── forward_native.cpp     #   ggml-free forward (Qwen2 + QNN)
│   │   ├── kv_cache.cpp           #   SP compressed KV + DualKvCache
│   │   ├── llama_weights.cpp      #   Weight binding (all archs)
│   │   ├── gguf_loader.cpp        #   GGUF metadata
│   │   ├── engine.cpp             #   Top-level generate loop
│   │   └── cli/main.cpp           #   CLI (info/logits/generate/chat/cache_ppl)
│   ├── beast_canyon/              # Beast Canyon NUC orchestrator
│   ├── crt/                       # CRT multi-GPU decomposition
│   ├── cuda/                      # CUDA kernels
│   ├── vulkan/                    # Vulkan compute shaders
│   ├── l0/                        # Intel Level Zero
│   └── torch/                     # PyTorch bridge
├── mobile/
│   ├── adreno/                    # Adreno GPU backend
│   ├── hexagon/scaffold/          # Hexagon V69 DSP (FastRPC + HVX)
│   └── qnn/                       # QNN/AI-Hub workflows
├── bridge/
│   ├── llama/                     # llama.cpp C bridge hooks
│   ├── comfyui/                   # ComfyUI nodes
│   └── tools/                     # Python tools
├── tests/                         # Test suites
├── docs/                          # All documentation
├── scripts/                       # Build/bench scripts
└── vendor/ggml/                   # ggml 0.11.0 (b8763)
```
