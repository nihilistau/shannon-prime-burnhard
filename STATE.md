# Shannon-Prime BurnHard — Project State

Last updated: 2026-05-09 (GDN fused op landed)

## Overall Goals

1. **Custom distributed inference engine** — phone → multi-GPU heterogeneous pipeline
2. **Shannon-limit KV cache compression** via VHT2 + Möbius + sqfree/spinor
3. **Beast Canyon orchestrator** — CRT multi-GPU splitting MoE experts across RTX 2060 + Intel UHD 750
4. **Phone sidecar** — S22 Ultra via USB-C/ADB for speculative drafting
5. **Dynamic feature gating** — Full Pulse / Standalone Desktop / Non-Optane Fallback / Standalone Mobile
6. **Target models**: Qwen3.6-35B-A3B, Qwen3.6-27B, Gemma4-31B-it

## Repo Structure

- `shannon-prime-bernhard` — theory vault (papers, proofs, LaTeX)
- `shannon-prime-burnhard` — production mono-repo:
  - `/core` — SP math (VHT2, Möbius, CRT, band quant, sqfree, Cauchy, PrimePE, modelpack)
  - `/desktop` — Engine + Beast Canyon + CRT multi-GPU + CUDA + Vulkan + Level Zero
  - `/mobile` — Adreno NEON + Hexagon FastRPC/HVX + QNN HTP + Halide + freethedsp
  - `/bridge` — llama.cpp integration + ComfyUI integration + compression tools
  - `/vendor` — ggml (0.11.0) + cpp-httplib
- `shannon-prime-llama` — custom llama.cpp with LM Studio build + expert splitting (https://github.com/nihilistau/shannon-prime-llama)

## Checklist

### Completed
- [x] Bernhard + burnhard directory structure created
- [x] 2,172 files ported from real SP codebase (not stubs)
- [x] Full CMakeLists.txt with all backend options
- [x] Build working: MSVC/VS2019 + CUDA 13.2 + Ninja + AVX-512
- [x] KV cache smoke test passing
- [x] Prefill + cache_ppl on Qwen3.6-35B-A3B (10/40 attn layers with SP compression)
- [x] Beast Canyon orchestrator init: RTX 2060 + UHD 750 + AVX-512 + Optane reservoir
- [x] ggml updated to 0.11.0 (from llama.cpp b8763)
- [x] Non-CUDA build verified clean (73/73, zero errors)
- [x] Runtime verified on Qwen2.5-Coder-0.5B
- [x] GDN fused op: ggml_gated_delta_net replaces chunked prefill (caca3ea)
- [x] All 30 GDN layers computing on Qwen3.6-35B-A3B (zero crashes, zero NaN)

### In Progress
- [ ] **Q4_K centered nibbles**: Implement `(q - 8)` bit-slicing in /core to fix repetitive tokens
- [ ] **mRoPE Mode 8**: 3-segment partitioned rotation for Qwen 3.6

### Next Up
- [ ] **CRT multi-GPU expert splitting**: Wire RTX 2060 (M1) + UHD 750 (M2) dispatch in Beast Canyon
- [ ] **Phone sidecar**: Wire ADB bridge into Beast Canyon orchestrator
- [ ] **Validate 27B + 31B**: Run Qwen3.6-27B and Gemma4-31B through full pipeline
- [ ] **Storage topology**: Plan for 32GB NVMe + P4800X 375GB Optane U.2

### Future
- [ ] Full Pulse mode: i9 + RTX + UHD + Optane + S22 all active
- [ ] Standalone Desktop fallback (internal speculation if S22 detached)
- [ ] Non-Optane fallback (NVMe/RAM if reservoir missing)
- [ ] Standalone Mobile mode (phone runs independently)
- [ ] fp8/fp4 integration
- [ ] Fisher information research
