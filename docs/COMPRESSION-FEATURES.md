# Shannon-Prime Compression Features

A detailed reference for every compression feature in the Shannon-Prime system. Each section covers what the feature does, why it exists, where it lives in the code, and what its measured impact is.

---

## 1. VHT2 — The Vilenkin-Hartley Transform

**What:** An orthonormal staged transform that converts spatial-domain KV vectors into spectral coefficients. Self-inverse: `VHT2(VHT2(x)) = x`.

**Why:** RoPE imprints predictable spectral structure on K vectors. In the spectral domain, energy concentrates in a few coefficients and decays toward the tail. This concentration is what makes banded quantization effective.

**Where:** `sp_vht2_forward_f32()` and `sp_vht2_forward_f16()` in `core/shannon_prime.h`.

**How it works:** For dimension n factoring into primes {p_1, p_2, ...}, VHT2 applies one p × p Hartley stage per prime factor, each normalized by 1/√p. The Hartley kernel is `cas(x) = cos(x) + sin(x)`. At n = 2^k (the common case), all stages are p=2 butterflies: `a' = (a+b)/√2, b' = (a−b)/√2`.

**Self-inverse property:** The transform matrix V satisfies V·V = N·I (where N = product of primes), and the 1/√p normalization at each stage absorbs this factor. The round-trip error is below float32 ULP.

**Impact:** Enables the entire compression stack. Without VHT2, there is no spectral concentration to exploit.

---

## 2. Möbius Squarefree-First Reorder

**What:** Permutes VHT2 coefficients so that squarefree indices (those where the Möbius function μ(n) ≠ 0) appear first in the vector.

**Why:** Squarefree indices carry 61.4% of the signal energy at N=210. By pushing them to the front, the highest-bit bands in the banded quantizer contain the most important coefficients.

**Where:** `sp_mobius_mask_t` struct, `sp_mobius_reorder()`, `sp_mobius_unreorder()` in `core/shannon_prime.h`. Scratch-free hot-path variants: `sp_mobius_reorder_ex()`, `sp_mobius_unreorder_ex()`.

**How it works:** Build a permutation array where squarefree indices come first (sorted by |μ(n)|), followed by non-squarefree indices. The Möbius function values μ(0..n-1) are computed at init time. Reorder is a gather; unreorder is a scatter.

**Impact:** +0.14 PPL improvement at identical bit budget. Cross-platform invariant: K correlation 0.997 on both hd=128 and hd=64.

---

## 3. Banded Quantization

**What:** Splits the (optionally reordered) VHT2 coefficient vector into N equal bands and quantizes each band independently with its own fp16 scale factor and integer bit depth.

**Why:** VHT2 energy decay means early bands (high energy) need more bits and late bands (noise tail) need fewer. Uniform quantization wastes bits on the tail; banded quantization matches bit allocation to signal density.

**Where:** `sp_band_config_t`, `sp_band_quantize()`, `sp_band_dequantize()`, `sp_band_dequantize_partial()` in `core/shannon_prime.h`.

**Ship allocation (hd=128, 4 bands):**

| Band | Coefficients | Bits | Bytes (data + scale) | Signal Content |
|---|---|---|---|---|
| 0 | 32 | 5 | 22 | Highest-energy (fundamental frequencies) |
| 1 | 32 | 5 | 22 | High-energy harmonics |
| 2 | 32 | 4 | 18 | Mid-energy detail |
| 3 | 32 | 3 | 14 | Noise tail |
| **Total** | **128** | — | **76** | **From 256 bytes fp16 = 3.37×** |

**Key findings from validation:**
- 5/5/4/3 **beats lossless fp16** by 0.04% (spectral regularization effect)
- 4/4/4/4 is off the Pareto frontier (4/4/4/3 is strictly better)
- 3-bit floor: 2-bit on any band is catastrophic
- Flat allocation beats banded for V vectors (no exceptions found)

---

## 4. Ternary Noise-Tail Quantization

**What:** Optionally quantizes a band to ternary {-1, 0, +1} (≈1.58 bits/coefficient information content) instead of the specified bit depth.

**Why:** The strange-attractor analysis predicts that the noise tail (typically band 3 in a 5/5/4/3 configuration) is statistically indistinguishable from ternary. Storing it at ~1.58 bpp instead of 3 bpp saves space with no measurable quality loss.

**Where:** `sp_band_config_t.ternary_band_mask`, `sp_band_config_init_ext()` in `core/shannon_prime.h`.

**Implementation:** Current packing uses 2 bpp (4 trits/byte). Tighter 5-trits-per-byte packing (true 1.58 bpp via base-3 encoding) is a future optimization that doesn't change the API. Ternary bands share the fp16 scale header with regular bands.

**Impact:** The 5/5/4/1.58 configuration matches 5/5/4/3 quality with additional compression on band 3.

---

## 5. Progressive Band Reads (Partial Reconstruction)

**What:** Reconstructs KV vectors using only the first N bands, treating unread bands as zero coefficients.

**Why:** Energy concentration in early bands means partial reconstruction is surprisingly effective. This primitive enables System 1/2 switching and the disk-tier architecture.

**Where:** `sp_band_dequantize_partial()`, `sp_shadow_read_k_partial()`, `sp_shadow_read_v_partial()`, `sp_shadow_cache_load_partial()` in `core/shannon_prime.h`.

**Measured correlation (hd=128, ship path):**

| Bands Read | Bytes | Correlation | Use Case |
|---|---|---|---|
| 0 | 0 | 0.00 | Sentinel (all-zero) |
| 1 | 22 | 0.30 | Coarse screening |
| 2 | 44 | 0.86 | System 1 fast path |
| 3 | 60 | 0.88 | Diminishing returns |
| 4 | 76 | 0.99 | Full reconstruction |

**Cost saving:** Reduced dequantize math (skipping unread bands' bit-unpack inner loop) + reduced inverse VHT2 cost (zero coefficients collapse butterfly contributions).

---

## 6. Sqfree Padding

**What:** Pads head_dim to the next squarefree-factorable number that factors into small primes {2, 3, 5, 7, 11, 13}.

**Why:** The multi-prime VHT2 basis is richer than the power-of-2 basis. More prime factors → more Kronecker sub-projections → better prediction and skeleton extraction.

**Where:** `sp_sqfree_pad_dim()`, `sp_sqfree_pad_f32()`, `sp_sqfree_unpad_f32()` in `core/shannon_prime.h`.

**Known pad dimensions:**
- hd=64 → 66 = 2·3·11
- hd=128 → 154 = 2·7·11
- hd=256 → 330 = 2·3·5·11

**Padding method:** Mean-fill. The extra positions are filled with the mean of the original vector. This preserves the DC component and doesn't inject artificial high-frequency content.

---

## 7. Knight Skeleton + Möbius CSR Predictor

**What:** Extracts the top-K squarefree indices by variance as a "skeleton" and uses the Möbius function's divisor relationships to predict the remaining "residual" coefficients from the skeleton.

**Why:** The Möbius function encodes multiplicative relationships between indices. If skeleton coefficient at index d is known, and n/d is also in the skeleton, then μ(d) tells us the sign and weight of the prediction for index n. This is a genuine predictor (not just a heuristic) on sqfree-padded Vilenkin bases: r=0.40–0.58.

**Where:** `sp_knight_mask_t`, `sp_knight_mask_init()` in `core/shannon_prime.h`.

**CSR (Compressed Sparse Row) representation:** For each residual index r, the prediction is:
```
pred[r] = Σ μ(d) · skel_vals[slot(n/d)]
```
over divisors d of (r+1) where μ(d) ≠ 0 and (n/d)-1 is in the skeleton. The CSR arrays store offsets, skeleton slot indices, and μ signs for O(1) per-residual prediction.

**Skeleton size:** Default L/2 (half the padded dimension). Variance-ranked selection (after calibration) significantly outperforms index-order selection.

---

## 8. SU(2) Spinor Sheet Bit

**What:** A 1-bit correction per residual position that records which sheet of the SU(2) double cover the original coefficient occupies.

**Why:** The VHT2 spectrum has a phase ambiguity — the Möbius predictor can predict the magnitude but not always the sign of the residual. The spinor bit resolves this ambiguity. It is the first architectural feature that *shifts* the Pareto frontier (not merely moves along it).

**Where:** `sp_knight_mask_t.use_spinor`, residual encoding in `core/shannon_prime_sqfree.c`.

**Impact:** On Qwen3-8B Q8 hd=128, sqfree+spinor achieves K+μ+3bit+spinor 3/3/3/3/3 at PPL 7.32 @ 3.3× — matching MOBIUS default 7.31 @ 2.6×, with +27% additional compression.

**Cost:** 1 bit per residual position. At n_res=77 (hd=128 pad=154 L/2 skeleton), that's ~10 bytes per KV vector.

---

## 9. Hierarchical Vilenkin Predictor

**What:** Uses the Kronecker product structure of the multi-prime VHT2 basis to build a small core skeleton (the sub-projection over a subset of primes) and a calibrated linear map W to predict all remaining coefficients.

**Why:** The Knight skeleton uses L/2 coefficients (~50%). The hierarchical predictor uses a Kronecker sub-projection (~9%) and makes up the difference with a learned predictor. Much smaller skeleton → much higher compression.

**Where:** `sp_hier_predictor_t`, `sp_hier_cache_t` in `core/shannon_prime.h`.

**Example (hd=128, pad=154 = 2·7·11):**
- Kronecker sub-projection Z/2Z × Z/7Z → 14 skeleton coefficients (9.1%)
- Linear predictor W: 14 × 140 fp16 matrix per (layer, head) slot
- Calibrated via ridge regression during first prefill

**Storage per position:**
- Skeleton: 14 × 5 bits = 70 bits (banded at 5,5)
- Residual: 140 × 2 bits = 280 bits
- Total: 350 bits = 43.75 bytes (from 308 bytes fp16 = 7.0× compression)

**Aggressive configuration:** 14 × 4 bits + 1-bit residual = 196 bits = 24.5 bytes = 12.6× compression.

**Calibration:** `sp_hier_calibrate_begin/feed/end`. Sticky-EMA variant available for online adaptation: `sp_hier_calibrate_end_blend(hp, W_prev, keep_frac)`.

---

## 10. Variance-Ranked Calibration

**What:** Adaptive reordering of VHT2 coefficients based on empirically measured per-coefficient variance from warmup data.

**Why:** The default Möbius reorder is structurally optimal (based on number theory), but data-adaptive ordering can be even better for specific models. High-variance coefficients should land in high-bit bands; low-variance coefficients in low-bit bands.

**Where:** `sp_shadow_cache_t.use_var_reorder`, `sp_shadow_calibrate_begin/feed/end()` for ship path. `sp_sqfree_calibrate_begin/feed/end()` for sqfree path.

**Lifecycle:**
1. `calibrate_begin()` — allocate per-coefficient running sum and sum-of-squares accumulators
2. `calibrate_feed(vec)` — for each warmup vector: VHT2 → accumulate
3. `calibrate_end()` — compute variance, sort, rebuild permutation arrays

After calibration, write/read paths use the empirical ordering. If calibration is never called, the default Möbius/index ordering is used — still correct, just not data-adapted.

---

## 11. System 1/2 Dual-Cache Switching

**What:** Two cache modes operating on the same compressed data. System 1 reads partial bands (fast, approximate). System 2 reads all bands (slow, exact). Switching is driven by attention entropy.

**Why:** 99% of attention queries can be answered from partial data (band 0+1 gives 0.86 correlation). Only when the softmax distribution is flat (many positions contribute equally) does full reconstruction matter.

**Where:** `DualKvCache` wrapper in `shannon-prime-engine/src/kv_cache.h`.

**Mechanism:**
- System 1: Read band 0 only (22 bytes/position). Compute approximate Q·K^T. If the top-scoring position has sufficient margin over second-place, accept.
- System 2: Read all bands (76 bytes/position). Full-fidelity attention.
- Switching criterion: attention entropy threshold (tunable per model).

**Impact:** System 1 reads 71% less data per position. On memory-bound hardware (phones, laptops), this translates directly to reduced latency.

---

## 12. Disk-Tier Progressive Loading

**What:** Band-major file format (v3) that enables reading only the first N bands from disk without touching the remaining bands' bytes.

**Why:** When the KV cache exceeds RAM, bands can be stored on different tiers: band 0 in RAM, band 1 on NVMe, bands 2–3 on Optane or network. System 1 reads from RAM only; System 2 pulls additional bands from storage.

**Where:** `SP_CACHE_VERSION_BAND_MAJOR = 3`, `sp_shadow_cache_save()`, `sp_shadow_cache_load_partial()` in `core/shannon_prime.h`.

**v3 file layout:**
```
[64-byte header]
[band 0: all heads × all positions × band_0_stride]   ← contiguous
[band 1: all heads × all positions × band_1_stride]   ← contiguous
[band 2: ...]
[band 3: ...]
```

**IO win:** Reading bands [0, max_bands) is a single contiguous read on NVMe. On Optane (byte-addressable via DAX), it's a direct memory reference — sub-millisecond.

---

## 13. Cauchy Reset (Ricci Sentinel + Mertens Oracle)

**What:** Detects when compression errors have accumulated beyond a safe threshold and re-anchors the cache.

**Why:** Over long decode chains, small per-token compression errors compound. The spectral signature of this compounding is detectable in the p=3 band energy: when it drifts from its calibrated baseline, the compressed past no longer determines the correct future.

**Where:** `sp_ricci_sentinel_t`, `sp_mertens_oracle_t`, `sp_cauchy_ctrl_t` in `core/shannon_prime.h`.

**Ricci sentinel:** Tracks EMA of (current_p3_energy / calibrated_p3_energy). Threshold is model-size-dependent: 0.05 for 1B models, 0.15 for 8B+.

**Mertens oracle:** M(n) = Σ μ(k) oscillates with a half-period of 200–500 tokens at typical context scales, driven by the spectral decomposition of the zeta function. The oracle pre-computes a risk schedule and provides O(1) lookup.

**Controller modes:**
- Mode 0: Off
- Mode 1: Fixed-N (reset every N tokens)
- Mode 2: Dynamic (Ricci triggers, Mertens schedules)

---

## 14. PrimePE — Lattice-Aligned Positional Encoding

**What:** Frequency multipliers that blend multiplicative-lattice frequencies into standard geometric RoPE.

**Why:** Standard RoPE's geometric spacing doesn't match the multiplicative structure of the integers. PrimePE corrects this frequency mismatch at zero runtime cost (the multipliers are computed once and baked into the model's frequency tensor).

**Where:** `sp_prime_pe_freq_factors()` in `core/shannon_prime.h`.

**Impact:**
- alpha=0.17: −0.6% PPL (deployment-robust)
- alpha=0.22: −0.8% PPL (slightly less robust)
- Universal across architectures and quantization levels
- Zero retraining required

---

## 15. K-Corr Scaling Law

**What:** An empirical law connecting KV-cache reconstruction fidelity to downstream perplexity.

**Where:** `sp_predicted_ppl_ratio()`, `sp_is_pareto_viable()`, `sp_min_k_corr_for_budget()` in `core/shannon_prime.h`.

**The law:** `log(PPL/base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)`

**Exponent interpretation:**
- Quadratic in K-error: K·Q·V bilinearity squares the error
- Sub-linear in params: bigger models are more robust
- Super-linear in bits: Q4 amplifies K error ~2.8× vs Q8

**Use:** Pre-bench filter. Before running a full perplexity evaluation, compute the predicted PPL ratio. If it exceeds your quality budget, skip the configuration.

---

## 16. Zero-Copy Pointers and Stateless Operation

**What:** Each KV vector is compressed and stored independently. No sequential dependencies between positions. Optane/mmap backends provide direct pointers into mapped files.

**Why:** Stateless compression enables: arbitrary eviction (drop any position without affecting others), parallel compression (any thread can write any position), reordering (positions can be stored in any order), and zero-copy access (the shredder reads directly from the mapped GGUF file).

**Where:** All shadow cache write/read functions operate on `[layer][head][pos]` without referencing adjacent positions. The Optane reservoir's `sp_optane_tensor()` returns a direct pointer into the mmap'd file.

**Impact:** Enables Beast Canyon's pipeline (read from Optane, shred directly into LLC) and the phone's disk-tier architecture (evict positions freely as memory pressure changes).

---

## 17. Model-Pack Registry

**What:** Per-architecture compression defaults stored as small records. Resolves from GGUF metadata at model load time.

**Why:** Different architectures have different spectral characteristics. A configuration that works well for Llama may not be optimal for Qwen MoE. The model-pack registry provides known-good defaults per architecture family.

**Where:** `core/shannon_prime_modelpack.h`, `docs/MODEL-PACK.md`, `docs/MODEL-PACK-CALIBRATION.md`.

**Status:** phi3 CALIBRATED (+2.44% PPL, within budget). qwen3 edge-fail (+5.14%). 7 architectures PROVISIONAL.

**CLI:** `--model-preset auto` reads `general.architecture` from GGUF and applies the matching preset. Explicit user flags override preset defaults.

---

## 18. 1D-Circle Granite Reconstruction (ComfyUI)

**What:** Strict 1D-circle reconstruction for the most stable DiT blocks (tier 0, "Granite" blocks) in video generation models.

**Why:** In Wan 2.x, blocks L00–L03 have cos_sim > 0.95 across 10 denoising steps. Their self-attention outputs lie on a 1D circle in the block's output space. Rather than caching the full output tensor, cache the 1D projection and reconstruct from it.

**Where:** `shannon-prime-comfyui/src/`, enabled via `enable_one_dim_granite` flag.

**Impact:** Further memory reduction for the most stable blocks. Combined with block-skip caching, Wan 2.2 5B achieves 4.6× step speed.

---

## 19. Cross-Tier Energy Borrowing (ComfyUI)

**What:** Allows higher-stability tiers to "borrow" energy budgets from lower-stability tiers when the lower tiers are disabled.

**Why:** When tier 2/3 blocks are not cached (volatile blocks), their unused energy budget can be redistributed to tier 0/1 blocks, allowing longer cache windows without quality degradation.

**Where:** `shannon-prime-comfyui/src/`, enabled via `enable_cross_tier_borrow` flag.

---

## 20. Per-Token VHT2 Skeleton Fraction (ComfyUI)

**What:** Varies the fraction of VHT2 skeleton coefficients retained per token based on the token's position in the denoising schedule.

**Why:** Early denoising steps (high noise) need less precision than late steps (fine detail). Adapting the skeleton fraction saves memory and compute without affecting final output quality.

**Where:** `shannon-prime-comfyui/src/`, enabled via `enable_per_token_skeleton` flag.

---

## 21. Speculative Decoding Integration

**What:** Per-model SP compression for speculative decoding. Draft model gets its own shadow cache (typically more aggressive settings). Target model gets a separate cache at ship quality.

**Why:** Draft errors are recoverable on target verification. The draft model can tolerate higher compression because incorrect draft tokens are rejected and regenerated. The target model needs ship quality because its outputs are the final tokens.

**Where:** Environment variable schema: `SHANNON_PRIME_ROLE=draft` or `SHANNON_PRIME_ROLE=target` for per-model configuration.

**Validated:** Qwen2.5-Coder-3B (target) + 0.5B (draft) + --draft 8 = 43.72 t/s on phone CPU (3.58× vs vanilla).

---

## 22. FP8 and FP4 Quantization Paths

**What:** Alternative banded quantization using FP8 (E4M3FN) or FP4 (MXFP4) instead of integer quantization.

**Why:** FP8 has higher dynamic range than int8 for smooth distributions (V cache). FP4 is native on Blackwell tensor cores (sm_120+).

**Where:** `backends/cuda/shannon_prime_fp8.cu`, compile-time gated via `SP_FP8=1` and `SP_FP4=1`.

**Status:** CUDA implementation shipped. CPU fp8 fallback not yet implemented. Backends without fp8 path log a warning and fall back to int quantization.
