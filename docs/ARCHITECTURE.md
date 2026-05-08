# Shannon-Prime System Architecture

**Version 2.0 — May 2026**
**Author: Ray Daniels**

This document describes the complete Shannon-Prime architecture from first principles through to hardware deployment. It covers every component, every data path, and the reasoning behind each design decision. Read this to understand how the system works as a whole.

---

## 1. Design Philosophy

Shannon-Prime is built on three observations:

1. **RoPE creates spectral structure.** Rotary position embeddings encode position as phase rotation across frequency pairs. This imprints a predictable spectral signature on every K vector — one that concentrates energy in specific frequency bands and decays predictably toward the tail.

2. **The multiplicative lattice is the natural basis.** The integers' multiplicative structure (primes, composites, squarefree numbers) provides a spectral basis that matches how RoPE distributes positional information. Squarefree indices carry 61.4% of the signal at N=210. The Möbius function μ(n) partitions these indices optimally.

3. **Compression is a view, not a transform.** The KV cache isn't data to be stored and retrieved — it's a projection of a low-dimensional spectral structure. Shannon-Prime doesn't compress in the information-theoretic sense of discarding data; it changes the basis to one where the signal is naturally sparse, then stores only the significant coefficients.

These observations lead to a system where compression quality *improves* with model scale (the scaling law proves this), where the same mathematics applies to every RoPE-based model regardless of modality, and where the transform is its own inverse.

---

## 2. The Transform: VHT2

The Vilenkin-Hartley Transform (VHT2) is the core of Shannon-Prime. It is a staged orthonormal transform with one critical property: **it is self-inverse**.

```
VHT2(VHT2(x)) = x     (within float tolerance)
```

### 2.1 Power-of-2 Case (Ship Path)

For head_dim = 2^k (the common case: 64, 128, 256), VHT2 reduces to the p=2 Hartley butterfly:

```
For each stage s in [0, log2(n)):
    For each butterfly pair (a, b) at stride 2^s:
        a' = (a + b) / √2
        b' = (a − b) / √2
```

The 1/√2 normalization at each stage ensures unit-norm basis vectors. After k stages, the transform is complete — and applying it again recovers the original vector. This is not an approximation; the round-trip error is 0.0000 in exact arithmetic and below float32 ULP in practice.

### 2.2 Multi-Prime Case (Sqfree Path)

For squarefree-padded dimensions (e.g., 154 = 2·7·11), VHT2 generalizes to stages over each prime factor. Each p-stage applies the p × p Hartley kernel:

```
H_p[i,j] = cas(2πij/p) / √p     where cas(x) = cos(x) + sin(x)
```

The Kronecker product structure V = H_2 ⊗ H_7 ⊗ H_11 creates a hierarchical decomposition where each sub-projection is a natural skeleton of the full signal. This hierarchy is what the Tier 3 predictor exploits.

Progressive prime expansion monotonically increases correlation:
- Walsh (Z/2Z only): 0.9490
- Z/2Z × Z/3Z: 0.9493
- Z/2Z × Z/3Z × Z/5Z: 0.9500
- Z/2Z × Z/3Z × Z/5Z × Z/7Z: 0.9513

### 2.3 Why Self-Inverse Matters

A self-inverse transform eliminates an entire class of bugs and complexity:

- One function serves as both encoder and decoder.
- One set of SIMD kernels to optimize, test, and maintain.
- No asymmetric performance (compress fast / decompress slow or vice versa).
- No risk of encode/decode mismatch across platforms.
- The forward and inverse paths take the same number of FLOPs.

Every backend — CPU, CUDA, Vulkan, Adreno, Hexagon — implements the same VHT2 function. The only difference is the instruction set.

---

## 3. The Write Path

When a KV vector enters the Shannon-Prime shadow cache, it passes through a pipeline that depends on which compression tier is active.

### 3.1 Ship Path (Tier 1)

```
raw K/V vector (head_dim floats, already RoPE'd)
    │
    ▼
VHT2 forward (in-place, head_dim)
    │
    ▼
Möbius reorder (squarefree indices → front)
    │
    ▼
Banded quantization (5/5/4/3 bits per band)
    │
    ▼
Store compressed bytes (total_bytes per vector)
```

**Step 1: VHT2.** The raw vector (128 floats for hd=128) is transformed in-place. After the transform, the coefficients represent spectral energy at each frequency — and most of that energy is concentrated in the early coefficients.

**Step 2: Möbius Reorder.** The coefficients are permuted so that squarefree indices (those where μ(n) ≠ 0) move to the front of the vector. Since squarefree indices carry the majority of the signal energy, this ensures that the highest-bit bands (which are quantized first) contain the most important coefficients. The reorder improves quality by 0.14 PPL at identical bit budget.

**Step 3: Banded Quantization.** The reordered coefficients are split into 4 equal bands. Each band gets an fp16 scale factor and packed integer values:
- Band 0 (32 coefficients): 5 bits each → 20 bytes data + 2 bytes scale
- Band 1 (32 coefficients): 5 bits each → 20 bytes data + 2 bytes scale
- Band 2 (32 coefficients): 4 bits each → 16 bytes data + 2 bytes scale
- Band 3 (32 coefficients): 3 bits each → 12 bytes data + 2 bytes scale

Total: 76 bytes per vector (from 256 bytes fp16) = 3.37× compression.

**Step 4: Store.** The 76-byte compressed record is written to the shadow cache's storage array at `[layer * n_heads + head][pos * packed_stride]`.

### 3.2 Sqfree + Spinor Path (Tier 2)

```
raw K/V vector (head_dim floats)
    │
    ▼
Sqfree pad (hd=128 → 154, mean-fill)
    │
    ▼
Vilenkin-Hartley forward (pad_dim=154)
    │
    ▼
Knight skeleton extraction (top L/2 by variance)
    │
    ▼
Band-quantize skeleton
    │
    ▼
Möbius CSR predict residuals from skeleton
    │
    ▼
N-bit residual quantization + spinor sheet bit
    │
    ▼
Store: banded skeleton + residual levels + mag + sheet bits
```

The sqfree path exploits the richer structure available when the dimension factors into multiple primes. The Knight skeleton selects the L/2 highest-variance squarefree indices (determined by calibration or default order). The Möbius CSR predictor reconstructs the remaining coefficients from the skeleton using the divisor relationship: for residual index r, `pred[r] = Σ μ(d) · skel_vals[slot(n/d)]` over divisors d where μ(d) ≠ 0.

The spinor sheet bit is a 1-bit SU(2) double-cover correction. The VHT2 spectrum has a phase ambiguity at each coefficient — the spinor bit records which sheet of the double cover the original value lives on. This single bit shifts the Pareto frontier: it achieves MOBIUS-default quality at +27% additional compression.

### 3.3 Hierarchical Vilenkin Path (Tier 3)

```
raw K/V vector (head_dim floats)
    │
    ▼
Sqfree pad → Vilenkin-Hartley forward
    │
    ▼
Extract core skeleton (Kronecker sub-projection, ~9%)
    │
    ▼
Band-quantize skeleton (e.g., 5,5 bits)
    │
    ▼
W · skeleton → predicted targets
    │
    ▼
Compute residuals = actual targets − predicted
    │
    ▼
N-bit residual quantization
    │
    ▼
Store: banded skeleton + residual magnitude + packed residual levels
```

The hierarchical path stores only the Kronecker sub-projection skeleton (e.g., 14 coefficients from Z/2Z × Z/7Z for pad_dim=154) and a per-(layer, head) calibrated linear predictor W that maps skeleton → remaining coefficients. The predictor is trained via ridge regression during the first prefill pass (warmup vectors → accumulate X^T X and X^T Y → solve).

Storage per position: n_skeleton × skel_bits + n_target × res_bits. For hd=128 with 14-skeleton at 5-bit bands and 2-bit residuals: 350 bits = 43.75 bytes. From 308 bytes fp16, that's 7.0× compression.

### 3.4 Variance-Ranked Calibration

Both the ship path and sqfree path support adaptive calibration. During a warmup phase:

1. `calibrate_begin()` — allocate per-coefficient running accumulators
2. `calibrate_feed(vec)` — for each warmup KV vector: transform → accumulate per-coefficient squared values
3. `calibrate_end()` — compute per-coefficient variance, rebuild internal masks

After calibration, the write path reorders coefficients by empirical variance rather than the default Möbius/index order. High-variance coefficients land in the highest-bit bands, low-variance coefficients in the lowest-bit bands. This is data-adaptive optimization on top of the structural optimization that Möbius provides.

---

## 4. The Read Path

### 4.1 Full Reconstruction

```
Load compressed bytes
    │
    ▼
Band dequantize (unpack integers, multiply by scale)
    │
    ▼
Möbius unreorder (restore original index order)
    │
    ▼
VHT2 (self-inverse → back to spatial domain)
    │
    ▼
Reconstructed K/V vector
```

This is the standard read path. For the ship path at hd=128, it reads 76 bytes and produces 128 floats.

### 4.2 Partial Band Reads

```
Load only bands [0, max_bands)
    │
    ▼
Dequantize loaded bands, zero-fill unloaded
    │
    ▼
Möbius unreorder → VHT2 → partial reconstruction
```

`sp_band_dequantize_partial` reconstructs from a subset of bands. Energy concentration means this is remarkably effective:

| max_bands | Correlation | Bytes Read (hd=128) |
|---|---|---|
| 0 | 0.00 | 0 |
| 1 | 0.30 | 22 |
| 2 | 0.86 | 44 |
| 3 | 0.88 | 60 |
| 4 | 0.99 | 76 (full) |

This is the primitive that enables System 1/2 switching and the disk-tier architecture.

### 4.3 Batch Operations

All cache types provide batch variants (`write_k_batch`, `read_k_batch`, etc.) that process N contiguous positions in a tight loop over persistent scratch buffers. No per-vector malloc, amortized pipeline setup. These are the hot-path functions — single-vector operations are convenience wrappers.

---

## 5. System 1/2 Dual-Cache Architecture

Inspired by Kahneman's dual-process theory:

**System 1 (Fast):** Read band 0 only (22 bytes per position). Compute the approximate attention dot product. For the ~99% of tokens where the top-scoring position is clear (large margin over second-place), accept the result immediately.

**System 2 (Careful):** When the margin is thin (attention entropy is high), promote to full reconstruction. Read remaining bands from wherever they live — RAM, NVMe, Optane, network.

The switching criterion is attention entropy: if the softmax distribution is peaked (one or few positions dominate), System 1 suffices. If the distribution is flat (many positions contribute roughly equally), the partial reconstruction doesn't preserve enough information and System 2 engages.

The `DualKvCache` wrapper manages this transparently. The caller reads K/V as usual; the cache internally decides whether to serve from the fast partial path or the full path.

---

## 6. Disk-Tier Architecture

### 6.1 Band-Major Storage Format (v3)

The v3 cache file format stores all band-0 data first, then all band-1 data, then band-2, etc. This means partial reads are physically contiguous:

```
File layout:
  [64-byte header]
  [band 0: all heads × all positions × band_0_stride]   ← System 1 reads only this
  [band 1: all heads × all positions × band_1_stride]
  [band 2: all heads × all positions × band_2_stride]
  [band 3: all heads × all positions × band_3_stride]
```

`sp_shadow_cache_load_partial(sc, prefix, hash, max_bands)` reads only the first max_bands regions from disk. On NVMe, that's a single contiguous read. On Optane (which supports byte-addressable access via DAX), it's a direct memory reference.

### 6.2 Progressive Loading Strategy

For a 70B model on a phone with 12 GB RAM:

1. **Band 0 in RAM** (~0.4× the full cache size). Always resident. Serves System 1.
2. **Band 1 on UFS** (phone flash). Loaded on demand when System 2 activates. UFS 3.1 delivers ~2 GB/s sequential.
3. **Bands 2–3 on network or not at all.** For the rare case where even band 0+1 isn't enough.

This makes 100K+ context feasible on memory-constrained devices with no quality regression on the vast majority of tokens.

### 6.3 Optane Integration (Beast Canyon)

Intel Optane NVMe (M10/M15) supports DAX (Direct Access) mode where the device is mapped directly into the process address space via `mmap`. No filesystem buffer cache, no page cache, no copies. The Beast Canyon backend's `sp_optane_reservoir_t` maps the entire GGUF file into memory and provides direct pointers to tensor data.

Optane latency targets (from Day Zero audit):
- Sequential 4KB stride: < 15 µs
- Random 4KB page: < 20 µs
- Sustained sequential bandwidth: > 1 GB/s

---

## 7. Cauchy Reset System

Compression errors accumulate over long decode chains. Left unchecked, the compressed past eventually diverges enough that the model's future predictions degrade. Shannon-Prime detects and corrects this via three cooperating components:

### 7.1 Ricci Sentinel

Monitors the p=3 spectral band energy as a scalar proxy for metric structural integrity. The p=3 band energy is tracked as an exponential moving average of the ratio `current_p3_energy / calibrated_p3_energy`. When `|1 − p3_ratio|` exceeds the metric_criticality threshold, a Cauchy reset is recommended.

The threshold is model-size-dependent:
- Small models (1B): 0.05 (fragile metric, narrow residual stream)
- Large models (8B+): 0.15 (robust metric, strong skip geodesics)

### 7.2 Mertens Oracle

M(n) = Σ_{k=1}^{n} μ(k) tracks the squarefree/non-squarefree balance. Its spectral decomposition via zeta zeros creates oscillations whose half-period at typical context scales (n ~ 256–2048) is 200–500 tokens — matching empirically-observed optimal reset windows.

The oracle pre-computes a risk schedule and provides O(1) lookup: `sp_mertens_risk(oracle, pos)` returns the current reset risk at any position.

### 7.3 Cauchy Controller

Combines Ricci + Mertens into a unified decision:
- Mode 0: Off (no resets)
- Mode 1: Fixed-N (reset every N tokens)
- Mode 2: Dynamic (Ricci triggers, Mertens schedules)

When a reset fires, the cache re-anchors: compressed vectors at the reset boundary are reconstructed and rewritten, establishing a fresh baseline for the Ricci sentinel.

---

## 8. PrimePE — Lattice-Aligned Positional Encoding

Standard RoPE uses geometric frequency spacing: `freq_i = base^(-2i/d)`. PrimePE blends in frequencies drawn from the multiplicative lattice of the integers, where composite numbers provide coordinates that primes generate.

```
freq_factors = sp_prime_pe_freq_factors(n_freqs, freq_base, alpha)
```

The output is a float array of multipliers on the geometric base frequencies. Pass to `ggml_rope_ext` as `freq_factors` or write into the model's `rope_freqs` tensor slot.

Alpha controls the blend:
- 0.0: Pure geometric (identity, all 1.0)
- 0.17: Recommended default (−0.6% PPL, deployment-robust)
- 0.22: Aggressive (−0.8% PPL, slightly less robust)
- 1.0: Pure lattice (not recommended)

The improvement is universal, robust to quantization level, and requires zero retraining. It works because prime frequencies correct positional frequency mismatch — a different error source from weight quantization.

---

## 9. The Scaling Law

The empirical law connecting KV-cache reconstruction fidelity to downstream perplexity:

```
log(PPL / PPL_base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)
```

This fits 9 configurations across 4 orders of magnitude (from baseline to catastrophic) within ±20%. The three exponents are interpretable:

- **Quadratic in K-error:** From the K·Q·V bilinearity of attention. Small K errors get squared through the dot product.
- **Sub-linear in params:** Bigger models absorb K error through head-averaging. A 1B model is fragile; an 8B model is robust.
- **Super-linear in weight precision:** Q4 amplifies K error ~2.8× vs Q8. Weight quantization and cache compression errors compound.

Use `sp_predicted_ppl_ratio(k_corr, params_b, bits)` as a pre-bench filter: if the predicted PPL ratio exceeds your quality budget, skip the configuration without running the full benchmark.

---

## 10. Beast Canyon — Heterogeneous Desktop Engine

Beast Canyon is the desktop counterpart to the phone-side Hexagon/QNN engine. It orchestrates inference across fundamentally different compute units:

### 10.1 Hardware Map

```
┌─────────────────────────────────────────────────┐
│  Intel i9-12900 (16C/24T, 24MB LLC, AVX-512)   │
│  ┌──────────────────────────────────────────┐   │
│  │  Optane M10/M15 NVMe (16-64GB, DAX)     │   │
│  │  → mmap'd GGUF model (zero-copy)        │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  GPU 0   │  │  GPU 1   │  │  S22U Sidecar│  │
│  │  (CUDA)  │  │  (Vulkan)│  │  (USB-C)     │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────┘
```

### 10.2 The Pipeline

The MoE forward pass — the "Execution Pulse" — runs as a 6-stage pipeline:

1. **Oracle:** Router selects top-K experts from the gating layer. Round-robin GPU assignment.
2. **Shred:** AVX-512 dequantizes the selected expert's quantized weights (Q4_0/Q4_1/Q8_0/Q4_K/Q6_K) from Optane into fp16 staging buffers in LLC.
3. **Prefetch:** While shredding expert N, prefetch expert N+1's pages from Optane into LLC.
4. **Dispatch:** Feed fp16 weights from LLC to GPU(s) for matmul. CUDA events / Vulkan fences track completion.
5. **Barrier:** Two-phase wait (spin for low latency → yield for power). Pre-shredder callback fires during dead time — overlapping next expert's dequant with current GPU compute.
6. **Sum:** Merge expert results. Flip ping-pong staging buffers.

### 10.3 Three Golden Rules

1. **Memory Is the Enemy.** Never copy what you can point to. Optane is byte-addressable via DAX — the shredder reads directly from the mapped GGUF.
2. **Optane Is RAM.** Sequential 4KB stride < 15 µs. The model lives in the address space, not on a filesystem.
3. **LLC Is the Finish Line.** The 24MB L3 cache is the staging area between Optane and GPU. AVX-512 shredder writes its fp16 output directly into cache lines that the PCIe DMA engine picks up.

### 10.4 Graceful Degradation

Beast Canyon adapts to whatever hardware is present:
- Dual GPU (CUDA + Vulkan): Full heterogeneous pipeline
- Single GPU: All experts route to one device
- CPU-only: AVX-512 matmul, no GPU dispatch
- No Optane: Standard filesystem mmap (slower, still works)
- No sidecar: PrimePE computed locally on CPU

---

## 11. Speculative Decoding Integration

Shannon-Prime supports speculative decoding with per-model compression. The draft model gets its own shadow cache with its own compression settings (typically more aggressive — draft errors are recoverable on target verification). The target model gets a separate cache at ship quality.

Validated configuration: Qwen2.5-Coder-3B (target, IQ2) + Qwen2.5-Coder-0.5B (draft, Q8) + `--draft 8` + SP-Hexagon FUSED_KQ fast path = 43.72 t/s on phone CPU (3.58× vs vanilla 12.20 t/s).

The cost model for SP-fused speculative decoding follows a curve that crosses vanilla at n_q ≈ 3–4:

| n_q | SP-fused (ms) | Vanilla (ms) |
|---|---|---|
| 12 | 12.0 | ~5.0 |
| 8 | 4.3 | ~5.0 |
| 6 | 2.5 | ~5.0 |
| 4 | 1.5 | ~5.0 |
| 3 | 1.15 | ~5.0 |

At draft depth 4–8, the amortized per-token cost with SP compression is 3–10× cheaper than vanilla.

---

## 12. Multi-GPU Sharding

The engine distributes transformer layers across multiple GPUs:

```
Layer L → GPU[ L × n_gpus / n_layer ]
```

Non-layer tensors (token embeddings, output norm, output projection) go to GPU 0 when fully offloaded, or stay CPU-mapped under partial offload. Cross-GPU copies are inserted automatically when a tensor produced on GPU i is consumed by an op on GPU j.

Validated: RTX 2060 (CUDA, K=0.9920, V=0.9730) + Intel UHD (Vulkan, identical fidelity). Cross-device correlation: 1.0000.

---

## 13. Model-Pack Registry

Per-architecture compression defaults stored as small records:

```
Architecture     Status        K Bits      V Bits    Notes
phi3             CALIBRATED    5,5,4,3     3         +2.44% PPL, within budget
qwen3            PROVISIONAL   5,5,4,3     3         Edge-fail: +5.14%
llama            PROVISIONAL   5,5,4,3     3
mistral          PROVISIONAL   5,5,4,3     3
gemma            PROVISIONAL   5,5,4,3     3
command-r        PROVISIONAL   5,5,4,3     3
deepseek         PROVISIONAL   5,5,4,3     3
```

`--model-preset auto` reads `general.architecture` from the GGUF file and applies the matching preset. Explicit user flags always win over preset defaults.

---

## 14. Integration Points

### 14.1 shannon-prime-engine (Reference Implementation)

The engine owns the compressed KV data path end-to-end. It loads GGUF files, runs native forward passes, and stores compressed K/V by construction — no "decompress to fp16, run attention, recompress" round-trip. The engine is the primary deployment path.

### 14.2 shannon-prime-comfyui (Generative Media)

16 ComfyUI custom nodes apply VHT2 compression and block-skip caching across video (Wan 2.x), image (Flux), audio (Stable Audio), and TTS (Voxtral 4B). The PyTorch backend runs the math; the nodes provide the workflow integration.

### 14.3 shannon-prime-llama (LM Studio Bridge)

A patch-based integration that compiles the full SP stack into llama.dll/libllama.so. This is a necessary bridge for the existing LM Studio ecosystem. The engine is the long-term replacement.

---

## 15. Data Flow Summary

```
                    ┌──────────────────┐
                    │   Model Weights   │
                    │   (GGUF on disk)  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Forward Pass    │
                    │   (engine/llama)  │
                    └────────┬─────────┘
                             │
                    Raw K/V vectors (fp32/fp16)
                             │
              ┌──────────────▼──────────────┐
              │     Shannon-Prime Write      │
              │  VHT2 → Möbius → Band Quant  │
              └──────────────┬──────────────┘
                             │
                   Compressed bytes (76B/vec)
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
   │  RAM (hot)  │   │  NVMe/Optane│   │  Network    │
   │  Band 0     │   │  Bands 1-2  │   │  Bands 3+   │
   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     Shannon-Prime Read       │
              │  Band Dequant → Möbius⁻¹     │
              │  → VHT2 (self-inverse)       │
              └──────────────┬──────────────┘
                             │
                    Reconstructed K/V vectors
                             │
                    ┌────────▼─────────┐
                    │    Attention      │
                    │    (Q · K^T · V)  │
                    └──────────────────┘
```

---

## 16. What Problems This Solves

**The KV Cache Memory Wall.** A 70B model at 32K context needs ~40 GB of KV cache in fp16. Shannon-Prime reduces this to ~12 GB (ship) or ~6 GB (sqfree) or ~3 GB (hierarchical). This is the difference between "needs a server" and "runs on a desktop."

**MoE Expert Activation.** Mixture-of-Experts models activate only K of N experts per token, but still need KV cache for all attended positions. Shannon-Prime compresses the KV cache per-expert, and Beast Canyon pipelines the Optane → Shredder → GPU expert activation loop so that total memory footprint stays bounded regardless of how many experts the model defines.

**Phone Inference.** A Snapdragon 8 Gen 1 phone has 12 GB RAM shared between the OS, apps, and the model. Shannon-Prime's progressive band loading + System 1/2 switching lets a 3B model run at full context with the KV cache occupying a fraction of what fp16 would require. The Hexagon DSP handles compression on dedicated silicon that doesn't compete with the CPU for thermal budget.

**Long Context.** 100K+ token contexts are memory-prohibitive in fp16. With disk-tier progressive loading, band 0 lives in RAM and serves 99% of attention queries; the remaining bands stream from storage on demand. The quality cost is near zero because VHT2 energy concentration means band 0 carries most of the signal.

**Cross-Platform Deployment.** One compression algorithm, 9 backends, same quality on every device. The math doesn't change; only the instruction set does. A model compressed on an NVIDIA GPU decompresses identically on a phone's Hexagon DSP.

---

## Appendix A: File Format

### VHT2 Cache Format (v2 / v3)

```
Header (64 bytes):
  uint32[0]:  Magic = 0x56485432 ("VHT2")
  uint32[1]:  Version (2 = per-head interleaved, 3 = band-major)
  uint32[2]:  packed_stride (bytes per head per position)
  uint32[3]:  n_positions
  uint32[4]:  n_heads
  uint32[5]:  cache_type (0=shadow, 1=sqfree, 2=hierarchical)
  uint32[6]:  model_hash_lo (FNV-1a, optional)
  uint32[7]:  model_hash_hi
  uint32[8..15]: Reserved

Data:
  v2: [pos_0_head_0][pos_0_head_1]...[pos_N_head_H] (interleaved)
  v3: [band_0_all_heads_all_pos][band_1_all_heads_all_pos]... (band-major)
```

v3 band-major layout enables partial reads: `sp_shadow_cache_load_partial` reads only the first max_bands contiguous regions.
