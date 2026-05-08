# Shannon-Prime: Video Generation Modality

**Version 1.0 -- 2026**
**Author: Ray Daniels**
**License: AGPLv3 / Commercial Dual License**

---

## Overview

Shannon-Prime's video generation integration targets Wan 2.x Diffusion Transformer
(DiT) models through the `shannon-prime-comfyui` repository. Unlike the LLM path,
which compresses KV cache entries via VHT2 spectral banding, the video path uses
**block-skip caching**: caching entire DiT block outputs and skipping recomputation
on subsequent denoising steps when drift is below threshold.

The two mechanisms share mathematical foundations (VHT2, Fisher-weighted spectral
similarity, Mertens scheduling) but apply them at different granularity:

| | LLM Path | Video Path |
|---|---|---|
| **Unit of caching** | Individual K/V vectors per head | Entire block output tensors |
| **Skip target** | Memory (KV cache size) | Compute (attention FLOPs) |
| **VHT2 role** | Compress/decompress cached K/V | Drift detection; optional block compression |
| **Drift metric** | Perplexity accumulation | Fisher-weighted spectral cosine similarity |

**Validated configuration:** 1280x720, 9 frames, Wan 2.2 TI2V-5B Q8, RTX 2060 12GB.
Result: 4.6x step speedup (32 s/step to 7 s/step) with TURBO preset.

---

## Wan 2.x Architecture Summary

Wan 2.x models are video Diffusion Transformers with 3D RoPE (temporal + 2 spatial
axes), T5 cross-attention for text conditioning, and adaLN (adaptive Layer Norm)
modulation driven by the diffusion timestep embedding.

```
Wan DiT Block (repeated N times)
========================================================

  Video Latents (x)   Timestep Embedding (e)
       |                       |
       |              adaLN Modulation
       |              e -> [shift, scale, gate] x 3
       |                       |
       +----> norm1 + shift/scale -----+
       |                               |
       |       Self-Attention          |
       |       Q,K: 3D RoPE           |
       |       (temporal + spatial)    |
       |               |               |
       |          y = attn(Q,K,V)      |
       |               |               |
       |          y = y * gate[0]   <--+  adaLN gate
       |               |
       +----> x = x + y
       |
       +----> Cross-Attention <--- T5 text embeddings
       |       K,V: no RoPE (vanilla linear projections)
       |
       +----> FFN (norm2 + shift/scale + MLP + gate)
       |
       v
  Output (to next block)
```

### Model Variants

| Model | Blocks | Architecture | Parameters | Expert Handling |
|---|---|---|---|---|
| Wan 2.1 14B | 40 | Dense DiT | 14B | None |
| Wan 2.1 1.3B | 30 | Dense DiT | 1.3B | None |
| Wan 2.2 A14B T2V | 40 | MoE DiT (2 experts) | 14B active | Expert switch at sigma=0.875 |
| Wan 2.2 A14B I2V | 40 | MoE DiT (2 experts) | 14B active | Expert switch at sigma=0.900 |
| Wan 2.2 TI2V-5B | 30 | Dense DiT | 5B | None |

---

## Block-Skip Caching: Core Mechanism

### The Observation

During denoising, not all DiT blocks contribute equally at every timestep. Early
blocks (L00-L03, "Permanent Granite") establish global composition and are highly
stable across consecutive steps -- their outputs change minimally (cosine similarity
> 0.95). Late blocks ("Jazz") handle texture and detail refinement, changing
significantly each step.

Block-skip caching exploits this by caching stable block outputs and replaying
them on subsequent steps, skipping the expensive Q/K/V projection and attention
computation entirely.

### adaLN Gate Reapplication

The critical correctness detail: cached outputs cannot be replayed verbatim because
the adaLN gate depends on the current timestep's sigma. On a cache hit, the system:

1. Recomputes the adaLN modulation from the current timestep embedding (trivial cost)
2. Retrieves the cached pre-gate attention output `y`
3. Applies the **current** step's gate value: `output = y * gate_current`

This ensures sigma-dependent brightness and contrast tracking remains correct even
on cached steps. Cross-attention and FFN run fresh every step (they are sigma-accurate
and the cross-attention K/V caching from the LLM-style path handles that separately).

```
Cache HIT path (skips ~50% of block compute):
==============================================

  Timestep embedding (e) --> adaLN --> gate[0]  (recomputed, trivial)
                                          |
  Cached y (pre-gate) -----> y * gate[0] ---> x = x + gated_y
                                          |
  Cross-attention (fresh) ------>         |
  FFN (fresh) ------>                     |
                                          v
                                      block output

Cache MISS path (full computation):
====================================

  Full self-attention: norm1 -> Q,K,V projection -> 3D RoPE
  -> attention scores -> output projection -> y
  -> store y in cache
  -> y * gate[0] -> x = x + gated_y
  -> cross-attention -> FFN -> block output
```

### Cross-Attention K/V Caching (Complementary)

Separate from block-skip, Shannon-Prime also caches the cross-attention K/V
projections. T5 text embeddings are identical across all denoising timesteps, so
the linear projections `cross_attn_k(context)` and `cross_attn_v(context)` produce
identical results every step across every block. These are computed once, cached
(optionally VHT2-compressed), and reconstructed on subsequent calls.

For dense models this yields 30-40 cache misses (one per block, first step) followed
by hundreds of hits. For MoE models, each expert has different projection weights,
so the cache is keyed by `(expert_id, block_index)`, yielding 80 misses for 40-block
MoE models (40 blocks x 2 experts).

---

## Block Tier System

Blocks are classified into tiers based on empirical sigma-sweep data (Phase 12).
Each tier has a configurable cache window (number of steps a cached output is
considered valid before mandatory recomputation).

### 4-Tier Skip Presets

| Tier | Blocks | Name | Default Window | Character |
|---|---|---|---|---|
| Tier 0 | L00-L03 | Permanent Granite | 10 steps | Global composition, maximally stable (cos_sim > 0.95) |
| Tier 1 | L04-L08 | Stable Sand | 3 steps | Mid-level structure, moderate stability |
| Tier 2 | L09-L15 | Volatile | 0 steps (disabled) | Texture/detail blocks, high drift |
| Tier 3 | L16-L39 | Deep/Late | 0 steps (disabled) | Fine detail, highest drift rate |

### Skip Profiles

| Profile | Tier 0 | Tier 1 | Tier 2 | Tier 3 | FFN Cache | Use Case |
|---|---|---|---|---|---|---|
| **CONSERVATIVE** | 5 | 2 | 0 | 0 | No | Quality-first, minimal risk |
| **BALANCED** | 10 | 3 | 0 | 0 | No | Default. Good quality/speed tradeoff |
| **AGGRESSIVE** | 15 | 5 | 2 | 0 | No | Speed-focused, small quality trade |
| **TURBO** | 20 | 8 | 3 | 2 | Yes | Maximum speed. Caches FFN too. ~4.6x on 5B |

### Memory Tiering (Hierarchical Cache)

Cache placement follows the tier hierarchy to balance VRAM pressure and access latency:

```
Tier 0 (L00-L03): GPU-resident
  - Zero overhead on cache hit: direct tensor reuse, no transfer
  - ~160 MB GPU for 4 blocks at 720p
  - Justified: these blocks hit 10+ consecutive steps

Tier 1 (L04-L08): CPU-resident
  - Fast .to(device) on hit (~5 ms per block transfer)
  - ~200 MB system RAM
  - Freed during VAE decode via CacheFlush node

Tier 2+ (L09+): No cache (recompute every step)
  - Unless explicitly enabled via tier_2_window/tier_3_window > 0
```

---

## VHT2 in the Video Path

VHT2 (Vilenkin-Hartley Transform) serves two distinct roles in the video path,
neither of which is traditional KV compression:

### 1. Spectral Drift Detection (Primary Role)

The Mertens Oracle inside BlockSkip uses Fisher-weighted cosine similarity in the
VHT2 spectral domain to decide when a cached block output has drifted too far
from the current true output.

The spectral basis separates "signal drift" from "noise drift":

- **Squarefree indices** (the prime harmonic basis: positions 1, 2, 3, 5, 6, 7, ...)
  carry structured information. Drift here means real content change.
- **Non-squarefree indices** (positions 4, 8, 9, 16, ...) are redundant overtones.
  Drift here is perceptually inert noise.

The Fisher diagonal weights squarefree positions at 1.0 and non-squarefree at 0.1,
then normalizes. This is a binary diagonal Fisher approximation: zero cost (single
elementwise multiply), but suppresses false cache invalidations from high-frequency
noise, yielding 10-20% longer cache windows at the same quality.

```
Fisher-weighted spectral cosine similarity:

  a_spectral = VHT2(cached_output) * fisher_weights
  b_spectral = VHT2(current_output) * fisher_weights

  similarity = dot(a_spectral, b_spectral) / (||a_spectral|| * ||b_spectral||)

  if similarity < drift_threshold:
      invalidate cache, recompute block
```

### 2. Block Output Compression (Optional, Default OFF)

For extreme VRAM pressure, cached block outputs can be VHT2-compressed using the
same skeleton extraction as the LLM path. A pre-allocated memory pool eliminates
the VRAM fragmentation that disabled VHT2 in Phase 15 LEAN:

```
VHT2 Memory Pool (pre-allocated at init):
  spectral_buf: [max_tokens, head_dim]   -- butterfly transform workspace
  skeleton_buf: [max_tokens, skel_size]  -- skeleton extraction target
  recon_buf:    [max_tokens, head_dim]   -- reconstruction workspace

Compress: block_output -> VHT2 forward -> extract skeleton -> CPU store
Decompress: CPU skeleton -> GPU scatter -> VHT2 inverse -> block_output
```

This achieves ~3.5x memory reduction but adds significant compute overhead at Wan
scale (165K vectors x 128D butterfly per block). Not recommended for normal use;
raw tensor caching is the default.

---

## MoE Expert Switching (Wan 2.2 A14B)

Wan 2.2's MoE architecture uses two full expert DiT models that activate based on
the noise level (sigma) during denoising:

```
Denoising Timeline (50 steps):
================================================================

  sigma=1.0              Boundary               sigma=0.0
   |                       |                       |
   |  HIGH-NOISE EXPERT    |   LOW-NOISE EXPERT    |
   |                       |                       |
   |  Global layout,       |   Detail refinement,  |
   |  composition          |   textures            |
   |                       |                       |
   v-----------------------v-----------------------v

  T2V boundary: sigma = 0.875
  I2V boundary: sigma = 0.900
```

Each expert has **different** weight matrices for cross-attention K/V projections,
self-attention projections, and FFN layers. Cached outputs from the high-noise expert
are **invalid** for the low-noise expert.

Shannon-Prime handles this by keying all caches by `(expert_id, block_index)`:

- On the first step of each expert regime, all blocks are cache misses (COMPUTE)
- Subsequent steps within the same regime are cache hits (RECONSTRUCT)
- At the expert switch boundary, the cache effectively flushes for the new expert

For the cross-attention K/V cache specifically, this means 80 total compulsory
misses for 40-block MoE models (40 blocks x 2 experts), versus 40 for dense models.

---

## 3D RoPE Characteristics

Wan uses factored 3D RoPE across three axes with anisotropic frequency allocation:

| Axis | Dimensions | Correlation (r) | Character |
|---|---|---|---|
| Temporal | 48 dims | 0.82 | High correlation -- causal anchor across frame window |
| Spatial-H | 40 dims | 0.73 | Lower correlation -- within-frame detail |
| Spatial-W | 40 dims | 0.73 | Lower correlation -- within-frame detail |

The higher temporal correlation (r=0.82 vs spatial r=0.73) confirms that inter-frame
coherence is stronger than intra-frame spatial coherence. Shannon-Prime's lattice
RoPE hook blends prime-indexed frequencies with anisotropic tier mapping:

- **Temporal axis:** Long-Tier primes (1009..8209) for low-frequency periodicity
- **Spatial axes:** Local-Tier primes (2..101) for high-frequency structure

The blend factor (alpha=0.17) is applied before cos/sin embedding computation.
Zero per-token cost: the frequency factors are constant per dimension.

---

## Mertens Oracle

The Mertens Oracle is the adaptive skip schedule built into BlockSkip. It derives
from the same Cauchy-reset mathematics used in the LLM path's decode-chain
supervisor, applied to diffusion denoising steps.

### Mechanism

Each block maintains a rolling cosine similarity between its cached output and
fresh computation. The oracle uses a zeta-derived schedule (50 Riemann zeta zeros)
to identify "high-risk" positions in the denoising timeline where drift is most
likely to accumulate.

At each step, for each block:

1. Compute Fisher-weighted cosine similarity between cached `y` and fresh `y`
2. Update the rolling similarity estimate
3. If `rolling_sim < drift_threshold`, halve the cache window (reactive tightening)
4. If `rolling_sim > restore_threshold`, restore the full window (adaptive relaxation)

This adaptive behavior means the effective cache window is not static: it
self-adjusts based on the actual stability of each block at each point in the
denoising process.

---

## Ricci Sentinel

The Ricci Sentinel is an optional diagnostic monitor (default: opt-in) that records
per-step sigma timelines, cache window decisions, and rolling similarity values
across the full denoising process.

**Measured benefit: 0 incremental** beyond what Mertens Oracle already provides.
The Sentinel is useful for debugging and tuning but does not improve output quality
or speed in production.

Output columns when enabled:

```
Step  sigma   regime  win[0]  win[4]  roll_sim[0]
  0   0.998   HIGH    15      5       ---
  1   0.982   HIGH    15      5       0.987
  2   0.961   HIGH    15      5       0.993
  ...
 25   0.501   LOW      7      2       0.942
 26   0.478   LOW      7      2       0.938
```

---

## Performance

### Step Timing (RTX 2060 12GB)

| Model | Profile | Step Time (Baseline) | Step Time (Cached) | Speedup |
|---|---|---|---|---|
| Wan 2.2 TI2V-5B Q8 | TURBO | 32 s/step | 7 s/step | **4.6x** |
| Wan 2.2 TI2V-5B Q8 | BALANCED | 32 s/step | 12 s/step | **2.7x** |
| Wan 2.2 A14B T2V | BALANCED | -- | -- | **3.5x** |
| Wan 2.2 A14B I2V | BALANCED | -- | -- | **3.5x** |

### Cache Statistics (Typical Run, 5B BALANCED, 50 Steps)

```
Block-skip cache:
  Tier 0 (L00-L03): 4 blocks x 47 hits = 188 skipped attention ops
  Tier 1 (L04-L08): 5 blocks x 15 hits =  75 skipped attention ops
  Total attention computations avoided: 263 / 1500 (17.5%)

Cross-attention K/V cache:
  30 blocks x 1 compute each = 30 misses
  30 blocks x 49 reconstruct each = 1470 hits
  Hit rate: 1470/1500 = 98.0%
```

### VRAM Budget (720p, 9 Frames, 5B)

| Component | VRAM |
|---|---|
| Model weights (Q8) | ~5.5 GB |
| Tier 0 GPU cache (4 blocks) | ~160 MB |
| VHT2 memory pool (if enabled) | ~50 MB |
| Working memory (latents, attention) | ~2-3 GB |
| **Total** | **~8-9 GB** |

---

## How It Differs From Other Systems

### vs TeaCache

TeaCache uses naive L2 distance or cosine similarity on raw block outputs to decide
skip eligibility. Shannon-Prime differs in three ways:

1. **Spectral drift detection.** Fisher-weighted cosine similarity in the VHT2
   spectral basis separates meaningful content drift from perceptually inert
   high-frequency noise. TeaCache's raw-domain similarity triggers false
   invalidations from noise drift, resulting in shorter effective cache windows.

2. **adaLN gate reapplication.** Shannon-Prime caches the pre-gate output and
   reapplies the current timestep's gate on cache hits. TeaCache caches the
   post-gate output, which bakes in the wrong sigma-dependent modulation on
   replayed steps, causing brightness/contrast drift over cached windows.

3. **Expert-aware MoE handling.** Shannon-Prime keys caches by `(expert_id,
   block_index)` for Wan 2.2 A14B, with explicit sigma boundary detection for
   expert switching. TeaCache has no MoE awareness and would silently serve
   stale expert-0 outputs during expert-1 steps.

### vs DeepCache

DeepCache caches intermediate features at fixed intervals (every N steps) without
content-adaptive decisions. Shannon-Prime's Mertens Oracle adjusts the effective
cache window per-block based on measured drift, automatically tightening during
high-drift phases and relaxing during stable phases. DeepCache also lacks spectral
drift detection and adaLN gate correction.

### vs PAB (Pyramid Attention Broadcast)

PAB broadcasts attention outputs from one frame to adjacent frames at different
pyramid levels. It operates on the temporal axis (inter-frame) while Shannon-Prime's
block-skip operates on the denoising axis (inter-step). The two are orthogonal and
could in principle be combined, though this is not currently implemented.

### Comparison Table

| Feature | Shannon-Prime | TeaCache | DeepCache | PAB |
|---|---|---|---|---|
| Drift detection | Fisher-weighted spectral (VHT2) | Raw cosine/L2 | None (fixed schedule) | None (fixed pyramid) |
| adaLN gate correction | Yes (pre-gate cache + reapply) | No (post-gate cache) | No | N/A |
| MoE expert awareness | Yes (keyed by expert_id) | No | No | No |
| Adaptive window | Yes (Mertens Oracle) | No | No | No |
| Skip axis | Denoising steps | Denoising steps | Denoising steps | Temporal (frames) |
| Memory tiering | GPU/CPU/None per tier | Flat GPU | Flat GPU | Flat GPU |

---

## ComfyUI Nodes Reference

The `shannon-prime-comfyui` repository provides 8 nodes for Wan video generation.

### ShannonPrimeWanCache

**Display name:** Shannon-Prime: Wan Cross-Attn Cache

Wraps Wan DiT cross-attention K/V linear layers with VHT2-compressed caching.
Intercepts `block.cross_attn.k` and `block.cross_attn.v` (and `.k_img`/`.v_img`
for I2V), caching the first computation and returning VHT2-reconstructed values
on subsequent calls. Input-change detection uses `tensor.data_ptr()`.

| Input | Type | Default | Description |
|---|---|---|---|
| model | MODEL | required | Model to patch |

### ShannonPrimeWanBlockSkip

**Display name:** Shannon-Prime: Wan Block-Level Self-Attn Skip (VHT2)

The primary acceleration node. Patches `WanAttentionBlock.forward()` to skip
Q/K/V projection and attention computation on cache-hit steps. Includes the
Mertens Oracle (adaptive drift detection) and hierarchical memory tiering.

| Input | Type | Default | Description |
|---|---|---|---|
| model | MODEL | required | Model to patch |
| tier_0_window | INT | 10 | Cache window for L00-L03 (Permanent Granite) |
| tier_1_window | INT | 3 | Cache window for L04-L08 (Stable Sand) |
| tier_2_window | INT | 0 | Cache window for L09-L15 (Volatile) |
| tier_3_window | INT | 0 | Cache window for L16-L39 (Deep/Late) |
| cache_ffn | BOOLEAN | False | Also cache FFN output (TURBO mode) |
| cache_dtype | ENUM | fp16 | fp16 / fp8 / mixed |
| cache_compress | ENUM | raw | raw / vht2 (spectral compression via memory pool) |
| verbose | BOOLEAN | False | Print per-block HIT/MISS logs |

### ShannonPrimeWanSigmaSwitch

**Display name:** Shannon-Prime: Wan Sigma Switch (Phase 13)

Sigma-aware cache window adaptation. Expands cache windows at high sigma (early
steps, Arithmetic Granite regime) and contracts them at low sigma (late steps,
Semantic Sand regime). Captures real sigma from the sampler via
`set_model_unet_function_wrapper()`.

| Input | Type | Default | Description |
|---|---|---|---|
| model | MODEL | required | Model to patch (after BlockSkip) |
| high_sigma_mult | FLOAT | 1.5 | Window multiplier at high sigma |
| low_sigma_mult | FLOAT | 0.5 | Window multiplier at low sigma |
| sigma_split_frac | FLOAT | 0.5 | Fraction of sigma range for HIGH/LOW boundary |
| verbose | BOOLEAN | False | Print regime transitions |

### ShannonPrimeWanCacheFlush

**Display name:** Shannon-Prime: Wan Block Cache Flush (before VAE)

Clears all cached block outputs and cross-attention K/V before VAE decode. Place
between KSampler and VAEDecode to free GPU/CPU memory before the VRAM-intensive
VAE pass.

| Input | Type | Default | Description |
|---|---|---|---|
| model | MODEL | required | Model to flush |

### ShannonPrimeWanRicciSentinel

**Display name:** Shannon-Prime: Wan Ricci Sentinel (Phase 13 diag)

Optional diagnostic node. Records per-step sigma, regime classification, effective
windows, and rolling cosine similarity. Prints a summary table at generation
boundaries. Measured 0 incremental benefit; useful for tuning and debugging only.

| Input | Type | Default | Description |
|---|---|---|---|
| model | MODEL | required | Model to monitor (after BlockSkip + SigmaSwitch) |
| sigma_split_frac | FLOAT | 0.5 | Match SigmaSwitch setting |
| verbose | BOOLEAN | True | Per-step printing |

### ShannonPrimeWanCacheStats

**Display name:** Shannon-Prime: Cache Stats

Reports cross-attention K/V cache hit/miss statistics after a generation completes.

### ShannonPrimeWanCacheSqfree

**Display name:** Shannon-Prime: Wan Cross-Attn Cache (Sqfree)

Variant of WanCache using squarefree skeleton extraction for cross-attention K/V
compression instead of full VHT2 banding.

### ShannonPrimeWanSelfExtract

**Display name:** Shannon-Prime: Wan Self-Attn Extract (Phase 12)

Phase 12 self-attention analysis node. Extracts and logs self-attention patterns
for sigma-sweep analysis and block stability characterization.

---

## Workflow Reference

### Recommended Node Chain

```
UnetLoader
    |
    v
ShannonPrimeWanCache          (cross-attention K/V caching)
    |
    v
ShannonPrimeWanBlockSkip      (block-skip with tier windows)
    |
    v
ShannonPrimeWanSigmaSwitch    (sigma-adaptive window scaling)
    |
    v
[ShannonPrimeWanRicciSentinel] (optional diagnostic)
    |
    v
KSampler
    |
    v
ShannonPrimeWanCacheFlush     (free cache before VAE)
    |
    v
VAEDecode
    |
    v
SaveImage / SaveAnimatedWEBP
```

### Required ComfyUI Flags

```
python main.py --normalvram --disable-async-offload
```

- `--normalvram`: Prevents ComfyUI's aggressive model offloading, which would
  evict cached tensors from GPU between steps.
- `--disable-async-offload`: Prevents async CPU offload from racing with cache
  reads. Without this flag, a cache hit can reference a tensor that is mid-transfer
  to CPU.

### TURBO Preset (Maximum Speed)

For maximum throughput with acceptable quality trade-off:

```
ShannonPrimeWanBlockSkip:
  tier_0_window: 20
  tier_1_window: 8
  tier_2_window: 3
  tier_3_window: 2
  cache_ffn: True
  cache_dtype: mixed
```

This caches self-attention AND FFN outputs for all tiers, making cached steps
near-zero compute (only adaLN modulation + cross-attention remain). Measured at
4.6x on Wan 2.2 TI2V-5B, RTX 2060.

### CONSERVATIVE Preset (Quality-First)

For minimal quality risk:

```
ShannonPrimeWanBlockSkip:
  tier_0_window: 5
  tier_1_window: 2
  tier_2_window: 0
  tier_3_window: 0
  cache_ffn: False
  cache_dtype: fp16
```

---

## Appendix: Architecture Diagrams

### Full System Flow

```
                    Shannon-Prime Video Generation Pipeline
================================================================

  Text Prompt                          Reference Image (I2V)
      |                                       |
      v                                       v
  T5 Encoder                            CLIP/VAE Encode
      |                                       |
      v                                       v
  text_embeddings ----+---- image_embeddings (I2V only)
                      |
                      v
        +--------------------------+
        |   Wan DiT (N blocks)     |
        |                          |
        |   For each denoising     |
        |   step (50 typical):     |
        |                          |
        |   +--------------------+ |
        |   | Block 0 (Tier 0)   | |     +------------------+
        |   | cos_sim > 0.95     |------>| GPU Cache        |
        |   | window = 10 steps  | |     | (Tier 0: ~160MB) |
        |   +--------------------+ |     +------------------+
        |                          |
        |   +--------------------+ |     +------------------+
        |   | Block 4 (Tier 1)   |------>| CPU Cache        |
        |   | window = 3 steps   | |     | (Tier 1: ~200MB) |
        |   +--------------------+ |     +------------------+
        |                          |
        |   +--------------------+ |
        |   | Block 12 (Tier 2)  | |     No cache (default)
        |   | Recompute every    | |     Full attention each step
        |   | step               | |
        |   +--------------------+ |
        |                          |
        |   Cross-Attention K/V    |     +------------------+
        |   (T5 embeddings)        |---->| VHT2 Compressed  |
        |   Compute once, cache    |     | K/V Cache        |
        |   for all steps          |     +------------------+
        |                          |
        +--------------------------+
                      |
                      v
                 CacheFlush
                      |
                      v
                  VAE Decode
                      |
                      v
                 Video Output
```

### MoE Expert Timeline

```
Wan 2.2 A14B Denoising (50 steps, T2V):
================================================================

  Step 0          Step ~6           Step 49
  sigma=1.0       sigma=0.875       sigma=0.0
   |                |                  |
   |  HIGH-NOISE    |  LOW-NOISE       |
   |  EXPERT        |  EXPERT          |
   |                |                  |
   |  MISS: all 40  |  MISS: all 40    |
   |  blocks        |  blocks          |
   |  (populate     |  (populate       |
   |   high_noise   |   low_noise      |
   |   cache)       |   cache)         |
   |                |                  |
   |  HIT: steps    |  HIT: steps      |
   |  1 through     |  ~7 through      |
   |  ~6            |  49              |
   |                |                  |
   v================v==================v

  Cache keys:  (high_noise, block_0)  (low_noise, block_0)
               (high_noise, block_1)  (low_noise, block_1)
               ...                    ...
               (high_noise, block_39) (low_noise, block_39)
```

### Sigma Regime Map

```
Sigma vs Block Stability (Phase 12 measured data):
================================================================

  cos_sim
  1.00 |  * * * * * * * * * * * *
       |  * * * * * * * * * .
  0.95 |--*-*-*-*-*-*-*------.-------- Tier 0 threshold
       |        * * * .   .
  0.90 |          * .   .
       |            .   .
  0.85 |          .   .
       |        .   .                   Tier 1 threshold
  0.80 |-------.---.----------------------
       |      .  .
  0.75 |    .  .
       |  .  .                          Jazz regime (no cache)
  0.70 | . .
       |..
  0.65 |.
       +--+--+--+--+--+--+--+--+--+--+--> step
        0  5  10 15 20 25 30 35 40 45 50

  Legend: * = L00 (Tier 0)    . = L12 (Tier 2)
```

---

## Related Documents

- [INTEGRATION-COMFYUI.md](INTEGRATION-COMFYUI.md) -- Cross-attention K/V cache detail
- [CAUCHY-RESET.md](CAUCHY-RESET.md) -- Mertens Oracle and Ricci Sentinel mathematics
- [Shannon-Prime.md](Shannon-Prime.md) -- Core VHT2 compression system overview
- [BACKEND-TORCH.md](BACKEND-TORCH.md) -- PyTorch backend implementation
- [PRIME-ENGINE.md](PRIME-ENGINE.md) -- Engine architecture
