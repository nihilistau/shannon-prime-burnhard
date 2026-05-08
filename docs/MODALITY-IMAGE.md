# Shannon-Prime: Image Generation Integration (Flux DiT)

## Overview

Shannon-Prime extends its KV cache compression system to image generation through
block-skip caching for Flux v1 and v2 Diffusion Transformer (DiT) models. Where
the video modality (Wan 2.1/2.2) caches cross-attention K/V projections, the image
modality caches entire block outputs during the denoising loop and replays them
with corrected modulation gates on cache hits.

The same "Granite/Jazz" principle applies: early blocks establish compositional
structure (layout, shapes, spatial relationships) while late blocks add texture
and fine detail. Block-skip exploits this by aggressively caching stable early
blocks and leaving volatile late blocks to run every step.

Three ComfyUI custom nodes provide the integration:
`ShannonPrimeFluxBlockSkip`, `ShannonPrimeFluxCacheFlush`, and
`ShannonPrimeFluxCacheFlushModel`.


## Flux Architecture

### Dual-Stream Design

Flux uses a dual-stream architecture with two block types:

```
Input: img latents + txt embeddings (T5/CLIP)
       │
       ▼
 ┌─────────────────────────────────────────────────────────┐
 │  DoubleStreamBlock ×19 (Flux v1) / ×N (Flux v2)        │
 │                                                         │
 │   img stream ──► img_norm1 ──► img_attn(QKV) ──┐       │
 │                                                 │       │
 │   txt stream ──► txt_norm1 ──► txt_attn(QKV) ──┤       │
 │                                                 │       │
 │                          Joint Attention ◄──────┘       │
 │                          (cat txt_Q+img_Q,              │
 │                           cat txt_K+img_K,              │
 │                           cat txt_V+img_V)              │
 │                                │                        │
 │                     ┌──────────┴──────────┐             │
 │                     ▼                     ▼             │
 │              img_attn_proj          txt_attn_proj       │
 │              + img_mod1.gate        + txt_mod1.gate     │
 │                     │                     │             │
 │              img_norm2 → img_mlp    txt_norm2 → txt_mlp │
 │              + img_mod2.gate        + txt_mod2.gate     │
 │                     │                     │             │
 │                     ▼                     ▼             │
 │                 img_out               txt_out           │
 └─────────────────────────────────────────────────────────┘
       │
       ▼
 ┌─────────────────────────────────────────────────────────┐
 │  SingleStreamBlock ×38 (Flux v1) / ×N (Flux v2)        │
 │                                                         │
 │   x = cat(txt, img) ──► pre_norm ──► linear1            │
 │                           │                             │
 │                    ┌──────┴──────┐                      │
 │                    ▼             ▼                       │
 │                 QKV split    MLP split                   │
 │                    │             │                       │
 │                 attention    mlp_act                     │
 │                    │             │                       │
 │                    └──────┬──────┘                       │
 │                           ▼                             │
 │                       linear2 → output                  │
 │                       + mod.gate                        │
 └─────────────────────────────────────────────────────────┘
       │
       ▼
   Denoised latents → VAE Decode → Image
```

**Key difference from Wan:** Flux has no separate cross-attention layer. Text
embeddings are folded into joint attention by concatenating txt and img Q/K/V
before computing attention scores. This means there is no "static text K/V" to
cache across timesteps (as in Wan). Instead, Shannon-Prime caches the block
output residuals and replays them with corrected modulation gates.

### Model Parameters

| Parameter | Flux v1 | Flux v2 |
|-----------|---------|---------|
| `head_dim` | 128 | 64 |
| `num_heads` | 24 | 48 |
| `hidden_size` | 3072 | 3072 |
| `axes_dim` (RoPE) | `[16, 56, 56]` | `[32, 32, 32, 32]` |
| Double blocks | 19 | varies |
| Single blocks | 38 | varies |

### adaLN Modulation

Both block types use Adaptive Layer Normalization (adaLN) for timestep
conditioning. The `Modulation` module produces `ModulationOut(shift, scale, gate)`
from the conditioning vector `vec`. This is load-bearing for cache correctness:
on a cache hit, we must recompute the modulation from the current timestep's `vec`
and apply the current `gate` to the cached residual.

```
vec (timestep conditioning)
  │
  ▼
Modulation → ModulationOut(shift, scale, gate)
  │
  ├── shift, scale: applied to pre-attention normalized input
  │                  (only on MISS — not needed on HIT)
  │
  └── gate: applied to post-attention output (ALWAYS recomputed)
            This is how cache hits stay sigma-accurate.
```


## Block-Skip Caching Mechanism

### Core Principle

During denoising, each DiT block produces an attention output residual that is
added to the hidden state. On subsequent timesteps, if the block's behaviour is
stable (determined by cache window and streak limits), the cached residual is
replayed with the current step's modulation gate instead of recomputing the full
joint attention.

```
Denoising Step Timeline
═══════════════════════════════════════════════════════════
  Step 1        Step 2        Step 3        Step 4
  ──────        ──────        ──────        ──────
  Block D00:    Block D00:    Block D00:    Block D00:
  MISS          HIT           HIT           HIT
  (compute,     (replay       (replay       (replay
   cache)        + gate)       + gate)       + gate)

  Block D12:    Block D12:    Block D12:    Block D12:
  MISS          MISS          MISS          MISS
  (always       (always       (always       (always
   compute)      compute)      compute)      compute)
```

### DoubleStreamBlock Cache Hit Path

On cache hit, the expensive joint attention (QKV projection, attention scores,
output projection) is skipped entirely:

1. **Recompute modulation** from current `vec` (trivial cost — one linear layer)
2. **Load cached `img_attn_proj`** and apply with current `img_mod1.gate`
3. **Load cached `txt_attn_proj`** and apply with current `txt_mod1.gate`
4. **Run MLP fresh** (sigma-accurate, uses current `img_mod2` / `txt_mod2`)
5. **TURBO mode** (optional): also cache and replay the MLP output, achieving
   near-zero compute on HIT at the cost of potential fine-detail loss

### SingleStreamBlock Cache Hit Path

1. **Recompute modulation** from current `vec`
2. **Load cached output** (fused attention+MLP result) and apply with `mod.gate`

SingleStreamBlocks fuse attention and MLP into a single output, so there is no
separate MLP path on cache hit.

### Cache Invalidation

Three mechanisms prevent stale cache from degrading quality:

| Mechanism | Purpose |
|-----------|---------|
| **Window** | Maximum age (in steps) before a cached entry expires |
| **Streak limit** | Maximum consecutive hits before forcing a fresh compute |
| **Shape validation** | Batch/sequence dimension mismatch triggers immediate invalidation |

### Block Tier Map

Blocks are organized into tiers based on their role in the generation process.
Each tier has a different cache window (how many steps a cached result stays
valid) and streak limit (max consecutive cache hits).

| Tier | Blocks | Window (default) | Streak Limit | Role |
|------|--------|-----------------|--------------|------|
| Double Tier-0 | D00-D03 | 8 | 8 | Compositional "Granite" — layout, composition |
| Double Tier-1 | D04-D11 | 3 | 4 | Stable mid-layers — structural detail |
| Double Tier-2 | D12+ | 0 (disabled) | 3 | Volatile detail — texture, fine features |
| Single Tier-0 | S00-S07 | 2 | 4 | Refinement — moderate caching |
| Single Tier-1 | S08+ | 0 (disabled) | 3 | Final detail — no cache by default |

**These are starting values.** Run a sigma-sweep (Ricci sentinel) to empirically
measure per-block stability and tune windows for your specific model and prompts.


## 2D Lattice RoPE

### Why Lattice Frequencies Matter for Images

Flux uses 2D Rotary Position Encoding to encode spatial position in the image.
The `axes_dim` parameter splits the head dimension into per-axis groups:

- **Flux v1**: `[16, 56, 56]` — 16 dims for temporal (degenerate for single-frame
  images), 56 for height, 56 for width
- **Flux v2**: `[32, 32, 32, 32]` — four equal groups

Shannon-Prime's `_tiered_lattice_factors` function computes per-axis frequency
multipliers that blend geometric base frequencies with lattice-aligned
(composite-number) frequencies. This exploits the same prime-harmonic structure
as VHT2: squarefree indices correspond to the independent harmonic basis
functions, and composite numbers fill the lattice gaps.

### Tier Assignment by Axis

```
Flux v1 axes_dim = [16, 56, 56]
                     │    │    │
                     │    │    └── Width  → Local-Tier (fine spatial detail)
                     │    └─────── Height → Local-Tier (fine spatial detail)
                     └──────────── Temporal → Long-Tier (causal anchors)

dim <= 32 → Long-Tier:  large composites (500-8209), low frequency
dim >  32 → Local-Tier: small composites (4-199), high frequency
```

- **Local-Tier** (spatial axes): Small composite numbers (4, 6, 8, 9, 10, ...)
  produce high-frequency lattice points suited for encoding fine spatial detail
  in height and width dimensions.

- **Long-Tier** (temporal axis): Large composite numbers (500+) produce
  low-frequency anchors suited for the temporal/causal dimension, which is
  degenerate (single frame) for static images.

### Implementation

The lattice RoPE is installed by monkey-patching `comfy.ldm.flux.math.rope()`.
This is idempotent (safe to call multiple times) and controlled by two
parameters:

- `lattice_rope` (bool, default `True`): Enable/disable the patch
- `lattice_alpha` (float, default `0.17`): Blend factor. 0.0 = pure geometric
  (standard RoPE), 0.17 = paper default blend

The blended frequency for each position is:

```
omega_blended = (1 - alpha) * omega_geometric + alpha * omega_lattice
```

At `alpha = 0.0`, the factors tensor is all ones (no modification). At
`alpha = 0.17`, low-frequency lattice structure is blended in at 17% weight.


## VHT2 Role in Image Caching

### Fisher-Weighted Spectral Similarity

Cache hit/miss decisions use Fisher-weighted cosine similarity rather than naive
L2 distance. The Fisher diagonal weights are derived from the VHT2 spectral
basis:

- **Squarefree indices** (positions 1, 2, 3, 5, 6, 7, 10, 11, ...): Weight 1.0.
  These correspond to the independent prime harmonic basis functions and carry
  structured information.
- **Non-squarefree indices** (positions 4, 8, 9, 16, 18, ...): Weight 0.1.
  These are redundant overtones — drift along these dimensions is perceptually
  inert.

This suppresses false cache invalidations from high-frequency noise, yielding
longer effective cache windows at equivalent quality.

```python
# For head_dim=128:
#   ~79 squarefree positions → weight 1.0 (signal)
#   ~49 non-squarefree       → weight 0.1 (noise)
#
# Fisher-weighted cosine similarity:
#   sim = sum(a_i * b_i * w_i^2) / (||a*w|| * ||b*w||)
```

### Block Output Caching

Cached block outputs are stored in the VHT2 spectral domain's natural structure.
The cache stores:

| Key | Stored Tensor | Used By |
|-----|--------------|---------|
| `attn_cache_img[block_idx]` | Post-projection img attention residual | DoubleStreamBlock hit |
| `attn_cache_txt[block_idx]` | Post-projection txt attention residual | DoubleStreamBlock hit |
| `mlp_cache_img[block_idx]` | MLP output for img stream | TURBO mode only |
| `mlp_cache_txt[block_idx]` | MLP output for txt stream | TURBO mode only |
| `single_cache[block_idx]` | Fused attention+MLP output | SingleStreamBlock hit |

Cache dtype is configurable: `fp16` (default), `fp8` (e4m3fn, memory savings),
or `mixed` (fp16 for early blocks, fp8 for later blocks).


## ComfyUI Nodes

### ShannonPrimeFluxBlockSkip

**Display name:** Shannon-Prime: Flux Block-Level Attention Skip (VHT2)

The main integration node. Patches both DoubleStreamBlock and SingleStreamBlock
forward methods on the input MODEL. Returns the patched MODEL.

**Inputs:**

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | MODEL | (required) | Flux model from UnetLoader |
| `double_tier0_window` | INT | 8 | Cache window for D00-D03 (Granite). 0 = disabled |
| `double_tier1_window` | INT | 3 | Cache window for D04-D11 (stable mid). 0 = disabled |
| `double_tier2_window` | INT | 0 | Cache window for D12+ (volatile). 0 = disabled |
| `single_tier0_window` | INT | 2 | Cache window for S00-S07 (refinement). 0 = disabled |
| `single_tier1_window` | INT | 0 | Cache window for S08+ (final detail). 0 = disabled |
| `cache_mlp` | BOOLEAN | False | TURBO mode: also cache MLP outputs |
| `cache_dtype` | ENUM | fp16 | fp16 / fp8 / mixed |
| `lattice_rope` | BOOLEAN | True | Install factored 2D lattice RoPE |
| `lattice_alpha` | FLOAT | 0.17 | Lattice blend alpha (0.0 = pure geometric) |
| `verbose` | BOOLEAN | False | Print per-block HIT/MISS + Fisher similarity |

**Workflow placement:**

```
UnetLoader → ShannonPrimeFluxBlockSkip → KSampler → ...
```

### ShannonPrimeFluxCacheFlush

**Display name:** Shannon-Prime: Flux Cache Flush (before VAE)

Latent-passthrough node that triggers garbage collection and
`torch.cuda.empty_cache()` to free memory from cached attention tensors before
VAE decode. Does not require model access.

**Inputs:** `samples` (LATENT)
**Outputs:** `samples` (LATENT, passthrough)

### ShannonPrimeFluxCacheFlushModel

**Display name:** Shannon-Prime: Flux Cache Flush (Model-Aware)

Model-aware flush that walks the actual Flux blocks and deterministically clears
all closure-captured cache state dicts. This is the recommended flush node when
the model reference is available.

**Inputs:** `model` (MODEL)
**Outputs:** `model` (MODEL, passthrough)

### Recommended Workflow

```
UnetLoader
    │
    ▼
ShannonPrimeFluxBlockSkip  (patches model with block-skip caching)
    │
    ▼
KSampler  (denoising loop — cache hits accelerate this)
    │
    ├──► ShannonPrimeFluxCacheFlush  (free memory before VAE)
    │        │
    │        ▼
    │    VAEDecode
    │        │
    │        ▼
    │    SaveImage
    │
    └──► ShannonPrimeFluxCacheFlushModel  (clear model caches for next gen)
```


## Configuration Guide

### Mode Presets

| Mode | Double T0 | Double T1 | Double T2 | Single T0 | Single T1 | MLP Cache | Notes |
|------|-----------|-----------|-----------|-----------|-----------|-----------|-------|
| **Conservative** | 4 | 2 | 0 | 1 | 0 | Off | Minimal quality impact. Safe starting point. |
| **Balanced** (default) | 8 | 3 | 0 | 2 | 0 | Off | Good speed/quality tradeoff. |
| **Aggressive** | 12 | 6 | 2 | 4 | 2 | Off | Noticeable speedup. May affect fine texture. |
| **Turbo** | 15 | 8 | 3 | 6 | 3 | On | Maximum speed. MLP also cached. Some quality loss. |

### Cache Dtype Selection

| Dtype | Memory | Precision | Recommended For |
|-------|--------|-----------|-----------------|
| `fp16` | Baseline | Full | Default. Use unless VRAM constrained. |
| `fp8` (e4m3fn) | ~50% of fp16 | Reduced | VRAM-limited GPUs (8GB class) |
| `mixed` | ~75% of fp16 | Early=full, Late=reduced | Balance: precision where it matters, savings where it doesn't |

In `mixed` mode, blocks with global index < 8 use fp16 and blocks >= 8 use fp8.


## Comparison with Existing Flux Acceleration

| Feature | Shannon-Prime | Naive Block Skip | TeaCache / DeepCache |
|---------|--------------|------------------|---------------------|
| **Drift detection** | Fisher-weighted spectral similarity | None or L2 distance | Cosine similarity (unweighted) |
| **Gate correction** | Recomputes adaLN modulation per step | None (stale gate) | Varies |
| **Dual-stream aware** | Caches img + txt residuals separately | Often single-stream only | Architecture-specific |
| **MLP handling** | Fresh by default, TURBO caches | Always cached or always skipped | Always cached |
| **Lattice RoPE** | 2D factored lattice frequencies | Standard geometric RoPE | Standard geometric RoPE |
| **Streak limiting** | Prevents runaway cache reuse | No safeguard | Step-count based |
| **Dtype flexibility** | fp16 / fp8 / mixed per-block | Fixed | Fixed |

Shannon-Prime's key advantages:

1. **Spectral-aware drift detection:** Fisher weighting by squarefree structure
   suppresses false invalidations from perceptually inert high-frequency noise,
   allowing longer cache windows without quality loss.

2. **Modulation gate reapplication:** On every cache hit, the current timestep's
   modulation gate is recomputed and applied. This keeps cached outputs
   sigma-accurate even as the noise schedule progresses.

3. **Tier-aware block mapping:** Different cache aggressiveness per block tier
   based on the Granite/Jazz stability principle, rather than uniform caching
   across all blocks.

4. **2D lattice RoPE injection:** Spatial frequency decomposition via
   `_tiered_lattice_factors` aligns the position encoding with the VHT2 prime
   harmonic structure at zero runtime cost (one-time patch).


## Validation

The Flux integration is covered by the unified test suite at
`tests/unified/test_flux_integration.py` (13 sections, 40+ assertions):

- Node registration and `NODE_CLASS_MAPPINGS`
- `INPUT_TYPES` validation with default value checks
- Block iterator behaviour on None/empty models
- Head dimension detection with fallback to Flux v1 defaults
- 2D lattice RoPE factor computation (temporal vs spatial tiers)
- Fisher diagonal weights at both head_dim=128 (v1) and head_dim=64 (v2)
- Fisher cosine similarity identity and divergence properties
- Input fingerprinting determinism
- CacheFlush node passthrough behaviour
- Flux v2 compatibility (axes_dim=[32,32,32,32])
- VHT2 self-inverse roundtrip at both head dimensions
- Architecture invariants (head_dim = hidden_size / num_heads, axes_dim sum)

```bash
cd shannon-prime
python tests/unified/run_tests.py --suite flux
```


## File Locations

| File | Purpose |
|------|---------|
| `shannon-prime-comfyui/nodes/shannon_prime_flux_nodes.py` | Node implementations (3 nodes + helpers) |
| `shannon-prime-comfyui/nodes/shannon_prime_nodes.py` | Shared infrastructure (Fisher weights, lattice factors, VHT2 pool) |
| `shannon-prime/tests/unified/test_flux_integration.py` | Unified test suite for Flux |
| `shannon-prime/docs/INTEGRATION-COMFYUI.md` | Wan video modality (companion doc) |
