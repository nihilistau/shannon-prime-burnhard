# Shannon-Prime: ComfyUI Integration Guide

## What This Does

Shannon-Prime provides VHT2 compressed caching for cross-attention K/V in video generation models. In Wan 2.1/2.2, cross-attention recomputes K/V from identical T5 text embeddings every timestep across every DiT block. Shannon-Prime computes once, compresses via VHT2, and reconstructs on subsequent calls.

Production results on Wan 2.2 14B (RTX 2060): 1.20× cross-attention speedup, 0.9984 output correlation, 899/900 cache hits per generation.

## How Wan Models Use Cross-Attention

```
Wan DiT Block (×40 for 14B, ×30 for 5B)
════════════════════════════════════════

  Video Latents (x)
       │
       ├──► Self-Attention (3D RoPE on Q,K)
       │         Uses RoPE → K has spectral structure
       │
       ├──► Cross-Attention ◄── T5 Text Embeddings (context)
       │         │
       │    K = cross_attn_k(context)  ← Linear projection, NO RoPE
       │    V = cross_attn_v(context)  ← Linear projection, NO RoPE
       │         │
       │    ┌────▼──────────────────────────────────┐
       │    │  Shannon-Prime Cache                   │
       │    │                                        │
       │    │  Timestep 1: COMPUTE K,V → compress    │
       │    │  Timestep 2-50: RECONSTRUCT from cache │
       │    └────────────────────────────────────────┘
       │
       ├──► FFN
       │
       ▼
  Output (to next block)
```

**Important architectural detail:** Cross-attention K/V in Wan do NOT have RoPE applied. They are vanilla linear projections of T5 text embeddings. This means K and V have similar spectral profiles (unlike self-attention where K has RoPE periodicity). Shannon-Prime handles this by applying Möbius reordering and matching bit allocation to both K and V.

## Wan Model Variants

| Model | Architecture | Expert Handling | Cache Behavior |
|-------|-------------|-----------------|----------------|
| Wan 2.1 14B | Dense, single DiT | None | 40 blocks × 1 compute each = 40 misses, rest hits |
| Wan 2.1 1.3B | Dense, single DiT | None | 30 blocks × 1 compute each |
| Wan 2.2 A14B T2V | MoE, 2 experts | Switch at σ=0.875 | 40 blocks × 2 experts = 80 misses |
| Wan 2.2 A14B I2V | MoE, 2 experts | Switch at σ=0.900 | 40 blocks × 2 experts = 80 misses |
| Wan 2.2 TI2V-5B | Dense, single DiT | None | 30 blocks × 1 compute each |

### MoE Expert Switching (Wan 2.2 A14B)

Wan 2.2's MoE uses two full expert DiT models:

- **High-noise expert:** Handles early denoising (σ > boundary). Focuses on overall layout.
- **Low-noise expert:** Handles late denoising (σ ≤ boundary). Refines details.

Each expert has **different** `cross_attn_k` and `cross_attn_v` weight matrices. K/V cached from the high-noise expert are WRONG for the low-noise expert. Shannon-Prime keys its cache by `(expert_id, block_index)` to handle this correctly.

```
Denoising Timeline (50 steps, T2V):
═══════════════════════════════════════════

  σ=1.0                    σ=0.875                    σ=0.0
   │     High-Noise Expert    │    Low-Noise Expert      │
   │                          │                          │
   │  Step 1: COMPUTE all 40  │  First low-noise step:   │
   │  cross-attn K/V, cache   │  COMPUTE all 40 K/V,     │
   │  under "high_noise:*"    │  cache under "low_noise:*"│
   │                          │                          │
   │  Steps 2-N: RECONSTRUCT  │  Remaining: RECONSTRUCT  │
   │  from high_noise cache   │  from low_noise cache    │
   │                          │                          │
   ▼──────────────────────────▼──────────────────────────▼
```

## Usage

### Basic (Wan 2.1 or TI2V-5B)

```python
from shannon_prime_comfyui import WanVHT2Wrapper

wrapper = WanVHT2Wrapper(head_dim=128, model_type='wan21')

for timestep in range(50):
    for block_idx in range(40):
        k, v = wrapper.get_or_compute(
            block_id=f"block_{block_idx}",
            compute_fn=lambda: (
                block.cross_attn_k(context),
                block.cross_attn_v(context),
            )
        )
        # Use k, v in cross-attention...

wrapper.reset()  # Between generations
```

### MoE-Aware (Wan 2.2 A14B)

```python
from shannon_prime_comfyui import WanVHT2Wrapper

wrapper = WanVHT2Wrapper(
    head_dim=128,
    model_type='wan22_moe',
    task_type='t2v',  # or 'i2v' — affects boundary (0.875 vs 0.900)
)

for step, sigma in enumerate(sigmas):
    # Auto-detect expert from noise level
    wrapper.set_expert_from_sigma(sigma)

    for block_idx in range(40):
        k, v = wrapper.get_or_compute(
            block_id=f"block_{block_idx}",
            compute_fn=lambda: compute_cross_attn(block_idx, context)
        )

wrapper.reset()
```

### Linear Layer Replacement (Deepest Integration)

Replace the actual `cross_attn_k` and `cross_attn_v` linear layers with caching wrappers:

```python
from shannon_prime_comfyui import VHT2CrossAttentionCache, WanCrossAttnCachingLinear

cache = VHT2CrossAttentionCache(head_dim=128)

# Patch each DiT block
for i, block in enumerate(model.blocks):
    block.cross_attn_k = WanCrossAttnCachingLinear(
        block.cross_attn_k, cache, f"block_{i}_k"
    )
    block.cross_attn_v = WanCrossAttnCachingLinear(
        block.cross_attn_v, cache, f"block_{i}_v"
    )

# For MoE: call cache.set_expert('high_noise') or cache.set_expert('low_noise')
# before each timestep group
```

## Configuration

```python
VHT2CrossAttentionCache(
    head_dim=128,           # Must match model (128 for 14B, 128 for 5B)
    k_band_bits=[5,4,4,3],  # Cross-attn K has no RoPE — use balanced allocation
    v_band_bits=[5,4,4,3],  # Cross-attn V gets same allocation (no asymmetry)
    use_mobius=True,         # Möbius reorder on both K and V
)
```

**Why cross-attention uses different defaults than self-attention:**

In self-attention, K has RoPE periodicity (spectral concentration → 5/5/4/3 for K, flat 3-bit for V). In cross-attention, K and V are both vanilla linear projections of T5 features — similar spectral profiles, no asymmetry. So both get the same balanced allocation.

## ComfyUI Node Integration

For integration into ComfyUI's node graph (either native Wan nodes or Kijai's WanVideoWrapper), the hook point is inside the `WanAttentionBlock.forward()` method, at the cross-attention computation where `context` (T5 embeddings) is projected through `cross_attn_k` and `cross_attn_v` linear layers.

The `WanCrossAttnCachingLinear` wrapper is the cleanest integration — it's a drop-in `nn.Module` replacement that requires no changes to the attention computation itself.

## Validation

```bash
cd shannon-prime
python3 tests/test_comfyui.py  # 25 tests
```

The test suite validates:

- Cross-attention cache put/get with correlation checks
- Wan 2.1 dense path (40 blocks × 50 timesteps, 98% hit rate)
- Wan 2.2 MoE expert switching (80 misses for 2 experts, 96% hit rate)
- Verification that different experts cache different K/V
- I2V vs T2V boundary handling
- TI2V-5B dense path
- Linear layer caching wrapper
- Expert-specific cache clearing
- Compression ratio
