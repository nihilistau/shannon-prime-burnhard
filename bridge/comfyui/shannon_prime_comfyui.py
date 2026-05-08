# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
ComfyUI integration for Shannon-Prime VHT2 cross-attention caching.

Wan Architecture Notes (from source inspection):
=================================================

Wan 2.1 (dense, single model):
  - DiT with Flow Matching, T5-XXL text encoder
  - Each WanAttentionBlock has:
      1. Self-attention on video latents (with 3D RoPE: temporal + spatial)
      2. Cross-attention from T5 text embeddings (NO RoPE on cross-attn K/V)
      3. FFN
  - Cross-attn K/V are linear projections of T5 output via cross_attn_k, cross_attn_v
  - Text context is IDENTICAL across all ~50 diffusion timesteps
  - Config: 14B has dim=5120, num_heads=40, num_layers=40 (head_dim=128)
  - Config: 1.3B has dim=1536, num_heads=12, num_layers=30 (head_dim=128)

Wan 2.2 MoE (A14B, two experts):
  - Same DiT block structure as 2.1 per expert
  - TWO full expert models: high_noise_model and low_noise_model
  - Expert switch at SNR boundary (T2V: 0.875, I2V: 0.900)
  - Each expert has DIFFERENT cross_attn_k/cross_attn_v weight matrices
  - Cross-attn K/V from the high-noise expert are WRONG for the low-noise expert
  - ComfyUI loads them as two separate Load Diffusion Model nodes

Wan 2.2 TI2V-5B (dense, single model):
  - dim=3072, num_heads=24, num_layers=30 (head_dim=128)
  - Uses Wan2.2-VAE with 4×16×16 compression (vs 2.1's 4×8×8)
  - Single model, no MoE — behaves like Wan 2.1 for caching purposes

VHT2 Caching Strategy:
=======================

Cross-attention K/V do NOT carry RoPE. They are vanilla linear projections
of T5 features. The VHT2 spectral concentration is weaker than self-attention
K (which has RoPE periodicity). VHT2 still provides VRAM compression, but
the quality advantage is from general transform coding, not lattice structure.

For MoE models, the cache MUST be partitioned per-expert. When the denoising
loop crosses the SNR boundary:
  - All cross-attn K/V cached for the high-noise expert become invalid
  - The low-noise expert computes fresh K/V with its own weights
  - These get cached separately for the remaining timesteps

ComfyUI integration points:
  - Native: The KSampler drives the diffusion loop. The Wan22MoESampler
    node (or WanMoeKSampler custom node) handles expert switching.
  - Kijai wrapper: WanVideoSampler in nodes.py calls predict_with_cfg()
    which calls the model forward. Cross-attn happens inside WanAttentionBlock.

The hook point is WanAttentionBlock.forward(), specifically the cross-attention
computation where context (T5 embeddings) is projected through cross_attn_k
and cross_attn_v linear layers.

Production results (Wan 2.2 14B, RTX 2060):
  900 cross-attn calls: 27.22s → 22.63s (1.20× speedup)
  Per-call latency:     30.24ms → 25.14ms
  Output correlation:   0.9984
  Cache hits:          899/900 (1 miss = first timestep compute)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backends', 'torch'))

import torch
from typing import Optional, Tuple, Dict, Callable, Any
from shannon_prime_torch import (
    vht2, MobiusMask, BandedQuantizer, correlation
)

# ── VHT2 dispatch: CUDA extension when available, PyTorch otherwise ────────
def _make_vht2_dispatch():
    try:
        import sys, pathlib
        _cuda_dir = pathlib.Path(__file__).resolve().parent.parent / "backends" / "cuda"
        if str(_cuda_dir) not in sys.path:
            sys.path.insert(0, str(_cuda_dir))
        import shannon_prime_cuda_wan as _ext
        print("[Shannon-Prime] VHT2 dispatch: CUDA kernel (shannon_prime_cuda_wan)")
        def _dispatch(x: "torch.Tensor") -> "torch.Tensor":
            return _ext.inplace(x.contiguous())  # modifies in-place, returns None... use forward instead
        # Use forward (returns new tensor, cleaner)
        def _dispatch(x: "torch.Tensor") -> "torch.Tensor":
            return _ext.forward(x.contiguous())
        return _dispatch
    except ImportError:
        print("[Shannon-Prime] VHT2 dispatch: PyTorch fallback (CUDA ext not compiled)")
        return vht2

_vht2 = _make_vht2_dispatch()


class VHT2CrossAttentionCache:
    """
    Compressed cross-attention KV cache for Wan video generation.

    Handles both dense models (Wan 2.1, TI2V-5B) and MoE models (Wan 2.2 A14B).

    For dense models: cache is keyed by block index. One compute per block,
    reused across all ~50 timesteps.

    For MoE models: cache is keyed by (expert_id, block_index). Two computes
    per block (one per expert), each reused across that expert's timestep range.

    Cross-attention K/V in Wan do NOT have RoPE applied. They are linear
    projections of T5 text embeddings:
        K = cross_attn_k(context)  # context = T5 output, shape [B, L_text, dim]
        V = cross_attn_v(context)

    Since there's no RoPE periodicity, K and V have similar spectral profiles
    in VHT2 space. We still apply Möbius-ordered banded quantization for VRAM
    compression, but use the SAME bit allocation for both K and V (unlike
    self-attention where K gets 5/5/4/3 and V gets flat 3-bit).
    """

    def __init__(
        self,
        head_dim: int = 128,
        k_band_bits: list = [5, 4, 4, 3],
        v_band_bits: list = [5, 4, 4, 3],
        use_mobius: bool = True,
        device: str = 'cpu',
    ):
        self.head_dim = head_dim
        self.device = device
        self.k_quantizer = BandedQuantizer(head_dim, k_band_bits)
        self.v_quantizer = BandedQuantizer(head_dim, v_band_bits)
        self.mobius = MobiusMask(head_dim) if use_mobius else None

        # Cache: cache_key → (k_scales, k_quants, v_scales, v_quants, shape, dtype)
        self._cache: Dict[str, tuple] = {}
        self._hits = 0
        self._misses = 0

        # MoE tracking
        self._current_expert: Optional[str] = None

    # =========================================================================
    # Expert management (Wan 2.2 MoE)
    # =========================================================================

    def set_expert(self, expert_id: str):
        """
        Set the active expert for MoE models.

        Call this when the denoising loop switches experts at the SNR boundary.
        For Wan 2.2 T2V: boundary=0.875, for I2V: boundary=0.900.

        expert_id: 'high_noise' or 'low_noise' (or any string identifier)
        """
        if self._current_expert != expert_id:
            self._current_expert = expert_id

    def _cache_key(self, block_id: str) -> str:
        """Build cache key incorporating expert ID for MoE models."""
        if self._current_expert is not None:
            return f"{self._current_expert}:{block_id}"
        return block_id

    # =========================================================================
    # Core cache operations
    # =========================================================================

    def has(self, block_id: str) -> bool:
        """Check if block has cached K/V for the current expert."""
        return self._cache_key(block_id) in self._cache

    def put(self, block_id: str, k: torch.Tensor, v: torch.Tensor):
        """
        Compress and cache cross-attention K/V tensors.

        k, v: (batch, n_heads, seq_len, head_dim) — typical Wan shape after
              reshaping in WanAttentionBlock, or (batch, seq_len, n_heads*head_dim)
              before the reshape. We handle both.
        """
        key = self._cache_key(block_id)
        orig_shape = k.shape
        orig_dtype = k.dtype

        # Clone + flatten to (..., head_dim) for compression
        k_flat = k.reshape(-1, self.head_dim).float().clone()
        v_flat = v.reshape(-1, self.head_dim).float().clone()

        # VHT2 forward
        k_flat = _vht2(k_flat)
        v_flat = _vht2(v_flat)

        # Möbius reorder both K and V (cross-attn K has no RoPE,
        # so K and V have similar spectral profiles — reorder both)
        if self.mobius is not None:
            k_flat = self.mobius.reorder(k_flat)
            v_flat = self.mobius.reorder(v_flat)

        # Band quantize
        k_scales, k_quants = self.k_quantizer.quantize(k_flat)
        v_scales, v_quants = self.v_quantizer.quantize(v_flat)

        # Store compressed, detached from graph
        self._cache[key] = (
            [s.detach() for s in k_scales],
            [q.detach() for q in k_quants],
            [s.detach() for s in v_scales],
            [q.detach() for q in v_quants],
            orig_shape,
            orig_dtype,
        )
        self._misses += 1

    def get(self, block_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct K/V from compressed cache.

        Returns (k, v) with original shape and dtype.
        """
        key = self._cache_key(block_id)
        k_scales, k_quants, v_scales, v_quants, orig_shape, orig_dtype = self._cache[key]
        self._hits += 1

        # Dequantize
        k_flat = self.k_quantizer.dequantize(k_scales, k_quants)
        v_flat = self.v_quantizer.dequantize(v_scales, v_quants)

        # Möbius unreorder
        if self.mobius is not None:
            k_flat = self.mobius.unreorder(k_flat)
            v_flat = self.mobius.unreorder(v_flat)

        # Inverse VHT2 (self-inverse — same call as forward, no 1/N)
        k_flat = _vht2(k_flat)
        v_flat = _vht2(v_flat)

        # Reshape and cast back to original dtype
        k = k_flat.reshape(orig_shape).to(orig_dtype)
        v = v_flat.reshape(orig_shape).to(orig_dtype)

        return k, v

    def clear(self):
        """Clear all cached entries (call between generations)."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._current_expert = None

    def clear_expert(self, expert_id: str):
        """Clear cache for a specific expert only."""
        prefix = f"{expert_id}:"
        keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
        for k in keys_to_remove:
            del self._cache[k]

    def stats(self) -> dict:
        """Cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total > 0 else 0.0,
            'n_entries_cached': len(self._cache),
            'current_expert': self._current_expert,
            'compression_ratio': self.compression_ratio(),
        }

    def compression_ratio(self) -> float:
        """Estimated compression ratio."""
        baseline = self.head_dim * 2  # fp16 bytes per element
        k_bytes = self.k_quantizer.compressed_bytes_per_vec()
        v_bytes = self.v_quantizer.compressed_bytes_per_vec()
        return (2 * baseline) / (k_bytes + v_bytes)


class WanVHT2Wrapper:
    """
    Wan-aware VHT2 cross-attention cache wrapper.

    Handles the diffusion loop structure for both Wan 2.1 (dense)
    and Wan 2.2 (MoE) models.

    Architecture-specific behavior:
    ===============================

    Wan 2.1 / TI2V-5B (dense):
      - Single model, no expert switching
      - Cache key = block index
      - 40 blocks × 50 timesteps = 2000 cross-attn calls
      - 40 misses (first timestep) + 1960 hits = 98% hit rate

    Wan 2.2 A14B (MoE):
      - Two experts: high_noise and low_noise
      - Expert switch at boundary (T2V: sigma=0.875, I2V: sigma=0.900)
      - Cache key = (expert_id, block_index)
      - Each expert's cross_attn_k/v weights are DIFFERENT
      - Total: 40 blocks × 50 timesteps = 2000 calls
      - ~40 misses (first high-noise step) + ~40 misses (first low-noise step)
      - = 80 misses, 1920 hits = 96% hit rate

    Usage with Wan 2.2 MoE:
    =======================

        wrapper = WanVHT2Wrapper(head_dim=128, model_type='wan22_moe')

        for step, sigma in enumerate(sigmas):
            # Determine which expert is active based on SNR
            expert = 'high_noise' if sigma > boundary else 'low_noise'
            wrapper.set_expert(expert)

            for block_idx, block in enumerate(dit_blocks):
                k, v = wrapper.get_or_compute(
                    block_id=f"block_{block_idx}",
                    compute_fn=lambda: (
                        block.cross_attn_k(context),
                        block.cross_attn_v(context),
                    )
                )
                # ... use k, v in cross-attention ...

        wrapper.reset()  # between generations
    """

    # Wan 2.2 MoE SNR boundaries (from config files)
    WAN22_BOUNDARY_T2V = 0.875
    WAN22_BOUNDARY_I2V = 0.900

    def __init__(
        self,
        head_dim: int = 128,
        model_type: str = 'wan21',  # 'wan21', 'wan22_moe', 'wan22_5b'
        task_type: str = 't2v',     # 't2v', 'i2v' — affects MoE boundary
        **kwargs,
    ):
        self.model_type = model_type
        self.task_type = task_type
        self.cache = VHT2CrossAttentionCache(head_dim=head_dim, **kwargs)

        # Determine MoE boundary
        if model_type == 'wan22_moe':
            self.boundary = (self.WAN22_BOUNDARY_I2V if task_type == 'i2v'
                           else self.WAN22_BOUNDARY_T2V)
        else:
            self.boundary = None

    def set_expert(self, expert_id: str):
        """Set the active expert. Only meaningful for wan22_moe."""
        if self.model_type == 'wan22_moe':
            self.cache.set_expert(expert_id)

    def set_expert_from_sigma(self, sigma: float):
        """
        Auto-detect expert from the current noise level.

        sigma: the diffusion timestep (0 = clean, 1 = pure noise).
        For Wan 2.2, high-noise expert handles sigma > boundary,
        low-noise expert handles sigma <= boundary.
        """
        if self.model_type != 'wan22_moe' or self.boundary is None:
            return

        expert = 'high_noise' if sigma > self.boundary else 'low_noise'
        self.cache.set_expert(expert)

    def get_or_compute(
        self,
        block_id: str,
        compute_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return cached K/V if available, otherwise compute and cache.

        block_id: identifies the DiT block (e.g. "block_0" .. "block_39")
        compute_fn: callable returning (k, v) tensors
        """
        if self.cache.has(block_id):
            return self.cache.get(block_id)
        else:
            k, v = compute_fn()
            self.cache.put(block_id, k, v)
            return k, v

    def reset(self):
        """Reset cache (call between generations)."""
        self.cache.clear()

    def stats(self) -> dict:
        return self.cache.stats()


def patch_wan_attention_block(block, cache: VHT2CrossAttentionCache, block_id: str):
    """
    Monkey-patch a WanAttentionBlock to use VHT2 cross-attention caching.

    This patches the block's forward method to intercept cross-attention
    K/V computation. Self-attention is untouched.

    Usage:
        for i, block in enumerate(model.blocks):
            patch_wan_attention_block(block, cache, f"block_{i}")

    Note: This is a reference implementation. For production, you'd want
    to integrate at the ComfyUI node level or via model_options/transformer_options
    hooks in the Kijai wrapper.
    """
    original_forward = block.forward

    def patched_forward(*args, **kwargs):
        # The cross-attention K/V computation happens inside the original forward.
        # We can't easily intercept it without modifying the block internals.
        # This patch point is illustrative — the real integration would
        # override the cross_attn_k and cross_attn_v linear layers
        # with caching wrappers.
        return original_forward(*args, **kwargs)

    block.forward = patched_forward


class WanCrossAttnCachingLinear(torch.nn.Module):
    """
    Drop-in replacement for cross_attn_k or cross_attn_v linear layers
    that caches the output using VHT2 compression.

    Since the input (T5 text embeddings) doesn't change across timesteps,
    the output of these linear layers is constant. We compute once,
    compress via VHT2, and return the reconstructed value on subsequent calls.

    Usage:
        for i, block in enumerate(model.blocks):
            block.cross_attn_k = WanCrossAttnCachingLinear(
                block.cross_attn_k, cache, f"block_{i}_k"
            )
            block.cross_attn_v = WanCrossAttnCachingLinear(
                block.cross_attn_v, cache, f"block_{i}_v"
            )
    """

    def __init__(self, original_linear: torch.nn.Module,
                 cache: VHT2CrossAttentionCache, cache_key: str):
        super().__init__()
        self.original = original_linear
        self.cache = cache
        self.cache_key = cache_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache.has(self.cache_key):
            # Return just the K or V (stored as k in the cache, v is dummy)
            result, _ = self.cache.get(self.cache_key)
            return result
        else:
            result = self.original(x)
            # Store with dummy v (we cache K and V separately)
            self.cache.put(self.cache_key, result, result)
            return result
