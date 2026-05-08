# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
ComfyUI sqfree + spinor integration for Wan 2.1/2.2 cross-attention.

Additive to tools/shannon_prime_comfyui.py — extends the Wan VHT2 wrapper
with the aggressive sqfree compression path.

Usage:
    from shannon_prime_comfyui_sqfree import WanSqfreeVHT2Wrapper

    wrapper = WanSqfreeVHT2Wrapper(
        head_dim=128,
        model_type='wan22_moe',
        task_type='t2v',
        use_spinor=True,        # default True
        residual_bits=3,        # default 3 (Pareto point)
        band_bits=[3,3,3,3,3],  # aggressive compression
    )

    for step, sigma in enumerate(sigmas):
        wrapper.set_expert_from_sigma(sigma)
        for block_idx in range(40):
            k, v = wrapper.get_or_compute(
                block_id=f"block_{block_idx}",
                compute_fn=lambda: (
                    block.cross_attn_k(context),
                    block.cross_attn_v(context),
                )
            )

Scaling-law prediction for Wan 2.2 14B bf16 (hd=128):
    K_corr floor for 3% PPL budget: 0.914 (very loose)
    Expected compression @ spinor 3/3/3/3/3: ~3.3× at K_corr ~0.97
    → predicted PPL impact: <0.5% (well within budget)

Environment variables (all opt-in):
    SHANNON_PRIME_SQFREE=1       Enable sqfree basis
    SHANNON_PRIME_SPINOR=1       Enable spinor sheet bit
    SHANNON_PRIME_RESIDUAL_BITS=3  Residual quant depth (1-4)
    SHANNON_PRIME_BAND_BITS=3,3,3,3,3  Band allocation
"""

import os
import torch
from typing import Callable, Tuple, Optional, List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backends.torch.shannon_prime_sqfree import (
    SqfreeShadowCache,
)
from backends.torch.shannon_prime_torch import correlation
from tools.sp_scaling_law import (
    predicted_ppl_ratio,
    is_pareto_viable,
    min_k_corr_for_budget,
)


class WanSqfreeVHT2Wrapper:
    """
    Sqfree + spinor VHT2 cached wrapper for Wan cross-attention.

    Same interface as WanVHT2Wrapper from shannon_prime_comfyui.py,
    but uses the sqfree prime-Hartley basis + Möbius CSR + spinor
    for higher compression at equivalent quality.

    Important: cross-attention K/V in Wan have NO RoPE. Both K and V
    are vanilla linear projections of T5 text embeddings. Both get
    the same compression treatment (same band_bits for K and V).
    """

    # Expert switching boundaries (sigma thresholds)
    BOUNDARY_T2V = 0.875
    BOUNDARY_I2V = 0.900

    def __init__(
        self,
        head_dim: int = 128,
        model_type: str = 'wan22_moe',
        task_type: str = 't2v',
        use_spinor: bool = True,
        residual_bits: int = 3,
        band_bits: Optional[List[int]] = None,
        sk_frac: float = 0.75,
        n_blocks: int = 40,
        device: str = 'cpu',
    ):
        # Environment overrides
        if os.environ.get('SHANNON_PRIME_SQFREE') == '1':
            pass  # Already sqfree by being in this class
        if os.environ.get('SHANNON_PRIME_SPINOR') == '1':
            use_spinor = True
        elif os.environ.get('SHANNON_PRIME_SPINOR') == '0':
            use_spinor = False
        if os.environ.get('SHANNON_PRIME_RESIDUAL_BITS'):
            residual_bits = int(os.environ['SHANNON_PRIME_RESIDUAL_BITS'])
        if os.environ.get('SHANNON_PRIME_BAND_BITS'):
            band_bits = [int(x) for x in os.environ['SHANNON_PRIME_BAND_BITS'].split(',')]

        if band_bits is None:
            band_bits = [3, 3, 3, 3, 3]  # Aggressive default for sqfree

        self.head_dim = head_dim
        self.model_type = model_type
        self.task_type = task_type
        self.n_blocks = n_blocks
        self.device = device
        self.use_spinor = use_spinor
        self.residual_bits = residual_bits
        self.band_bits = band_bits

        # MoE handling
        self.is_moe = 'moe' in model_type
        self.boundary = self.BOUNDARY_T2V if task_type == 't2v' else self.BOUNDARY_I2V
        self.current_expert = 'high_noise'

        # One sqfree cache per expert
        # n_layers=1, n_heads_kv=n_blocks since we key by block not layer/head
        self._caches = {}
        self._make_cache('high_noise')
        if self.is_moe:
            self._make_cache('low_noise')

        # Track hits/misses
        self._hits = 0
        self._misses = 0
        self._computed = set()

        # Scaling law check
        # Wan 2.2 14B bf16: params=14.0, bits=16
        params_b = 14.0 if '14' in model_type else 5.0
        bits = 16  # bf16
        floor = min_k_corr_for_budget(params_b, bits, 0.03)
        self._k_corr_floor = floor

    def _make_cache(self, expert_name: str):
        """Create a sqfree cache for one expert."""
        self._caches[expert_name] = SqfreeShadowCache(
            head_dim=self.head_dim,
            n_layers=1,
            n_heads_kv=self.n_blocks,
            max_seq_len=2,  # Only need 1 position per block (cross-attn is static)
            band_bits=self.band_bits,
            residual_bits=self.residual_bits,
            use_spinor=self.use_spinor,
            sk_frac=0.75,
            device=self.device,
        )

    def set_expert_from_sigma(self, sigma: float):
        """Auto-detect MoE expert from noise level."""
        if not self.is_moe:
            return
        self.current_expert = 'high_noise' if sigma > self.boundary else 'low_noise'

    def get_or_compute(
        self,
        block_id: str,
        compute_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached K/V or compute, compress, and cache them.

        Args:
            block_id: Unique identifier (e.g. "block_0")
            compute_fn: Callable that returns (K, V) tensors

        Returns:
            (K, V) tensors — either from cache or freshly computed
        """
        cache_key = f"{self.current_expert}:{block_id}"

        if cache_key in self._computed:
            # Cache hit — reconstruct from compressed storage
            self._hits += 1
            block_idx = int(block_id.split('_')[-1]) if '_' in block_id else 0
            cache = self._caches[self.current_expert]
            k = cache.read_k(layer=0, head=block_idx, pos=0)
            v = cache.read_v(layer=0, head=block_idx, pos=0)
            return k, v
        else:
            # Cache miss — compute, compress, store
            self._misses += 1
            k, v = compute_fn()

            block_idx = int(block_id.split('_')[-1]) if '_' in block_id else 0
            cache = self._caches[self.current_expert]

            # Compress per-head vectors
            # Cross-attention K/V are typically (batch, heads, seq, head_dim)
            # or flattened. Handle the common shapes.
            if k.dim() == 1 and k.shape[0] == self.head_dim:
                cache.write_k(layer=0, head=block_idx, pos=0, k_vec=k)
                cache.write_v(layer=0, head=block_idx, pos=0, v_vec=v)
            else:
                # For multi-head: compress the first head as representative
                # Full per-head compression would iterate over heads
                flat_k = k.reshape(-1, self.head_dim)[0]
                flat_v = v.reshape(-1, self.head_dim)[0]
                cache.write_k(layer=0, head=block_idx, pos=0, k_vec=flat_k)
                cache.write_v(layer=0, head=block_idx, pos=0, v_vec=flat_v)

            self._computed.add(cache_key)
            return k, v

    def reset(self):
        """Reset cache between generations."""
        self._computed.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """Runtime statistics."""
        total = self._hits + self._misses
        return {
            'hits': self._hits,
            'misses': self._misses,
            'total': total,
            'hit_rate': self._hits / max(total, 1),
            'expert': self.current_expert,
            'compression_ratio': self._caches[self.current_expert].compression_ratio(),
            'spinor': self.use_spinor,
            'residual_bits': self.residual_bits,
            'band_bits': self.band_bits,
            'k_corr_floor': self._k_corr_floor,
        }


def _input_fingerprint(x: torch.Tensor):
    """Content hash for cache-invalidation — identical semantics to the ship
    wrapper's `_input_fingerprint` in `shannon_prime_comfyui.py`. Three flat
    anchors + shape + dtype distinguish "same context across timesteps" from
    "fresh context for a new generation" without scanning the full tensor."""
    flat = x.view(-1) if x.is_contiguous() else x.reshape(-1)
    n = flat.numel()
    i_mid = n // 2 if n > 1 else 0
    i_end = n - 1 if n > 0 else 0
    return (
        tuple(x.shape),
        x.dtype,
        float(flat[0].item()),
        float(flat[i_mid].item()),
        float(flat[i_end].item()),
    )


class VHT2SqfreeCrossAttentionCache:
    """Sqfree + spinor VHT2 cache for Wan cross-attention K/V tensors.

    Mirrors the put/get contract of `VHT2CrossAttentionCache` in
    `shannon_prime_comfyui.py` but compresses each `(N, head_dim)` row
    through the sqfree pipeline (squarefree pad → VHT2 → Knight skeleton
    → banded quantize → Möbius CSR predict → 3-bit residual → spinor sheet
    bit). Entries are stored in a plain Python dict keyed by
    `(expert, block_id)` so arbitrary cross-attn shapes are supported
    without pre-sizing the internal slot count.
    """

    def __init__(
        self,
        head_dim: int = 128,
        use_spinor: bool = True,
        residual_bits: int = 3,
        band_bits: Optional[List[int]] = None,
        # `max_blocks` kept for backwards compat with the older API; it is
        # unused now because storage is a dict, not a fixed slot array.
        max_blocks: Optional[int] = None,
    ):
        if band_bits is None:
            band_bits = [3, 3, 3, 3, 3]
        del max_blocks  # intentionally unused

        # A SqfreeShadowCache instance is used only for its mask / primes /
        # quantizers / per-vector compress+reconstruct methods. Storage
        # inside it is a 1×1×1 scratch area that's never indexed — actual
        # cache entries live in `self._entries`.
        self._compressor = SqfreeShadowCache(
            head_dim=head_dim,
            n_layers=1,
            n_heads_kv=1,
            max_seq_len=1,
            band_bits=band_bits,
            residual_bits=residual_bits,
            use_spinor=use_spinor,
        )
        self._head_dim = head_dim
        # {cache_key: (orig_shape, orig_dtype, orig_device,
        #              [k_compressed_per_row], [v_compressed_per_row])}
        self._entries: dict = {}
        self._expert = 'default'
        self._hits = 0
        self._misses = 0

    # ----- put/get API (matches the ship-path VHT2CrossAttentionCache) ----

    def _cache_key(self, block_id: str) -> str:
        return f"{self._expert}:{block_id}"

    def has(self, block_id: str) -> bool:
        return self._cache_key(block_id) in self._entries

    def put(self, block_id: str, k: torch.Tensor, v: torch.Tensor):
        """Store the FULL K and V tensors. Reshapes to `(N, head_dim)` and
        compresses each row via the sqfree pipeline."""
        key = self._cache_key(block_id)
        hd = self._head_dim
        orig_shape = k.shape
        orig_dtype = k.dtype
        orig_device = k.device

        k_flat = k.reshape(-1, hd).float().contiguous()
        v_flat = v.reshape(-1, hd).float().contiguous()

        k_rows = [
            self._compressor._compress_vec(k_flat[i], self._compressor.k_quantizer)
            for i in range(k_flat.shape[0])
        ]
        v_rows = [
            self._compressor._compress_vec(v_flat[i], self._compressor.v_quantizer)
            for i in range(v_flat.shape[0])
        ]
        self._entries[key] = (orig_shape, orig_dtype, orig_device, k_rows, v_rows)
        self._misses += 1

    def get(self, block_id: str):
        """Reconstruct the FULL K and V tensors at their original shape/dtype."""
        key = self._cache_key(block_id)
        orig_shape, orig_dtype, orig_device, k_rows, v_rows = self._entries[key]
        self._hits += 1

        k_recon = torch.stack([
            self._compressor._reconstruct_vec(c, self._compressor.k_quantizer)
            for c in k_rows
        ], dim=0)
        v_recon = torch.stack([
            self._compressor._reconstruct_vec(c, self._compressor.v_quantizer)
            for c in v_rows
        ], dim=0)

        k = k_recon.reshape(orig_shape).to(dtype=orig_dtype, device=orig_device)
        v = v_recon.reshape(orig_shape).to(dtype=orig_dtype, device=orig_device)
        return k, v

    def set_expert(self, expert_name: str):
        self._expert = expert_name

    def reset(self):
        self._entries.clear()
        self._hits = 0
        self._misses = 0

    def clear_expert(self, expert_name: str):
        prefix = f"{expert_name}:"
        for k in list(self._entries.keys()):
            if k.startswith(prefix):
                del self._entries[k]

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            'hits': self._hits,
            'misses': self._misses,
            'total': total,
            'hit_rate': self._hits / max(total, 1),
            'n_entries_cached': len(self._entries),
            'compression_ratio': (
                self._compressor.compression_ratio()
                if hasattr(self._compressor, 'compression_ratio') else 0.0
            ),
        }


class WanSqfreeCrossAttnCachingLinear(torch.nn.Module):
    """Drop-in replacement for Wan cross_attn.k / .v (and .k_img / .v_img).

    Uses content fingerprinting (not pointer identity) to decide whether
    the upstream T5/UMT5 context matches the last seen input: ComfyUI's
    sampler re-allocates the conditioning tensor each diffusion timestep,
    so a data_ptr() check would invalidate every step.
    """

    def __init__(self, linear: torch.nn.Linear,
                 cache: 'VHT2SqfreeCrossAttentionCache',
                 cache_key: str):
        super().__init__()
        self.linear = linear
        self.cache = cache
        self.cache_key = cache_key
        self._last_fp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fp = _input_fingerprint(x)

        # Cache hit — return the sqfree-reconstructed tensor (K slot;
        # the current wrapper caches a single linear per (block, k/v) key,
        # so the K-slot content is the correct output for that linear).
        if fp == self._last_fp and self.cache.has(self.cache_key):
            k_recon, _ = self.cache.get(self.cache_key)
            return k_recon

        # Miss — compute, cache, remember fingerprint.
        result = self.linear(x)
        self.cache.put(self.cache_key, result, result)
        self._last_fp = fp
        return result


def probe_wan_cache_shape(model) -> dict:
    """Walk a Wan ModelPatcher, introspect the first cross-attn block,
    and report the per-call K/V tensor shape expected by the cross-attn
    cache. Useful for sizing budget estimates ahead of time — the cache
    grows on demand so this is informational, not required.
    """
    inner = getattr(model, 'model', model)
    diff = getattr(inner, 'diffusion_model', None) or getattr(inner, 'model', None)
    if diff is None or not hasattr(diff, 'blocks'):
        return {'n_blocks': 0, 'head_dim': None, 'per_block_linears': 0}
    blocks = list(diff.blocks)
    if not blocks:
        return {'n_blocks': 0, 'head_dim': None, 'per_block_linears': 0}
    blk0 = blocks[0]
    head_dim = getattr(getattr(blk0, 'self_attn', None), 'head_dim', None)
    cross = getattr(blk0, 'cross_attn', None)
    per_block = 0
    if cross is not None:
        for attr in ('k', 'v', 'k_img', 'v_img'):
            if getattr(cross, attr, None) is not None:
                per_block += 1
    return {
        'n_blocks': len(blocks),
        'head_dim': head_dim,
        'per_block_linears': per_block,
        'estimated_entries': len(blocks) * per_block,
    }