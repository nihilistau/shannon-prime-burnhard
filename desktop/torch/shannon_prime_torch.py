# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
Pure PyTorch implementation of Shannon-Prime VHT2 KV cache compression.

This backend runs on any device PyTorch supports (CPU, CUDA, MPS).
It's the reference for ComfyUI integration and the easiest to test against.

Usage:
    from shannon_prime_torch import ShadowCache

    cache = ShadowCache(head_dim=128, n_layers=32, n_heads_kv=8)
    cache.write_k(layer=0, head=0, pos=0, k_vec=tensor)
    k_reconstructed = cache.read_k(layer=0, head=0, pos=0)
"""

import torch
import math
from typing import Optional, Tuple, List


# =============================================================================
# VHT2 — Vilenkin-Hartley Transform (the single transform)
# =============================================================================
#
# VHT2 is the squarefree-prime-factor generalization of the Walsh-Hadamard
# Transform. Each stage applies a p x p Hartley kernel — cas(θ) = cos(θ)+sin(θ)
# — normalized by 1/√p. At p=2 the stage is exactly the Hadamard butterfly divided
# by √2, and log2(N) stacked stages give an orthonormal transform with
# VHT2(VHT2(x)) = x (no division by N on the inverse).
#
# On any dimension that factors into the supported primes {2,3,5,7,11} the
# transform works directly. Non-factoring dimensions are handled by the
# sqfree_pad helper below (head_dim -> next small-prime sqfree multiple).

_SMALL_PRIMES: Tuple[int, ...] = (2, 3, 5, 7, 11)


def _factor_small_primes(n: int) -> List[int]:
    """Prime factorization of n into the supported small-prime set.

    Returns a list of prime factors (with multiplicity, smallest first).
    Raises ValueError if any residual prime factor is > 11 — callers that need
    to support hd=256 with a non-sqfree factoring should pad via sqfree_pad.
    """
    if n < 1:
        raise ValueError(f"dim must be >= 1, got {n}")
    d = n
    primes: List[int] = []
    for p in _SMALL_PRIMES:
        while d % p == 0:
            primes.append(p)
            d //= p
    if d != 1:
        raise ValueError(
            f"dim {n} has a prime factor > 11 (residue {d}); "
            f"use sqfree_pad(head_dim) to find a supported dimension"
        )
    return primes


def _hartley_kernel(p: int, device, dtype) -> torch.Tensor:
    """p x p real Hartley matrix H[i,j] = cas(2πij/p) / √p.

    H is real, symmetric, and orthonormal: H @ H = I. Computed in fp64 and
    cast to the caller's dtype to minimise float rounding in the √p factor.
    """
    two_pi_over_p = 2.0 * math.pi / p
    H = torch.empty((p, p), dtype=torch.float64)
    for i in range(p):
        for j in range(p):
            angle = two_pi_over_p * i * j
            H[i, j] = math.cos(angle) + math.sin(angle)
    H /= math.sqrt(p)
    return H.to(device=device, dtype=dtype)


def vht2(x: torch.Tensor) -> torch.Tensor:
    """Vilenkin-Hartley transform on the last dim.

    Self-inverse: vht2(vht2(x)) ≈ x within float tolerance. For any
    last-dim n that factors into {2,3,5,7,11}.

    The iterative staged formulation is O(n · Σp) rather than the O(n²)
    dense-matrix formulation. Returns a new tensor (not an in-place write).
    """
    n = x.shape[-1]
    primes = _factor_small_primes(n)

    out = x
    stride = 1
    for p in primes:
        outer = n // (p * stride)
        # (..., outer, p, stride) — length-p groups live on axis -2
        y = out.reshape(*out.shape[:-1], outer, p, stride)
        H = _hartley_kernel(p, device=y.device, dtype=y.dtype)
        # y_new[..., o, k, s] = Σ_j H[k, j] · y[..., o, j, s]
        y = torch.einsum("kj,...ojs->...oks", H, y)
        out = y.reshape(*out.shape[:-1], n)
        stride *= p

    return out


# ----- Sqfree pad helper (used when head_dim needs padding for VHT2) --------

_SMALL_PRIMES_SQFREE = (2, 3, 5, 7, 11)


def _is_small_prime_sqfree_factorable(n: int) -> bool:
    if n < 1:
        return False
    d = n
    for p in _SMALL_PRIMES_SQFREE:
        c = 0
        while d % p == 0:
            d //= p
            c += 1
            if c > 1:
                return False  # squared prime
    return d == 1


def sqfree_pad_dim(head_dim: int) -> int:
    """Next squarefree dimension >= head_dim that factors into {2,3,5,7,11}.

    hd=64  -> 66  = 2·3·11
    hd=128 -> 154 = 2·7·11
    hd=256 -> 330 = 2·3·5·11
    """
    for n in range(max(head_dim, 2), head_dim * 4):
        if _is_small_prime_sqfree_factorable(n):
            return n
    raise ValueError(f"no small-prime sqfree dim found >= {head_dim}")


# =============================================================================
# Möbius Mask
# =============================================================================

class MobiusMask:
    """
    Squarefree-first coefficient ordering for VHT2 coefficients.
    
    The Möbius function μ(n) is non-zero iff n is squarefree.
    Prioritizing squarefree indices during banded quantization
    improves quality by +0.14 PPL at identical coefficient budget.
    
    Cross-platform invariant: K correlation 0.997 on both hd=128 and hd=64.
    """

    def __init__(self, n: int):
        self.n = n
        self.mu = self._compute_mobius(n)
        
        # Build permutation: squarefree first, then non-squarefree
        squarefree = [i for i in range(n) if self.mu[i] != 0]
        non_squarefree = [i for i in range(n) if self.mu[i] == 0]
        self.order = squarefree + non_squarefree
        self.n_squarefree = len(squarefree)
        
        # Build inverse permutation
        self.inv_order = [0] * n
        for i, idx in enumerate(self.order):
            self.inv_order[idx] = i

        # Pre-compute as tensors (will be moved to device on first use)
        self._order_tensor = torch.tensor(self.order, dtype=torch.long)
        self._inv_order_tensor = torch.tensor(self.inv_order, dtype=torch.long)

    @staticmethod
    def _compute_mobius(n: int) -> List[int]:
        mu = [0] * n
        if n > 1:
            mu[1] = 1
        
        for i in range(2, n):
            # Factor i and check for squared primes
            val = i
            n_factors = 0
            has_square = False
            p = 2
            while p * p <= val:
                if val % p == 0:
                    n_factors += 1
                    val //= p
                    if val % p == 0:
                        has_square = True
                        break
                p += 1
            if not has_square and val > 1:
                n_factors += 1
            
            if has_square:
                mu[i] = 0
            else:
                mu[i] = 1 if n_factors % 2 == 0 else -1
        
        return mu

    def reorder(self, x: torch.Tensor) -> torch.Tensor:
        """Squarefree-first reordering. x: (..., n)."""
        idx = self._order_tensor.to(x.device)
        return x[..., idx]

    def unreorder(self, x: torch.Tensor) -> torch.Tensor:
        """Restore original index order. x: (..., n)."""
        idx = self._inv_order_tensor.to(x.device)
        return x[..., idx]


# =============================================================================
# Banded Quantization
# =============================================================================

class BandedQuantizer:
    """
    VHT2 banded quantization of spectral coefficients.
    
    Splits n coefficients into k equal bands. Each band gets:
      - 1 fp16 scale (max(abs(band)) / (2^(bits-1) - 1))
      - Packed signed integers at the band's bit depth
    
    Key results from paper:
      - 5/5/4/3 BEATS lossless fp16 by 0.04% (spectral regularization)
      - 4/4/4/4 is off the Pareto frontier
      - 3-bit floor: 2-bit on any band is catastrophic
      - Flat beats banded for V vectors (no exceptions)
    """

    def __init__(self, n: int, band_bits: List[int],
                 ternary_bands: Optional[List[int]] = None):
        self.n = n
        self.n_bands = len(band_bits)
        self.band_bits = band_bits
        self.band_size = n // self.n_bands
        # If n is not evenly divisible by n_bands, the last band absorbs the
        # remainder (matches the C core's sp_band_span helper). For 10 bands
        # at pad_dim=154 this means bands 0..8 have size 15, band 9 has size 19.
        self._last_band_size = n - self.band_size * (self.n_bands - 1)
        # Strange-attractor stack piece 4/4: bands listed here are quantized
        # to ternary {-1, 0, +1} (≈1.58 bits/coefficient information content)
        # regardless of band_bits[b]. The deadband threshold is fixed at
        # scale*0.5, which is the L1-optimal quantizer for symmetric
        # zero-mean signals. Empty list = identical to ship behavior.
        self.ternary_bands = set(ternary_bands or [])

    def _band_span(self, b: int) -> Tuple[int, int]:
        start = b * self.band_size
        size  = self._last_band_size if b == self.n_bands - 1 else self.band_size
        return start, size

    def quantize(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Quantize VHT2 coefficients into banded format.

        x: (..., n) float tensor
        Returns: (scales, quant_vals) per band
          scales[b]: (...,) fp16 scale
          quant_vals[b]: (..., band_size_b) int8/int16 quantized values
        """
        scales = []
        quant_vals = []

        for b in range(self.n_bands):
            start, size = self._band_span(b)
            end   = start + size
            band  = x[..., start:end]

            if b in self.ternary_bands:
                # Ternary {-1, 0, +1}: scale = amax (no max_val division).
                # Deadband at scale*0.5 — values in [-0.5,0.5]·amax round to 0.
                # This is the L1-optimal three-level quantizer for symmetric
                # zero-mean signals and matches the "1.58-bit" information
                # content the strange-attractor stack predicts is enough
                # for the noise tail.
                amax  = band.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12)
                scale = amax
                normalized = band / scale  # in [-1, 1]
                # Round to nearest of {-1, 0, +1}: > 0.5 → +1, < -0.5 → -1, else 0
                q = torch.zeros_like(normalized, dtype=torch.int8)
                q = torch.where(normalized >  0.5, torch.ones_like(q),  q)
                q = torch.where(normalized < -0.5, -torch.ones_like(q), q)
            else:
                bits  = self.band_bits[b]
                max_val = (1 << (bits - 1)) - 1
                amax = band.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12)
                scale = amax / max_val
                q = (band / scale).round().clamp(-max_val, max_val).to(torch.int8)

            scales.append(scale.squeeze(-1).half())
            quant_vals.append(q)

        return scales, quant_vals

    def dequantize(self, scales: List[torch.Tensor],
                   quant_vals: List[torch.Tensor]) -> torch.Tensor:
        """
        Dequantize banded format back to float VHT2 coefficients.
        
        Returns: (..., n) float tensor
        """
        bands = []
        for b in range(self.n_bands):
            scale = scales[b].float().unsqueeze(-1)
            # Mirror the C core guard: an fp16 scale that round-trips to
            # +/-Inf or NaN (amax overflowed fp16 on encode) must not
            # propagate into the inverse VHT2. Zero the scale; the band
            # then decodes as all zeros, matching CPU semantics.
            scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
            q     = quant_vals[b].float()
            bands.append(q * scale)

        return torch.cat(bands, dim=-1)

    def compressed_bytes_per_vec(self) -> int:
        """Bytes per compressed vector."""
        import math as _math
        total = 0
        for b, bits in enumerate(self.band_bits):
            _, size = self._band_span(b)
            total += 2  # fp16 scale
            if b in self.ternary_bands:
                # Ternary 5-per-byte packing target: ceil(size * log2(3) / 8).
                # Current quantize() returns int8 because pack/unpack lives in
                # the C/CUDA layers; this is the post-pack byte count for the
                # compression-ratio honesty check.
                total += int(_math.ceil(size * _math.log2(3) / 8))
            else:
                total += (size * bits + 7) // 8  # packed data
        return total


# =============================================================================
# Shadow Cache — the main integration point
# =============================================================================

class ShadowCache:
    """
    VHT2 compressed KV cache for transformer models.

    Write path: raw KV -> VHT2 forward -> Mobius reorder -> band quantize -> store
    Read path:  load   -> band dequantize -> Mobius unreorder -> VHT2 forward -> KV

    The transform on the read path is the *same* VHT2 as on write — it is
    self-inverse (each stage normalised by 1/√p), so no division by N is
    required to recover the original vector.

    Ship-safe configuration:
      K: 4 bands at 5/5/4/3 bits (with Möbius reorder)
      V: 1 band at 3 bits (flat, no reorder)

    Achieves 3.4–3.8× total compression at <1.25% PPL cost.
    Spectral regularization: 5/5/4/3 BEATS fp16 by 0.04%.
    """

    def __init__(
        self,
        head_dim: int = 128,
        n_layers: int = 32,
        n_heads_kv: int = 8,
        max_seq_len: int = 4096,
        k_band_bits: List[int] = [5, 5, 4, 3],
        v_band_bits: List[int] = [3],
        k_ternary_bands: Optional[List[int]] = None,
        v_ternary_bands: Optional[List[int]] = None,
        use_mobius: bool = True,
        # ---- Progressive-enhancement flags (default off = classic ship path) --
        use_sqfree_pad: bool = False,
        use_mobius_predict: bool = False,
        residual_bits: int = 0,
        use_spinor: bool = False,
        sk_frac: float = 0.75,
        device: str = 'cpu',
    ):
        """ShadowCache — the single VHT2 KV compression cache.

        Default flags reproduce the ship-path pipeline exactly
        (VHT2 → Möbius reorder → banded quantize → store). Enabling any of
        `use_sqfree_pad` / `use_mobius_predict` / `residual_bits > 0` /
        `use_spinor` routes internally to the sqfree+spinor aggressive
        pipeline defined in backends/torch/shannon_prime_sqfree.py —
        same VHT2 transform underneath, extra stages bolted on for the
        aggressive path's Knight-mask + Möbius CSR + residual + spinor
        machinery. Callers see one class; the flags pick the path.
        """
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.n_heads_kv = n_heads_kv
        self.max_seq_len = max_seq_len
        self.device = device

        # Progressive mode is active iff any sqfree-path flag is on.
        self._sqfree_mode = bool(
            use_sqfree_pad or use_mobius_predict or residual_bits > 0 or use_spinor
        )

        if self._sqfree_mode:
            # Lazy import: avoids circular when shannon_prime_sqfree itself
            # imports BandedQuantizer from this module.
            from shannon_prime_sqfree import SqfreeShadowCache  # type: ignore

            # Torus-aligned 5-band [5,4,4,4,5] is the sqfree default, but since
            # v1.03 any count in [1, SP_MAX_BANDS] is honoured (matches the C
            # core, CUDA kernel, and tools/sp_auto_bands.py's 10-band output).
            # Only fall back when the caller passed nothing.
            SP_MAX_BANDS = 16
            sqfree_bands = (list(k_band_bits)
                            if k_band_bits and 1 <= len(k_band_bits) <= SP_MAX_BANDS
                            else [5, 4, 4, 4, 5])
            self._impl = SqfreeShadowCache(
                head_dim=head_dim,
                n_layers=n_layers,
                n_heads_kv=n_heads_kv,
                max_seq_len=max_seq_len,
                band_bits=sqfree_bands,
                residual_bits=residual_bits if residual_bits > 0 else 3,
                use_spinor=bool(use_spinor),
                sk_frac=sk_frac,
                device=device,
            )
            # Expose the internal pad_dim for callers that want it
            self.pad_dim = getattr(self._impl, 'pad_dim', head_dim)
            # Quantizer references for compression_ratio() / memory_bytes()
            self.k_quantizer = self._impl.k_quantizer
            self.v_quantizer = self._impl.v_quantizer
            self.mobius = None  # sqfree path has its own mobius state
            return

        # ---- Classic VHT2 ship path (behavior identical to v1.0) ---------
        # Strange-attractor stack piece 4/4: optional ternary noise-tail.
        # When k_ternary_bands=[3] and k_band_bits=[5,5,4,3], band 3 is
        # quantized to {-1, 0, +1} instead of 3-bit signed. The 5/5/4/1.58
        # configuration tests whether the noise-tail can be reduced to
        # ternary without quality loss (per the multiplicative-lattice
        # scaling-law equation, head_dim=128 + bf16-class models should
        # tolerate this).
        self._impl = None
        self.k_quantizer = BandedQuantizer(head_dim, k_band_bits,
                                           ternary_bands=k_ternary_bands)
        self.v_quantizer = BandedQuantizer(head_dim, v_band_bits,
                                           ternary_bands=v_ternary_bands)
        self.mobius = MobiusMask(head_dim) if use_mobius else None

        # Storage: lists of (scales, quant_vals) per position
        # Indexed as cache[layer][head][pos]
        n_slots = n_layers * n_heads_kv
        self.k_scales = [[None] * max_seq_len for _ in range(n_slots)]
        self.k_quants = [[None] * max_seq_len for _ in range(n_slots)]
        self.v_scales = [[None] * max_seq_len for _ in range(n_slots)]
        self.v_quants = [[None] * max_seq_len for _ in range(n_slots)]

    def _slot(self, layer: int, head: int) -> int:
        return layer * self.n_heads_kv + head

    def write_k(self, layer: int, head: int, pos: int,
                k_vec: torch.Tensor):
        """Compress and store a K vector (head_dim,) already RoPE'd."""
        if self._sqfree_mode:
            return self._impl.write_k(layer, head, pos, k_vec)

        # VHT2 forward (self-inverse, 1/√p per stage)
        x = vht2(k_vec.unsqueeze(0)).squeeze(0)

        # Möbius reorder (squarefree first)
        if self.mobius is not None:
            x = self.mobius.reorder(x)

        # Band quantize
        scales, quants = self.k_quantizer.quantize(x.unsqueeze(0))
        scales = [s.squeeze(0) for s in scales]
        quants = [q.squeeze(0) for q in quants]

        slot = self._slot(layer, head)
        self.k_scales[slot][pos] = scales
        self.k_quants[slot][pos] = quants

    def write_v(self, layer: int, head: int, pos: int,
                v_vec: torch.Tensor):
        """Compress and store a V vector."""
        if self._sqfree_mode:
            return self._impl.write_v(layer, head, pos, v_vec)

        # VHT2 forward
        x = vht2(v_vec.unsqueeze(0)).squeeze(0)

        # No Möbius for V in self-attention (uniform spectrum — no benefit)
        scales, quants = self.v_quantizer.quantize(x.unsqueeze(0))
        scales = [s.squeeze(0) for s in scales]
        quants = [q.squeeze(0) for q in quants]

        slot = self._slot(layer, head)
        self.v_scales[slot][pos] = scales
        self.v_quants[slot][pos] = quants

    def read_k(self, layer: int, head: int, pos: int) -> torch.Tensor:
        """Reconstruct a K vector from compressed storage."""
        if self._sqfree_mode:
            return self._impl.read_k(layer, head, pos)

        slot = self._slot(layer, head)
        scales = [s.unsqueeze(0) for s in self.k_scales[slot][pos]]
        quants = [q.unsqueeze(0) for q in self.k_quants[slot][pos]]

        x = self.k_quantizer.dequantize(scales, quants).squeeze(0)

        if self.mobius is not None:
            x = self.mobius.unreorder(x)

        # Inverse == forward for the self-inverse VHT2 (no div by N).
        x = vht2(x.unsqueeze(0)).squeeze(0)

        return x

    def read_v(self, layer: int, head: int, pos: int) -> torch.Tensor:
        """Reconstruct a V vector from compressed storage."""
        if self._sqfree_mode:
            return self._impl.read_v(layer, head, pos)

        slot = self._slot(layer, head)
        scales = [s.unsqueeze(0) for s in self.v_scales[slot][pos]]
        quants = [q.unsqueeze(0) for q in self.v_quants[slot][pos]]

        x = self.v_quantizer.dequantize(scales, quants).squeeze(0)

        # Inverse == forward VHT2 (no div by N — self-inverse).
        x = vht2(x.unsqueeze(0)).squeeze(0)

        return x

    def compression_ratio(self) -> float:
        """Total K+V compression ratio vs fp16 baseline."""
        baseline = self.head_dim * 2  # fp16 bytes per element
        k_bytes = self.k_quantizer.compressed_bytes_per_vec()
        v_bytes = self.v_quantizer.compressed_bytes_per_vec()
        return (2 * baseline) / (k_bytes + v_bytes)

    def memory_bytes(self, seq_len: int) -> dict:
        """Estimate memory usage for given sequence length."""
        n_slots = self.n_layers * self.n_heads_kv
        k_bytes = n_slots * seq_len * self.k_quantizer.compressed_bytes_per_vec()
        v_bytes = n_slots * seq_len * self.v_quantizer.compressed_bytes_per_vec()
        baseline = n_slots * seq_len * self.head_dim * 2 * 2  # K+V fp16
        return {
            'k_bytes': k_bytes,
            'v_bytes': v_bytes,
            'total_bytes': k_bytes + v_bytes,
            'baseline_bytes': baseline,
            'ratio': baseline / (k_bytes + v_bytes),
        }


# =============================================================================
# Convenience: compute correlation
# =============================================================================

def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two tensors."""
    a = a.float().flatten()
    b = b.float().flatten()
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    num = (a_centered * b_centered).sum()
    den = (a_centered.norm() * b_centered.norm()).clamp(min=1e-12)
    return (num / den).item()
