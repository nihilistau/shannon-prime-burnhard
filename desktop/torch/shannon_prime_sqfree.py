# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
Squarefree prime-Hartley basis + Möbius CSR predictor + spinor sheet bit.

This module extends the VHT2-based ship config (shannon_prime_torch.py) with
the aggressive compression path validated on Qwen3-8B Q8 hd=128:

    K+μ+3bit+spinor 3/3/3/3/3:  PPL 7.32 @ 3.3× (matches MOBIUS default 7.31 @ 2.6×)

The key insight: the Möbius predictor gives r≈0 on the p=2 VHT2 basis (VHT2²=I
scrambles divisibility structure). On a squarefree prime-Hartley basis it
gives r=0.40-0.58, and the spinor sheet bit captures the systematic sign
errors at the causal-mask boundary for an additional compression win.

Usage:
    from shannon_prime_sqfree import SqfreeShadowCache

    cache = SqfreeShadowCache(head_dim=128, n_layers=32, n_heads_kv=8)
    cache.write_k(layer=0, head=0, pos=0, k_vec=tensor)
    k_recon = cache.read_k(layer=0, head=0, pos=0)

This path is opt-in. The VHT2 ship config in shannon_prime_torch.py remains
the default for production deployment.
"""

import torch
import math
import functools
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# =============================================================================
# Möbius function and squarefree detection
# =============================================================================

@functools.lru_cache(maxsize=2048)
def mobius(n: int) -> int:
    """Classical Möbius function μ(n). Cached for repeated access."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    val = n
    n_factors = 0
    p = 2
    while p * p <= val:
        if val % p == 0:
            n_factors += 1
            val //= p
            if val % p == 0:
                return 0  # p² divides n → not squarefree
        p += 1
    if val > 1:
        n_factors += 1
    return 1 if n_factors % 2 == 0 else -1


def is_squarefree(n: int) -> bool:
    """True if n has no repeated prime factors (μ(n) ≠ 0)."""
    return n >= 1 and mobius(n) != 0


# =============================================================================
# Squarefree basis padding
# =============================================================================

_PRIMES = [2, 3, 5, 7, 11]


def _factors_into_primes(n: int, primes: List[int] = _PRIMES) -> bool:
    """True if n factors completely into the given primes."""
    if n <= 0:
        return False
    d = n
    for p in primes:
        while d % p == 0:
            d //= p
    return d == 1


def _is_sqfree_factorable(n: int, primes: List[int] = _PRIMES) -> bool:
    """True if n is squarefree AND factors into the given primes."""
    if n <= 0 or not _factors_into_primes(n, primes):
        return False
    for p in primes:
        c = 0
        d = n
        while d % p == 0:
            d //= p
            c += 1
            if c > 1:
                return False
    return True


def sqfree_pad_dim(head_dim: int, primes: List[int] = _PRIMES) -> int:
    """
    Next squarefree dimension ≥ head_dim that factors into {2,3,5,7,11}.

    For hd=64:  66 = 2·3·11
    For hd=128: 154 = 2·7·11
    For hd=256: 330 = 2·3·5·11

    Mirrors prime_reconstruct.h:4273-4285.
    """
    for n in range(max(head_dim, 2), head_dim * 4):
        if _is_sqfree_factorable(n, primes):
            return n
    return head_dim  # Fallback (should not happen for practical head_dim)


def sqfree_pad(x: torch.Tensor, pad_dim: int) -> torch.Tensor:
    """Pad last axis from head_dim to pad_dim with mean-fill."""
    hd = x.shape[-1]
    if hd >= pad_dim:
        return x
    mean_val = x.mean(dim=-1, keepdim=True)
    tail = mean_val.expand(*x.shape[:-1], pad_dim - hd)
    return torch.cat([x, tail], dim=-1)


def sqfree_unpad(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Truncate last axis back to head_dim."""
    return x[..., :head_dim]


# =============================================================================
# Prime-Hartley Transform
# =============================================================================

@functools.lru_cache(maxsize=32)
def _hartley_matrix(p: int, dtype_str: str = 'float32') -> torch.Tensor:
    """
    Build the p×p Hartley matrix: H[k,n] = cas(2πkn/p) / √p.

    cas(x) = cos(x) + sin(x). This is the real-valued, self-inverse
    kernel for Z/pZ. When normalized by 1/√p, H·H = I.

    Cached per (p, dtype) to avoid recomputation.
    """
    dtype = getattr(torch, dtype_str)
    idx = torch.arange(p, dtype=torch.float64)
    angles = 2.0 * math.pi * idx.unsqueeze(0) * idx.unsqueeze(1) / p
    H = (torch.cos(angles) + torch.sin(angles)) / math.sqrt(p)
    return H.to(dtype)


def _prime_factorize(n: int, primes: List[int] = _PRIMES) -> List[int]:
    """Return list of prime factors of n (with multiplicity) from the given primes."""
    factors = []
    d = n
    for p in primes:
        while d % p == 0:
            factors.append(p)
            d //= p
    if d > 1:
        raise ValueError(f"{n} does not factor into {primes} (remainder {d})")
    return factors


def vilenkin_forward(x: torch.Tensor, primes: Optional[List[int]] = None) -> torch.Tensor:
    """
    Prime-Hartley forward transform on last axis.

    Applies successive Hartley transforms for each prime factor of x.shape[-1].
    The dimension must factor completely into the given primes.

    Self-inverse: vilenkin_forward(vilenkin_forward(x)) = x.

    Args:
        x: (..., N) tensor where N = ∏ p_i factors into primes.
        primes: which primes to use (default: [2,3,5,7,11,13]).

    Returns:
        (..., N) tensor of Vilenkin-Hartley coefficients.
    """
    if primes is None:
        primes = _PRIMES
    N = x.shape[-1]
    factors = _prime_factorize(N, primes)
    dtype_str = str(x.dtype).split('.')[-1]

    out = x.clone()
    stride = 1
    for p in factors:
        H = _hartley_matrix(p, dtype_str).to(x.device)
        # Reshape: (..., N//(stride*p), p, stride) → apply H on the p-axis
        shape = list(out.shape[:-1]) + [N // (stride * p), p, stride]
        buf = out.view(*shape)
        # Contract H over the p-axis: result[..., k, :] = Σ_n H[k,n] · buf[..., n, :]
        buf_perm = buf.permute(*range(len(shape) - 3), -3, -1, -2)  # (..., blocks, stride, p)
        transformed = buf_perm @ H.T  # (..., blocks, stride, p)
        out = transformed.permute(*range(len(shape) - 3), -3, -1, -2).reshape(*x.shape)
        stride *= p

    return out


def vilenkin_inverse(x: torch.Tensor, primes: Optional[List[int]] = None) -> torch.Tensor:
    """Inverse = forward (self-inverse when each H_p is normalized by 1/√p)."""
    return vilenkin_forward(x, primes)


# =============================================================================
# Knight-Ranked Mask + Möbius CSR Predictor
# =============================================================================

@dataclass
class KnightMask:
    """
    Knight-ranked squarefree-first skeleton with Möbius CSR predictor.

    The skeleton stores the top-K squarefree indices by variance. If K
    exceeds the squarefree count, fill with top-variance composites.
    Non-skeleton non-squarefree indices get the Möbius predictor +
    quantized residual path.

    Mirrors prime_reconstruct.h:4973-5041.
    """
    dim: int                           # Padded dimension (pad_dim)
    sk_k: int                          # Skeleton size
    skeleton_idx: torch.Tensor         # (sk_k,) int64, sorted ascending
    residual_idx: torch.Tensor         # (n_res,) int64, sorted ascending
    # Möbius CSR: pred[i] = Σ mu_sign[j] · skel_vals[skel_slot[j]]
    #             for j in [offsets[i], offsets[i+1])
    csr_offsets: torch.Tensor          # (n_res + 1,) int64
    csr_skel_slot: torch.Tensor        # (n_terms,) int64
    csr_mu_sign: torch.Tensor          # (n_terms,) int8


def build_knight_mask(pad_dim: int, sk_k: int,
                      variance: Optional[torch.Tensor] = None) -> KnightMask:
    """
    Build a KnightMask for the given padded dimension.

    Args:
        pad_dim:  Squarefree-padded head dimension (e.g. 154 for hd=128).
        sk_k:     Skeleton size (how many indices to store directly).
        variance: Optional (pad_dim,) tensor of per-index variance from a
                  calibration batch. If None, uses index order (squarefree first).
    """
    # Partition indices
    sqfree = [i for i in range(pad_dim) if is_squarefree(i + 1)]  # 1-indexed for μ
    composite = [i for i in range(pad_dim) if not is_squarefree(i + 1)]

    # Rank by variance if available, else by index
    if variance is not None:
        var = variance.detach().cpu().float()
        sqfree.sort(key=lambda i: float(var[i]), reverse=True)
        composite.sort(key=lambda i: float(var[i]), reverse=True)

    # Pick skeleton: squarefree first, fill with composites if needed
    k_sqf = min(sk_k, len(sqfree))
    picked = list(sqfree[:k_sqf])
    extra = sk_k - k_sqf
    if extra > 0:
        picked.extend(composite[:min(extra, len(composite))])
    picked.sort()
    skel_set = set(picked)
    idx_to_slot = {idx: slot for slot, idx in enumerate(picked)}

    # Residual = non-squarefree indices NOT in skeleton
    residual = [i for i in range(pad_dim)
                if (not is_squarefree(i + 1)) and (i not in skel_set)]
    residual.sort()

    # Build Möbius CSR
    offsets = [0]
    skel_slots = []
    mu_signs = []
    for n_idx in residual:
        n = n_idx + 1  # 1-indexed for μ
        d = 1
        while d <= n:
            if n % d == 0:
                mu_d = mobius(d)
                if mu_d != 0:
                    q = n // d
                    q_idx = q - 1  # back to 0-indexed
                    if 0 <= q_idx < pad_dim and q_idx in idx_to_slot:
                        skel_slots.append(idx_to_slot[q_idx])
                        mu_signs.append(mu_d)
            d += 1
        offsets.append(len(skel_slots))

    return KnightMask(
        dim=pad_dim,
        sk_k=len(picked),
        skeleton_idx=torch.tensor(picked, dtype=torch.long),
        residual_idx=torch.tensor(residual, dtype=torch.long),
        csr_offsets=torch.tensor(offsets, dtype=torch.long),
        csr_skel_slot=torch.tensor(skel_slots, dtype=torch.long) if skel_slots
                      else torch.zeros(0, dtype=torch.long),
        csr_mu_sign=torch.tensor(mu_signs, dtype=torch.int8) if mu_signs
                    else torch.zeros(0, dtype=torch.int8),
    )


# =============================================================================
# N-bit Symmetric Residual Quantization
# =============================================================================

def quantize_residual(vals: torch.Tensor, nbits: int,
                      mag: torch.Tensor) -> torch.Tensor:
    """
    Symmetric quantization to N bits.

    Maps to integer levels [0, 2^nbits - 1]. The saturation range widens with
    bit depth so higher nbits does not clip more of a Gaussian-distributed
    residual than it resolves: range = nbits * mag. At 1-bit this collapses
    to the Lloyd-Max optimum (levels at ±mag); at 3/4-bit it covers ~2.4σ/3.2σ.

    Args:
        vals: (..., R) residual values.
        mag:  (...,) or scalar — per-vector magnitude (mean |val| works well).
        nbits: 1, 2, 3, or 4.

    Returns:
        (..., R) int tensor of quantized levels in [0, 2^nbits - 1].
    """
    L = 1 << nbits
    center = (L - 1) / 2.0
    mag = mag.unsqueeze(-1).clamp(min=1e-12) if mag.dim() < vals.dim() else mag.clamp(min=1e-12)
    sat_range = mag * float(nbits)
    step = (2.0 * sat_range) / (L - 1)
    levels = ((vals / step) + center).round().clamp(0, L - 1).to(torch.int32)
    return levels


def dequantize_residual(levels: torch.Tensor, nbits: int,
                        mag: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_residual."""
    L = 1 << nbits
    center = (L - 1) / 2.0
    mag = mag.unsqueeze(-1) if mag.dim() < levels.dim() else mag
    sat_range = mag * float(nbits)
    step = (2.0 * sat_range) / (L - 1)
    return (levels.float() - center) * step


# =============================================================================
# Compressed representation
# =============================================================================

@dataclass
class SqfreeCompressed:
    """Compressed representation of a single K or V vector."""
    # Skeleton: banded-quantized VHT2 coefficients at skeleton indices
    skel_scales: List[torch.Tensor]     # Per-band fp16 scales
    skel_quants: List[torch.Tensor]     # Per-band int8 quantized values

    # Möbius residual (at residual indices)
    residual_levels: Optional[torch.Tensor] = None   # (n_res,) int32 levels
    residual_mag: Optional[torch.Tensor] = None      # scalar fp32

    # Spinor sheet bit
    sheet_bits: Optional[torch.Tensor] = None        # (n_res,) bool

    # Per-vector max-abs reference (used for residual scale recovery).
    orig_max_abs: float = 0.0


# =============================================================================
# Sqfree Shadow Cache — the aggressive path
# =============================================================================

class SqfreeShadowCache:
    """
    VHT2 compressed KV cache using squarefree prime-Hartley basis.

    Write path:
        raw KV → sqfree pad → Vilenkin forward → Knight skeleton extract
        → band quantize skeleton → Möbius predict residual → quantize
        residual → spinor sheet bit → store

    Read path:
        load → dequant skeleton → Möbius predict → dequant residual
        → spinor correct → scatter → Vilenkin inverse → unpad → KV

    Validated result (Qwen3-8B Q8 hd=128):
        K+μ+3bit+spinor 3/3/3/3/3:  PPL 7.32 @ 3.3×
        (matches MOBIUS default 7.31 @ 2.6×, +27% compression)
    """

    def __init__(
        self,
        head_dim: int = 128,
        n_layers: int = 32,
        n_heads_kv: int = 8,
        max_seq_len: int = 4096,
        band_bits: List[int] = None,
        residual_bits: int = 3,
        use_spinor: bool = True,
        sk_frac: float = 0.75,
        device: str = 'cpu',
    ):
        if band_bits is None:
            band_bits = [5, 4, 4, 4, 5]  # 5-band torus-aligned default

        self.head_dim = head_dim
        self.n_layers = n_layers
        self.n_heads_kv = n_heads_kv
        self.max_seq_len = max_seq_len
        self.residual_bits = residual_bits
        self.use_spinor = use_spinor
        self.device = device

        # Sqfree basis
        self.pad_dim = sqfree_pad_dim(head_dim)
        self.primes = _prime_factorize(self.pad_dim)

        # Skeleton size — snap to multiple of n_bands so BandedQuantizer can
        # divide the skeleton evenly across its bands.
        n_bands = len(band_bits)
        sk_k = int(self.pad_dim * sk_frac)
        sk_k = max(n_bands, (sk_k // n_bands) * n_bands)
        self.mask = build_knight_mask(self.pad_dim, sk_k)

        # Banded quantizer operates on skeleton indices only
        try:
            from shannon_prime_torch import BandedQuantizer
        except ImportError:
            from backends.torch.shannon_prime_torch import BandedQuantizer
        self.k_quantizer = BandedQuantizer(self.mask.sk_k, band_bits)
        self.v_quantizer = BandedQuantizer(self.mask.sk_k, band_bits)

        # Storage
        n_slots = n_layers * n_heads_kv
        self.k_store = [[None] * max_seq_len for _ in range(n_slots)]
        self.v_store = [[None] * max_seq_len for _ in range(n_slots)]

    def _slot(self, layer: int, head: int) -> int:
        return layer * self.n_heads_kv + head

    def _compress_vec(self, vec: torch.Tensor, quantizer) -> SqfreeCompressed:
        """Compress a single vector through the full pipeline."""
        orig_max = vec.abs().max().item()

        # 1. Pad to sqfree dimension
        padded = sqfree_pad(vec.unsqueeze(0), self.pad_dim)

        # 2. Vilenkin forward transform
        coeffs = vilenkin_forward(padded, self.primes).squeeze(0)

        # 3. Extract skeleton coefficients
        skel_idx = self.mask.skeleton_idx.to(vec.device)
        skel_vals = coeffs[skel_idx]

        # 4. Band-quantize skeleton
        skel_scales, skel_quants = quantizer.quantize(skel_vals.unsqueeze(0))
        skel_scales = [s.squeeze(0) for s in skel_scales]
        skel_quants = [q.squeeze(0) for q in skel_quants]

        # 5. Möbius predict residual positions
        res_idx = self.mask.residual_idx.to(vec.device)
        n_res = res_idx.numel()

        residual_levels = None
        residual_mag = None
        sheet_bits = None

        if n_res > 0:
            actual_res = coeffs[res_idx]

            # Compute predictions via CSR
            pred = torch.zeros(n_res, device=vec.device, dtype=vec.dtype)
            offsets = self.mask.csr_offsets
            slots = self.mask.csr_skel_slot.to(vec.device)
            signs = self.mask.csr_mu_sign.to(vec.device).float()
            for i in range(n_res):
                start, end = offsets[i].item(), offsets[i + 1].item()
                if start < end:
                    pred[i] = (signs[start:end] * skel_vals[slots[start:end]]).sum()

            # 6. Spinor: pick better sign for predictor
            if self.use_spinor:
                v_plus = actual_res - pred
                v_minus = actual_res + pred
                use_minus = v_minus.abs() < v_plus.abs()
                sheet_bits = use_minus
                deviation = torch.where(use_minus, v_minus, v_plus)
            else:
                deviation = actual_res - pred

            # 7. Quantize residual
            mag = deviation.abs().mean()
            levels = quantize_residual(deviation, self.residual_bits, mag)
            residual_levels = levels
            residual_mag = mag

        return SqfreeCompressed(
            skel_scales=skel_scales,
            skel_quants=skel_quants,
            residual_levels=residual_levels,
            residual_mag=residual_mag,
            sheet_bits=sheet_bits,
            orig_max_abs=orig_max,
        )

    def _reconstruct_vec(self, comp: SqfreeCompressed, quantizer) -> torch.Tensor:
        """Reconstruct a vector from compressed representation."""
        device = comp.skel_scales[0].device if comp.skel_scales else self.device

        # 1. Dequantize skeleton
        scales = [s.unsqueeze(0) for s in comp.skel_scales]
        quants = [q.unsqueeze(0) for q in comp.skel_quants]
        skel_vals = quantizer.dequantize(scales, quants).squeeze(0)

        # 2. Build full coefficient vector
        coeffs = torch.zeros(self.pad_dim, device=device, dtype=torch.float32)

        skel_idx = self.mask.skeleton_idx.to(device)
        coeffs[skel_idx] = skel_vals

        # 3. Reconstruct residual positions
        res_idx = self.mask.residual_idx.to(device)
        n_res = res_idx.numel()

        if n_res > 0 and comp.residual_levels is not None:
            # Compute predictions
            pred = torch.zeros(n_res, device=device, dtype=torch.float32)
            offsets = self.mask.csr_offsets
            slots = self.mask.csr_skel_slot.to(device)
            signs = self.mask.csr_mu_sign.to(device).float()
            for i in range(n_res):
                start, end = offsets[i].item(), offsets[i + 1].item()
                if start < end:
                    pred[i] = (signs[start:end] * skel_vals[slots[start:end]]).sum()

            # Spinor: flip pred sign where sheet bit is set
            if self.use_spinor and comp.sheet_bits is not None:
                pred = torch.where(comp.sheet_bits.to(device), -pred, pred)

            # Dequantize residual
            deviation = dequantize_residual(
                comp.residual_levels.to(device), self.residual_bits, comp.residual_mag.to(device)
            )
            coeffs[res_idx] = pred + deviation

        # 4. Vilenkin inverse
        full = vilenkin_inverse(coeffs.unsqueeze(0), self.primes).squeeze(0)

        # 5. Unpad
        out = sqfree_unpad(full, self.head_dim)

        return out

    def write_k(self, layer: int, head: int, pos: int, k_vec: torch.Tensor):
        """Compress and store a K vector (head_dim floats, already RoPE'd)."""
        comp = self._compress_vec(k_vec, self.k_quantizer)
        self.k_store[self._slot(layer, head)][pos] = comp

    def write_v(self, layer: int, head: int, pos: int, v_vec: torch.Tensor):
        """Compress and store a V vector."""
        comp = self._compress_vec(v_vec, self.v_quantizer)
        self.v_store[self._slot(layer, head)][pos] = comp

    def read_k(self, layer: int, head: int, pos: int) -> torch.Tensor:
        """Reconstruct a K vector from compressed storage."""
        comp = self.k_store[self._slot(layer, head)][pos]
        return self._reconstruct_vec(comp, self.k_quantizer)

    def read_v(self, layer: int, head: int, pos: int) -> torch.Tensor:
        """Reconstruct a V vector from compressed storage."""
        comp = self.v_store[self._slot(layer, head)][pos]
        return self._reconstruct_vec(comp, self.v_quantizer)

    def compression_ratio(self) -> float:
        """Estimated K+V compression ratio vs fp16."""
        baseline = self.head_dim * 2  # fp16 bytes
        k_bytes = self.k_quantizer.compressed_bytes_per_vec()
        v_bytes = self.v_quantizer.compressed_bytes_per_vec()
        # Add residual + spinor overhead
        n_res = self.mask.residual_idx.numel()
        res_bytes = (n_res * self.residual_bits + 7) // 8  # packed bits
        res_bytes += 4  # fp32 magnitude
        if self.use_spinor:
            res_bytes += (n_res + 7) // 8  # sheet bits
        k_total = k_bytes + res_bytes
        v_total = v_bytes + res_bytes
        return (2 * baseline) / (k_total + v_total)
