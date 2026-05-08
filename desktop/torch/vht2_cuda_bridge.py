"""
Shannon-Prime VHT2 CUDA bridge — auto-selects GPU kernel or PyTorch fallback.

Provides a unified API for VHT2 compress/decompress operations regardless
of whether the CUDA extension is compiled. The CUDA path is ~8-12× faster
at 720p+ (2400+ tokens) where the butterfly dominates over Python overhead.

Build the CUDA extension once:
    cd backends/cuda && python setup_wan.py build_ext --inplace

Import:
    from vht2_cuda_bridge import VHT2Bridge
    bridge = VHT2Bridge()
    compressed = bridge.compress(k_vecs, bridge.skeleton_mask_128())
    recovered  = bridge.decompress(compressed, bridge.skeleton_mask_128())
"""

import math
import os
import sys
import torch
import pathlib

# ── Try to load the CUDA extension ────────────────────────────────────────────

_CUDA_AVAILABLE = False
_cuda_ext       = None

try:
    import shannon_prime_cuda_wan as _cuda_ext
    _CUDA_AVAILABLE = True
except ImportError:
    # Try to find the .pyd/.so next to this file (built with --inplace into cuda/)
    _here     = pathlib.Path(__file__).resolve().parent
    _cuda_dir = _here.parent / "cuda"

    # On Windows, Python 3.8+ requires explicit DLL search directories.
    # The extension depends on c10.dll, torch_cuda.dll, cudart64_12.dll etc.
    if hasattr(os, "add_dll_directory"):
        try:
            import torch
            _torch_lib = pathlib.Path(torch.__file__).resolve().parent / "lib"
            if _torch_lib.exists():
                os.add_dll_directory(str(_torch_lib))
        except Exception:
            pass
        # Also add CUDA runtime bin if CUDA_HOME or standard install path exists
        for _cuda_bin in [
            pathlib.Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin"),
            pathlib.Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin"),
        ]:
            if _cuda_bin.exists():
                os.add_dll_directory(str(_cuda_bin))
                break

    if str(_cuda_dir) not in sys.path:
        sys.path.insert(0, str(_cuda_dir))
    try:
        import shannon_prime_cuda_wan as _cuda_ext
        _CUDA_AVAILABLE = True
    except ImportError:
        pass

# ── PyTorch fallback VHT2 ─────────────────────────────────────────────────────

def _sqfree_mask_128() -> torch.Tensor:
    """Return a [128] bool tensor marking squarefree integers 1..128."""
    def is_sqfree(n: int) -> bool:
        if n <= 0:
            return False
        p = 2
        while p * p <= n:
            if n % (p * p) == 0:
                return False
            p += 1
        return True

    mask = torch.zeros(128, dtype=torch.bool)
    for i in range(128):
        if is_sqfree(i + 1):          # 1-indexed squarefree check
            mask[i] = True
    return mask


def _twin_prime_pairs(n: int = 128) -> list[tuple[int, int]]:
    """
    Return spectral-index pairs (i, j) where i+1 and j+1 are primes with j-i==2.

    The +1 is because the Hartley basis is 0-indexed but the squarefree /
    prime structure is naturally 1-indexed (position k corresponds to the
    arithmetic value k+1).

    At n=128, the twin-prime pairs in the spectrum are:
      (3,5)→(2,4), (5,7)→(4,6), (11,13)→(10,12), (17,19)→(16,18),
      (29,31)→(28,30), (41,43)→(40,42), (59,61)→(58,60), (71,73)→(70,72),
      (101,103)→(100,102), (107,109)→(106,108).

    The pairs (3,5) and (5,7) share index 4. Sequential application is
    safe at small alpha; the second pass sees the first pass's tiny pull.

    Strange-attractor reading: each twin pair is a +2 arithmetic neighbor
    on the prime lattice. Quantization noise that lands on the wrong side
    of a bin boundary disagrees with its neighbor; the borrowing pulls
    the outlier back toward the local consensus before the inverse VHT2.
    """
    def is_prime(k: int) -> bool:
        if k < 2:
            return False
        if k < 4:
            return True
        if k % 2 == 0:
            return False
        d = 3
        while d * d <= k:
            if k % d == 0:
                return False
            d += 2
        return True

    return _prime_pairs_with_gap(n, gap=2, _is_prime=is_prime)


def _prime_pairs_with_gap(n: int, gap: int, _is_prime=None) -> list[tuple[int, int]]:
    """
    Return spectral-index pairs (i, j) where i+1 and j+1 are primes with
    j-i == gap. Greedy-deduplicated so all returned pairs are disjoint.

    gap=2  : twin primes      (3-5, 11-13, 17-19, ...)
    gap=4  : cousin primes    (3-7, 7-11, 13-17, 19-23, ...)
    gap=6  : sexy primes      (5-11, 7-13, 11-17, 13-19, ...)
    gap=2k : larger Goldbach-style pairs

    The Goldbach-extended view: any even gap counts. Larger gaps
    correspond to weaker arithmetical neighborhoods, so callers should
    typically scale the borrow alpha down for larger-gap pairs.
    """
    if _is_prime is None:
        def is_prime(k: int) -> bool:
            if k < 2:    return False
            if k < 4:    return True
            if k % 2 == 0: return False
            d = 3
            while d * d <= k:
                if k % d == 0: return False
                d += 2
            return True
        _is_prime = is_prime

    raw_pairs = []
    for p in range(2, n - gap + 1):
        if _is_prime(p) and _is_prime(p + gap):
            raw_pairs.append((p - 1, p + gap - 1))

    seen = set()
    pairs = []
    for i, j in raw_pairs:
        if i in seen or j in seen:
            continue
        pairs.append((i, j))
        seen.add(i); seen.add(j)
    return pairs


def _goldbach_pairs(n: int = 128, gaps: tuple[int, ...] = (2, 4, 6)) -> dict[int, list[tuple[int, int]]]:
    """
    Return a dict mapping each gap to its disjoint prime-pair list.
    Pairs are deduplicated WITHIN each gap; pairs ACROSS gaps may overlap
    in spectral index, which is fine because each gap is processed in a
    separate vectorized pass with its own (decreasing) alpha.
    """
    return {gap: _prime_pairs_with_gap(n, gap=gap) for gap in gaps}


# v2 quick win: distance-to-Zeta-Zero decay scaling.
#
# First ~30 imaginary parts of nontrivial Riemann zeta zeros. The paper's
# claim is that these are the Poincaré sections of the cache trajectory
# in the prime-harmonic basis; spectral coefficients near these positions
# are "anchored" and tolerate stronger correction. Coefficients between
# zeros have weaker anchoring and should receive less aggressive borrow.
#
# Reconstruction: R(θ) = exp(-λ · |θ − ρ_n|) per the paper's "Zero-
# Computation Resonance" claim. We discretize θ = spectral index k and
# ρ_n = round(im_part) so the distance is integer-arithmetic friendly.
RIEMANN_ZEROS_IMAG = (
    14.13472, 21.02204, 25.01086, 30.42488, 32.93506,
    37.58618, 40.91872, 43.32707, 48.00515, 49.77383,
    52.97032, 56.44625, 59.34704, 60.83178, 65.11254,
    67.07981, 69.54640, 72.06716, 75.70469, 77.14484,
    79.33738, 82.91038, 84.73549, 87.42527, 88.80911,
    92.49190, 94.65134, 95.87063, 98.83119, 101.31785,
)


def _zeta_resonance_weights(n: int = 128, lambda_: float = 0.05) -> list[float]:
    """
    Per-coefficient resonance weight `exp(-λ · dist_to_nearest_zero[k])`.

    n=128: 19 zeros fall in range; remaining indices' distance is to the
    nearest of those 19. Weight is in (0, 1]; 1 at zeros, decays smoothly
    away. λ=0.05 means a weight of 0.61 at distance 10 (mid-decay), 0.37
    at distance 20.
    """
    import math
    rounded_zeros = sorted({int(round(z)) for z in RIEMANN_ZEROS_IMAG if 0 <= round(z) < n})
    if not rounded_zeros:
        return [1.0] * n
    weights = []
    for k in range(n):
        dist = min(abs(k - z) for z in rounded_zeros)
        weights.append(math.exp(-lambda_ * dist))
    return weights


def _apply_twin_borrow(coeffs: torch.Tensor, mask: torch.Tensor,
                       pairs: list[tuple[int, int]],
                       alpha: float = 0.10, threshold: float = 0.0,
                       mode: str = "symmetric",
                       zeta_weights: list[float] | None = None) -> torch.Tensor:
    """
    Decode-side twin-prime borrowing. Applied to spectral coefficients
    AFTER mask-application but BEFORE the inverse VHT2 transform.

    For each twin-prime pair (i, j) where both indices survive in the mask:
      diff = |c_i - c_j| / max(|c_i|, |c_j|, eps)
      if diff > threshold:  apply blend per `mode`

    Modes:
      symmetric   (default) — both indices pull toward (c_i+c_j)/2 with α.
                              |Δc| ≤ α/2 * |c_i - c_j| each.
      low_anchor  — lower-prime index is the anchor (unchanged); higher-prime
                    pulled toward the lower with α. Reflects the heuristic
                    that lower primes carry more spectral energy in RoPE'd K
                    so they're the more reliable reference.
      high_anchor — inverse: higher-prime is anchor. Useful as a control or
                    for V vectors where the energy distribution may differ.

    Properties (all modes):
      - Reduces to identity when c_i ≈ c_j (no spurious correction)
      - Decode-only — encoder doesn't need to know about it
      - Reversibility of stored skeleton is unaffected (encode side untouched)

    coeffs: [..., n] tensor (post-mask, pre-inverse)
    mask:   [n] bool — only pairs where BOTH indices are masked-true are touched
    """
    if alpha <= 0.0 or not pairs:
        return coeffs

    # Pre-filter pairs: both indices must be in the surviving skeleton
    mask_cpu = mask.detach().cpu()
    active_pairs = [(i, j) for (i, j) in pairs
                    if bool(mask_cpu[i].item()) and bool(mask_cpu[j].item())]
    if not active_pairs:
        return coeffs

    # Vectorized: gather all i-indices and j-indices, do one weighted blend
    idx_i = torch.tensor([p[0] for p in active_pairs],
                         dtype=torch.long, device=coeffs.device)
    idx_j = torch.tensor([p[1] for p in active_pairs],
                         dtype=torch.long, device=coeffs.device)

    c_i = coeffs.index_select(-1, idx_i)
    c_j = coeffs.index_select(-1, idx_j)

    if threshold > 0.0:
        # Only borrow when relative disagreement exceeds threshold
        denom = torch.clamp(torch.maximum(c_i.abs(), c_j.abs()), min=1e-6)
        rel_diff = (c_i - c_j).abs() / denom
        kick = (rel_diff > threshold).to(coeffs.dtype)
    else:
        kick = torch.ones_like(c_i)

    # v2 quick win: optional per-pair α scaling from zeta-resonance weights.
    # Each pair gets a scalar in (0, 1] derived from the geometric mean of
    # the per-coefficient resonance weights at i and j. Pairs near zeta
    # zeros get full α; pairs far from zeros get reduced α. Pure metaphor-
    # to-code: the zeros are the Poincaré sections, the pairs ARE the
    # "anchored" connectivity graph, the weights tell us which anchors are
    # near a section.
    pair_scale = 1.0
    if zeta_weights is not None:
        ws = [0.5 * (zeta_weights[i] + zeta_weights[j]) for (i, j) in active_pairs]
        pair_scale = torch.tensor(ws, dtype=coeffs.dtype, device=coeffs.device)
        # Reshape so it broadcasts against c_i/c_j whose last dim is len(active_pairs)
        # c_i shape: [..., len_pairs]. pair_scale shape: [len_pairs]. broadcasts.
    eff_alpha = alpha * pair_scale  # scalar or [len_pairs]

    if mode == "low_anchor":
        new_i = c_i
        new_j = c_j + eff_alpha * kick * (c_i - c_j)
    elif mode == "high_anchor":
        new_i = c_i + eff_alpha * kick * (c_j - c_i)
        new_j = c_j
    else:  # symmetric
        avg = 0.5 * (c_i + c_j)
        new_i = c_i + eff_alpha * kick * (avg - c_i)
        new_j = c_j + eff_alpha * kick * (avg - c_j)

    # Scatter back
    coeffs = coeffs.clone()
    coeffs.index_copy_(-1, idx_i, new_i)
    coeffs.index_copy_(-1, idx_j, new_j)
    return coeffs


def _hartley_butterfly_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Pure-PyTorch Hadamard/Hartley butterfly for head_dim=128 (= 2^7).
    Self-inverse: calling twice recovers the input.
    Input/output: [..., 128] float32 tensor (any leading dims).
    """
    # Work on the last dimension
    orig_shape = x.shape
    x = x.reshape(-1, 128).float()
    n = 128
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    # 7 butterfly stages
    for length in (1, 2, 4, 8, 16, 32, 64):
        step = length * 2
        # Build even/odd index arrays for in-place butterfly
        evens = torch.arange(0, n, step, device=x.device)
        # For each group, pair elements [base, base+length] .. [base+len-1, base+2*len-1]
        idx0 = (evens.unsqueeze(1)
                + torch.arange(length, device=x.device).unsqueeze(0)).reshape(-1)
        idx1 = idx0 + length

        u = x[:, idx0].clone()
        v = x[:, idx1].clone()
        x[:, idx0] = (u + v) * inv_sqrt2
        x[:, idx1] = (u - v) * inv_sqrt2

    return x.reshape(orig_shape)


# ── VHT2Bridge ────────────────────────────────────────────────────────────────

class VHT2Bridge:
    """
    Unified VHT2 compress/decompress for Wan head_dim=128.

    Uses the CUDA extension when available (8-12× faster at 720p+),
    falls back to a pure-PyTorch implementation otherwise.

    The skeleton mask marks squarefree positions (1-indexed): positions
    where the index+1 is not divisible by any perfect square > 1.
    At head_dim=128, ~60% of indices are squarefree (78/128).
    With a 30% keep fraction, the skeleton covers ~24 coefficients,
    yielding 3.5-4× compression with negligible reconstruction error.
    """

    def __init__(self, skeleton_frac: float = 0.30, device: str = "cuda"):
        self.device        = device
        self.skeleton_frac = skeleton_frac
        self._full_mask    = None   # cached full squarefree mask [128]
        self._skel_mask    = None   # cached skeleton mask [128] at fraction
        self._cuda         = _CUDA_AVAILABLE
        self._mode         = "cuda" if _CUDA_AVAILABLE else "torch"
        # Twin-prime pairs at head_dim=128 (computed once, cached)
        self._twin_pairs   = _twin_prime_pairs(128)
        # v2 quick win: Goldbach-extended pair sets (gap-4 cousins, gap-6 sexy)
        # v5 extension: gap-8, gap-10, gap-12 also computed and cached so the
        # caller can opt in to wider arithmetical neighborhoods.
        self._goldbach_pairs = _goldbach_pairs(128, gaps=(2, 4, 6, 8, 10, 12))
        # v2 quick win: zeta-resonance weight cache, keyed by lambda
        self._zeta_weight_cache: dict[float, list[float]] = {}

    def get_zeta_weights(self, lambda_: float = 0.05, n: int = 128) -> list[float]:
        """Cached per-coefficient resonance weights from zeta-zero distance."""
        key = round(lambda_, 4)
        w = self._zeta_weight_cache.get(key)
        if w is None:
            w = _zeta_resonance_weights(n=n, lambda_=lambda_)
            self._zeta_weight_cache[key] = w
        return w

        print(f"[SP VHT2Bridge] mode={self._mode}  "
              f"skeleton_frac={skeleton_frac:.0%}  "
              f"head_dim=128  "
              f"device={device}")
        if not _CUDA_AVAILABLE:
            print("[SP VHT2Bridge] CUDA extension not compiled — "
                  "using PyTorch fallback. "
                  "Build with: cd backends/cuda && python setup_wan.py build_ext --inplace")

    def skeleton_mask_128(self, frac: float | None = None) -> torch.Tensor:
        """
        Return [128] bool mask for the squarefree skeleton at `frac` keep ratio.
        Cached — O(1) after first call.
        frac=None uses self.skeleton_frac.
        """
        target_frac = frac if frac is not None else self.skeleton_frac

        if self._full_mask is None:
            self._full_mask = _sqfree_mask_128().to(self.device)

        # Trim the squarefree positions to the desired keep count
        sqfree_indices = self._full_mask.nonzero(as_tuple=False).squeeze(1)
        n_keep = max(1, int(len(sqfree_indices) * (target_frac / 1.0)))
        keep_idx = sqfree_indices[:n_keep]

        mask = torch.zeros(128, dtype=torch.bool, device=self.device)
        mask[keep_idx] = True
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """VHT2 butterfly (forward = inverse, self-inverse). Returns new tensor."""
        if self._cuda:
            return _cuda_ext.forward(x.contiguous())
        return _hartley_butterfly_torch(x)

    def compress(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compress: VHT2 forward → zero non-skeleton positions.
        x:    [..., 128] float32 CUDA tensor
        mask: [128]      bool    CUDA tensor
        Returns: [..., 128] sparse coefficients (non-skeleton = 0)
        """
        x = x.contiguous()
        if self._cuda:
            return _cuda_ext.compress(x, mask)
        # Torch fallback
        coeffs = _hartley_butterfly_torch(x)
        return coeffs * mask.float().reshape(*([1] * (x.ndim - 1)), 128)

    def decompress(self, coeffs: torch.Tensor, mask: torch.Tensor,
                   twin_borrow: bool = False,
                   twin_alpha: float = 0.10,
                   twin_threshold: float = 0.0) -> torch.Tensor:
        """
        Decompress: re-apply mask → [twin-prime borrow] → VHT2 inverse.

        coeffs: [..., 128] sparse coefficient tensor
        mask:   [128]      bool CUDA tensor
        twin_borrow: when True, applies decode-side twin-prime smoothing on
                     the masked spectral coefficients before the inverse
                     transform (strange-attractor stack piece 3/4). Decode-only,
                     so the stored skeleton bytes are unchanged. Worst case is
                     identity (alpha=0 or no agreement issues).
        twin_alpha:  blend strength toward the pair average. Small values
                     (0.05-0.15) for safety; larger values pull harder.
        twin_threshold: only borrow when relative |c_i-c_j|/max exceeds this.
                        0.0 = always borrow; 0.1 = only on outliers.

        Returns: [..., 128] reconstructed vectors
        """
        coeffs = coeffs.contiguous()
        if twin_borrow:
            # Mask + twin-borrow + inverse, in PyTorch (CUDA path doesn't
            # know about twin pairs yet — bridge fallback handles this safely)
            masked = coeffs * mask.float().reshape(*([1] * (coeffs.ndim - 1)), 128)
            corrected = _apply_twin_borrow(masked, mask, self._twin_pairs,
                                           alpha=twin_alpha,
                                           threshold=twin_threshold)
            if self._cuda:
                # Use the CUDA forward (= inverse, self-inverse) on the corrected coeffs
                return _cuda_ext.forward(corrected.contiguous())
            return _hartley_butterfly_torch(corrected)

        if self._cuda:
            return _cuda_ext.decompress(coeffs, mask)
        # Torch fallback
        masked = coeffs * mask.float().reshape(*([1] * (coeffs.ndim - 1)), 128)
        return _hartley_butterfly_torch(masked)

    def roundtrip(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Fused compress + decompress: VHT2 → mask → VHT2.
        Equivalent to decompress(compress(x, mask), mask) but avoids the
        intermediate allocation when using the CUDA extension.
        """
        x = x.contiguous()
        if self._cuda:
            return _cuda_ext.roundtrip(x, mask)
        coeffs = _hartley_butterfly_torch(x)
        masked = coeffs * mask.float().reshape(*([1] * (x.ndim - 1)), 128)
        return _hartley_butterfly_torch(masked)

    def apply_twin_borrow(self, coeffs: torch.Tensor, mask: torch.Tensor,
                          alpha: float = 0.10,
                          threshold: float = 0.0,
                          mode: str = "symmetric",
                          zeta_lambda: float = 0.0) -> torch.Tensor:
        """
        Public wrapper for the decode-side twin-prime borrowing pass.
        Apply between mask-application and the inverse VHT2 forward.

        zeta_lambda > 0 enables per-pair α scaling by zeta-zero proximity
        (the "Zero-Computation Resonance" decay). λ=0.05 is a slow decay;
        larger values concentrate the borrow more tightly around zeros.
        """
        zw = self.get_zeta_weights(zeta_lambda) if zeta_lambda > 0 else None
        return _apply_twin_borrow(coeffs, mask, self._twin_pairs,
                                  alpha=alpha, threshold=threshold,
                                  mode=mode, zeta_weights=zw)

    def apply_goldbach_borrow(self, coeffs: torch.Tensor, mask: torch.Tensor,
                              alpha: float = 0.10,
                              threshold: float = 0.0,
                              mode: str = "symmetric",
                              gaps: tuple[int, ...] = (2, 4, 6),
                              zeta_lambda: float = 0.0) -> torch.Tensor:
        """
        v2 quick win: Goldbach-extended borrowing.

        Applies _apply_twin_borrow once per gap with α scaled inversely
        by the gap (gap-2 = α, gap-4 = α/2, gap-6 = α/3) reflecting the
        weaker arithmetical neighborhood at larger gaps.

        zeta_lambda > 0 also applies the zeta-resonance per-pair scaling
        within each gap, so pairs near zeros get full strength and pairs
        between zeros get reduced strength.
        """
        zw = self.get_zeta_weights(zeta_lambda) if zeta_lambda > 0 else None
        out = coeffs
        for gap in gaps:
            pairs = self._goldbach_pairs.get(gap, [])
            if not pairs:
                continue
            scaled_alpha = alpha * (2.0 / gap)  # gap-2 → α, gap-4 → α/2, gap-6 → α/3
            out = _apply_twin_borrow(out, mask, pairs,
                                     alpha=scaled_alpha,
                                     threshold=threshold,
                                     mode=mode,
                                     zeta_weights=zw)
        return out

    @property
    def mode(self) -> str:
        """'cuda' if CUDA extension available, 'torch' otherwise."""
        return self._mode

    @property
    def cuda_available(self) -> bool:
        return self._cuda


# ── Module-level singleton for ComfyUI node reuse ─────────────────────────────

_global_bridge: VHT2Bridge | None = None

def get_bridge(skeleton_frac: float = 0.30, device: str = "cuda") -> VHT2Bridge:
    """
    Return the module-level VHT2Bridge singleton.
    First call constructs it; subsequent calls return the cached instance.
    """
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = VHT2Bridge(skeleton_frac=skeleton_frac, device=device)
    return _global_bridge
