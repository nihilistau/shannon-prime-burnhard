#!/usr/bin/env python3
# Shannon-Prime: compute a per-band bit allocation from measured K energies.
#
# Reads a raw K dump produced by the sqfree llama hook when
# SHANNON_PRIME_DUMP_K=<path> is set, computes per-band VHT2 energy, and
# emits a comma-separated K_BITS string sized to the requested band count
# and total bit budget.
#
# Typical use:
#
#   SHANNON_PRIME_ENABLED=1 SHANNON_PRIME_SQFREE=1 SHANNON_PRIME_SPINOR=1 \
#   SHANNON_PRIME_DUMP_K=/tmp/k_dump.bin \
#   SHANNON_PRIME_DUMP_K_LIMIT=8192 \
#       llama-perplexity -m model.gguf -f wiki.test.raw -c 2048 --chunks 2 \
#         -ngl 99
#
#   python tools/sp_auto_bands.py \
#       --dump /tmp/k_dump.bin \
#       --head-dim 128 \
#       --n-bands 10 \
#       --total-bits 35 \
#       --min-bits 3 --max-bits 6
#   # → "4,4,4,4,4,3,3,3,3,3"
#
# `--head-dim` is the dump's element count (the llama hook writes raw K
# at the model's head_dim, not pad_dim). For sqfree models the analyser
# operates on a proxy basis — the Walsh basis at head_dim — which is a
# monotonic-decay stand-in for the runtime's sqfree-Vilenkin bands.
# The bit ordering holds but absolute magnitudes aren't comparable.
#
# The allocation follows
#     bits_b ≈ round( total_bits × log2(var_b + eps) / Σ_b log2(var_b + eps) )
# clipped into [min_bits, max_bits], with the residual rebalanced so the sum
# equals total_bits. The log is a proxy for Shannon's optimal bit allocation
# for a Gaussian source; it tracks measured VHT2 decay without needing an
# offline search.

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np


def _vht2_p2(x: np.ndarray) -> np.ndarray:
    """In-place power-of-2 VHT2 (p=2 Hadamard butterfly, 1/sqrt(2) per stage)."""
    n = x.shape[-1]
    if (n & (n - 1)) != 0:
        raise ValueError(f"power-of-2 VHT2 requires n=2^k, got n={n}")
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[..., j].copy()
                b = x[..., j + h].copy()
                x[..., j]     = (a + b) * (1.0 / math.sqrt(2))
                x[..., j + h] = (a - b) * (1.0 / math.sqrt(2))
        h *= 2
    return x


def _vht2_vilenkin(x: np.ndarray, primes: list[int]) -> np.ndarray:
    """Staged Hartley for arbitrary-prime products (matches vilenkin.comp)."""
    stride = 1
    n = x.shape[-1]
    for p in primes:
        block = stride * p
        n_blocks = n // block
        isp = 1.0 / math.sqrt(p)
        for blk in range(n_blocks):
            for s in range(stride):
                base = blk * block + s
                idx = [base + k * stride for k in range(p)]
                gathered = x[..., idx].copy()
                result = np.zeros_like(gathered)
                for k in range(p):
                    tot = np.zeros(x.shape[:-1], dtype=np.float32)
                    for j in range(p):
                        ang = 2.0 * math.pi * k * j / p
                        tot += (math.cos(ang) + math.sin(ang)) * gathered[..., j]
                    result[..., k] = tot * isp
                x[..., idx] = result
        stride *= p
    return x


def _factor_small_primes(n: int) -> list[int]:
    # Must match core/shannon_prime.c's SP_VHT2_MAX_P (=11). The runtime VHT2
    # won't apply a 13-point stage, so accepting 13 here would let us report
    # energies on a basis the runtime can't produce.
    factors = []
    for p in (2, 3, 5, 7, 11):
        while n % p == 0:
            factors.append(p)
            n //= p
    if n != 1:
        raise ValueError(f"head_dim has a prime factor > 11 (core VHT2 cap), "
                         f"got remainder {n}")
    return factors


def _band_energies(k_vht2: np.ndarray, n_bands: int) -> np.ndarray:
    """Return mean-squared magnitude per band, absorbing the remainder in the last band."""
    n = k_vht2.shape[-1]
    band_size = n // n_bands
    energies = np.zeros(n_bands, dtype=np.float64)
    for b in range(n_bands):
        off = b * band_size
        sz  = (n - off) if b == n_bands - 1 else band_size
        slc = k_vht2[..., off:off + sz]
        energies[b] = float((slc ** 2).mean())
    return energies


def _allocate(energies: np.ndarray, total_bits: int,
              min_bits: int, max_bits: int) -> list[int]:
    eps = 1e-12
    weights = np.log2(np.maximum(energies, eps) + eps)
    # shift so min weight → 0; this avoids negative proportional shares
    weights = weights - weights.min() + 1.0
    weights = weights / weights.sum()
    raw = weights * total_bits
    bits = np.clip(np.round(raw).astype(int), min_bits, max_bits)

    # Rebalance to hit total_bits exactly (drift from clipping/rounding)
    delta = int(total_bits - bits.sum())
    order = np.argsort(-raw) if delta > 0 else np.argsort(raw)
    i = 0
    while delta != 0 and i < 4 * len(bits):
        idx = int(order[i % len(order)])
        if delta > 0 and bits[idx] < max_bits:
            bits[idx] += 1
            delta -= 1
        elif delta < 0 and bits[idx] > min_bits:
            bits[idx] -= 1
            delta += 1
        i += 1
    return bits.tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto K_BITS from measured VHT2 energy")
    ap.add_argument("--dump", required=True,
                    help="Path to K-dump .bin (fp32) produced by SHANNON_PRIME_DUMP_K")
    ap.add_argument("--head-dim", type=int, required=True,
                    help="Element count of each dumped vector. The llama hook "
                         "writes raw K at the model's head_dim (not pad_dim), "
                         "so pass 128 for a hd=128 model even when sqfree is on.")
    ap.add_argument("--n-bands", type=int, required=True,
                    help="Target number of bands (e.g. 10)")
    ap.add_argument("--total-bits", type=int, required=True,
                    help="Sum of bits across bands (e.g. 30 for 10×3 avg)")
    # Default min-bits is 3 per CLAUDE.md invariant #3: "3-bit floor; 2-bit
    # is catastrophic" (measured at PPL=428.78 on Qwen3-8B 10×2 — see
    # logs/cuda_qwen3_sqfree_10band_2.json). Override at your own risk.
    ap.add_argument("--min-bits", type=int, default=3)
    ap.add_argument("--max-bits", type=int, default=6)
    ap.add_argument("--max-vecs", type=int, default=8192,
                    help="Cap on K vectors to read (default 8192)")
    ap.add_argument("--no-vht2", action="store_true",
                    help="Assume dump is already in VHT2 space (skip forward transform). "
                         "Implied by --basis vilenkin.")
    ap.add_argument("--basis", choices=["raw", "vilenkin"], default="raw",
                    help="`raw` (default): dump is raw K at head_dim; the analyser "
                         "applies Walsh-at-hd as a proxy basis. `vilenkin`: dump "
                         "came from SHANNON_PRIME_DUMP_VILENKIN=<path> and is "
                         "already in the runtime's sqfree-Vilenkin space — the "
                         "analyser skips its forward transform entirely.")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON with energies + bits instead of comma list")
    args = ap.parse_args()

    path = pathlib.Path(args.dump)
    if not path.exists():
        print(f"error: dump not found: {path}", file=sys.stderr)
        return 2

    fp32_bytes = path.stat().st_size
    if fp32_bytes % (args.head_dim * 4) != 0:
        print(f"error: dump size {fp32_bytes} not a multiple of head_dim*4", file=sys.stderr)
        return 2
    n_vecs = min(fp32_bytes // (args.head_dim * 4), args.max_vecs)

    raw = np.fromfile(path, dtype=np.float32, count=n_vecs * args.head_dim)
    raw = raw.reshape(n_vecs, args.head_dim).copy()

    skip_forward = args.no_vht2 or args.basis == "vilenkin"
    if not skip_forward:
        if (args.head_dim & (args.head_dim - 1)) == 0:
            _vht2_p2(raw)
        else:
            primes = _factor_small_primes(args.head_dim)
            _vht2_vilenkin(raw, primes)

    energies = _band_energies(raw, args.n_bands)
    bits = _allocate(energies, args.total_bits, args.min_bits, args.max_bits)

    if args.json:
        import json
        print(json.dumps({
            "n_vecs": int(n_vecs),
            "head_dim": args.head_dim,
            "n_bands": args.n_bands,
            "total_bits": args.total_bits,
            "band_energies": energies.tolist(),
            "k_bits": bits,
        }, indent=2))
    else:
        print(",".join(str(b) for b in bits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
