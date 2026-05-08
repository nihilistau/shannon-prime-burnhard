# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
sp_benchmark.py — Shannon-Prime VHT2 Compression Benchmark

Demonstrates VHT2 KV cache compression on synthetic or real K/V vectors.
Reports correlation, compression ratio, and spectral analysis.

This does NOT require a model — it validates the compression math itself.
For model-level PPL testing, use sp_inject_freqs.py + llama.cpp perplexity.

Usage:
    python sp_benchmark.py                          # Default: hd=128, all configs
    python sp_benchmark.py --head-dim 64            # Mobile head_dim
    python sp_benchmark.py --config 5,5,4,3         # Specific config
    python sp_benchmark.py --n-vectors 1000         # More vectors for stable stats
    python sp_benchmark.py --spectral               # Show VHT2 spectral analysis
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backends', 'torch'))

import torch
torch.manual_seed(42)

from shannon_prime_torch import (
    vht2, MobiusMask, BandedQuantizer, ShadowCache, correlation
)


def benchmark_config(head_dim: int, band_bits: list, n_vectors: int = 500,
                     use_mobius: bool = True, use_rope_like: bool = True):
    """
    Benchmark a specific VHT2 bit allocation configuration.

    Returns dict with correlation stats, compression ratio, and timing.
    """
    bq = BandedQuantizer(head_dim, band_bits)
    mask = MobiusMask(head_dim) if use_mobius else None

    correlations = []
    total_time = 0.0

    for _ in range(n_vectors):
        # Generate test vector
        if use_rope_like:
            # Simulate RoPE-structured K vector (periodic components)
            t = torch.arange(head_dim, dtype=torch.float32)
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            orig = torch.zeros(head_dim)
            for p in primes[:5]:
                orig += torch.cos(2 * 3.14159 * t / p) / p
            orig += 0.1 * torch.randn(head_dim)
        else:
            orig = torch.randn(head_dim)

        t0 = time.perf_counter()

        # Write path: VHT2 → (Möbius) → band quantize
        work = vht2(orig.unsqueeze(0)).squeeze(0)

        if mask is not None:
            work = mask.reorder(work)

        scales, quants = bq.quantize(work.unsqueeze(0))
        recon = bq.dequantize(scales, quants).squeeze(0)

        if mask is not None:
            recon = mask.unreorder(recon)

        # Inverse VHT2 (self-inverse — same call as forward, no 1/N)
        recon = vht2(recon.unsqueeze(0)).squeeze(0)

        total_time += time.perf_counter() - t0

        correlations.append(correlation(orig, recon))

    corr_tensor = torch.tensor(correlations)
    bytes_per_vec = bq.compressed_bytes_per_vec()
    baseline_bytes = head_dim * 2  # fp16

    return {
        'config': '/'.join(map(str, band_bits)),
        'head_dim': head_dim,
        'mobius': use_mobius,
        'compression': baseline_bytes / bytes_per_vec,
        'bytes_per_vec': bytes_per_vec,
        'corr_mean': corr_tensor.mean().item(),
        'corr_std': corr_tensor.std().item(),
        'corr_min': corr_tensor.min().item(),
        'corr_max': corr_tensor.max().item(),
        'time_us': total_time / n_vectors * 1e6,
        'n_vectors': n_vectors,
        'signal_type': 'RoPE-like' if use_rope_like else 'random',
    }


def spectral_analysis(head_dim: int):
    """
    Show VHT2 spectral energy distribution for RoPE-like vs random vectors.
    This is the foundational observation: K concentrates, V doesn't.
    """
    n_bands = 4
    band_size = head_dim // n_bands

    print(f"\n{'═' * 60}")
    print(f"Spectral Analysis (hd={head_dim})")
    print(f"{'═' * 60}")

    for signal_type in ['RoPE-like K', 'Random V']:
        if signal_type == 'RoPE-like K':
            t = torch.arange(head_dim, dtype=torch.float32)
            vec = (torch.cos(2*3.14159*t/7) + 0.5*torch.cos(2*3.14159*t/13)
                   + 0.3*torch.cos(2*3.14159*t/3))
        else:
            vec = torch.randn(head_dim)

        work = vht2(vec.unsqueeze(0)).squeeze(0)

        band_energy = []
        total_energy = (work ** 2).sum().item()
        for b in range(n_bands):
            e = (work[b*band_size:(b+1)*band_size] ** 2).sum().item()
            band_energy.append(e)

        print(f"\n  {signal_type}:")
        print(f"  {'Band':>6} {'Energy%':>10} {'Bar'}")
        for b in range(n_bands):
            pct = 100.0 * band_energy[b] / total_energy if total_energy > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  {b:6d} {pct:9.1f}%  {bar}")

        first_half = (band_energy[0] + band_energy[1]) / total_energy * 100
        print(f"  First half: {first_half:.1f}% "
              f"({'concentrated → banded quant helps' if first_half > 60 else 'uniform → flat quant OK'})")


def run_full_benchmark(head_dim: int, n_vectors: int, configs: list = None):
    """Run benchmark across all standard configurations."""

    if configs is None:
        configs = [
            [5, 5, 4, 3],  # Ship default
            [5, 4, 4, 3],  # Balanced
            [4, 4, 4, 3],  # Aggressive
            [4, 3, 3, 3],  # Mobile
            [3, 3, 3, 3],  # Floor
        ]

    print(f"\n{'═' * 70}")
    print(f"Shannon-Prime VHT2 Compression Benchmark")
    print(f"{'═' * 70}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Test vectors:   {n_vectors}")
    print(f"  Signal type:    RoPE-like (structured, periodic)")
    print()

    # Header
    print(f"  {'Config':<12} {'Compress':>9} {'Corr Mean':>10} {'Corr Min':>10} "
          f"{'Bytes/vec':>10} {'μs/vec':>8}")
    print(f"  {'─'*12} {'─'*9} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    for bits in configs:
        result = benchmark_config(head_dim, bits, n_vectors, use_mobius=True)
        print(f"  {result['config']:<12} {result['compression']:>8.1f}× "
              f"{result['corr_mean']:>10.4f} {result['corr_min']:>10.4f} "
              f"{result['bytes_per_vec']:>10d} {result['time_us']:>7.1f}")

    # Möbius comparison
    print(f"\n  Möbius effect (config=5/5/4/3):")
    r_mob = benchmark_config(head_dim, [5,5,4,3], n_vectors, use_mobius=True)
    r_no  = benchmark_config(head_dim, [5,5,4,3], n_vectors, use_mobius=False)
    print(f"    With Möbius:    corr={r_mob['corr_mean']:.4f}")
    print(f"    Without Möbius: corr={r_no['corr_mean']:.4f}")
    print(f"    Improvement:    {r_mob['corr_mean'] - r_no['corr_mean']:+.4f}")

    # V vector comparison (flat beats banded)
    print(f"\n  V vector compression (flat vs banded):")
    r_flat   = benchmark_config(head_dim, [3], n_vectors, use_mobius=False, use_rope_like=False)
    r_banded = benchmark_config(head_dim, [5,5,4,3], n_vectors, use_mobius=False, use_rope_like=False)
    print(f"    Flat 3-bit:  corr={r_flat['corr_mean']:.4f}, {r_flat['compression']:.1f}×")
    print(f"    Banded 5543: corr={r_banded['corr_mean']:.4f}, {r_banded['compression']:.1f}×")
    print(f"    Flat is {'better' if r_flat['corr_mean'] >= r_banded['corr_mean'] - 0.005 else 'worse'}"
          f" at {r_flat['compression'] / r_banded['compression']:.1f}× more compression")

    # Memory projection
    print(f"\n  Memory projection (32K context, 32 layers, 8 KV heads):")
    n_slots = 32 * 8
    seq_len = 32768
    baseline = n_slots * seq_len * head_dim * 2 * 2 / 1024 / 1024
    for bits in [[5,5,4,3], [4,4,4,3], [3,3,3,3]]:
        bq_k = BandedQuantizer(head_dim, bits)
        bq_v = BandedQuantizer(head_dim, [3])
        compressed = n_slots * seq_len * (bq_k.compressed_bytes_per_vec() + bq_v.compressed_bytes_per_vec()) / 1024 / 1024
        ratio = baseline / compressed
        name = '/'.join(map(str, bits))
        print(f"    {name}: {compressed:.0f} MB (vs {baseline:.0f} MB fp16, {ratio:.1f}×)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Shannon-Prime VHT2 Compression Benchmark'
    )
    parser.add_argument('--head-dim', type=int, default=128,
                        help='Head dimension (64, 128, 256). Default: 128')
    parser.add_argument('--n-vectors', type=int, default=500,
                        help='Number of test vectors. Default: 500')
    parser.add_argument('--config', type=str, default=None,
                        help='Specific bit config (e.g. "5,5,4,3")')
    parser.add_argument('--spectral', action='store_true',
                        help='Show spectral energy analysis')

    args = parser.parse_args()

    if args.spectral:
        spectral_analysis(args.head_dim)

    if args.config:
        bits = [int(b) for b in args.config.split(',')]
        run_full_benchmark(args.head_dim, args.n_vectors, configs=[bits])
    else:
        run_full_benchmark(args.head_dim, args.n_vectors)


if __name__ == '__main__':
    main()
