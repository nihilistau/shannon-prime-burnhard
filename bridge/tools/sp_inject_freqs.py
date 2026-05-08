# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
sp_inject_freqs.py — Shannon-Prime Frequency Injection for GGUF Models

Blends prime-lattice-aligned RoPE frequencies into an existing GGUF model's
geometric frequencies. This corrects positional frequency mismatch — a
different error from what weight quantization introduces.

The result is a new GGUF file with modified rope frequency factors.
Zero retraining required. The model loads and runs normally in llama.cpp;
the only difference is the RoPE frequencies used during inference.

Paper results (Position Is Arithmetic v7):
  Dolphin 3.0 Llama 3.2 1B:
    Q8_0:  11.6413 → 11.5462 PPL (−0.82%) at α=0.22
    Q6_K:  11.7615 → 11.6843 PPL (−0.66%) at α=0.17
    Q4_K_M: 12.2380 → 12.1630 PPL (−0.61%) at α=0.17

  Optimal α range: 0.15–0.22 (flat optimum, deployment-robust)
  Higher rope_freq_base → wider harmonic gaps → more room for injection

Usage:
    python sp_inject_freqs.py input.gguf output.gguf --alpha 0.17
    python sp_inject_freqs.py input.gguf output.gguf --alpha 0.22 --tier-mode composite
    python sp_inject_freqs.py input.gguf output.gguf --info   # Just show model info
"""

import argparse
import sys
import os
import math
import struct
import shutil
import numpy as np

try:
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False


# =============================================================================
# Prime / Composite frequency generation
# =============================================================================

def sieve_primes(n: int) -> list:
    """Sieve of Eratosthenes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]


def generate_tiered_frequencies(n_freqs: int, mode: str = 'prime') -> np.ndarray:
    """
    Generate lattice-aligned frequencies allocated across three tiers.

    Tiers (from paper §3.1):
      Local (25%):  range 2..101    — word/syntax scale
      Mid   (33%):  range 101..1009 — clause/paragraph scale
      Long  (42%):  range 1009..8209 — section/document scale

    Per-head scale diversity is load-bearing. Replacing ALL frequencies
    uniformly caused +370% context degradation. Tiered allocation preserves
    diversity while introducing arithmetic structure.

    mode:
      'prime'     — use prime numbers as frequencies (generators of the lattice)
      'composite' — use composites (coordinates in the lattice)
                    Paper showed identical performance: prime 129.2 PPL vs composite 129.4 PPL
    """
    # Tier allocation
    n_local = max(1, int(n_freqs * 0.25))
    n_mid   = max(1, int(n_freqs * 0.33))
    n_long  = n_freqs - n_local - n_mid

    primes = sieve_primes(10000)

    if mode == 'prime':
        candidates = primes
    elif mode == 'composite':
        prime_set = set(primes)
        candidates = [n for n in range(4, 10000) if n not in prime_set]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'prime' or 'composite'.")

    # Filter by tier ranges
    local_pool = [f for f in candidates if 2 <= f <= 101]
    mid_pool   = [f for f in candidates if 101 < f <= 1009]
    long_pool  = [f for f in candidates if 1009 < f <= 8209]

    # Select evenly spaced from each pool
    def pick(pool, n):
        if n == 0:
            return []
        if len(pool) <= n:
            return pool
        step = len(pool) / n
        return [pool[int(i * step)] for i in range(n)]

    freqs = pick(local_pool, n_local) + pick(mid_pool, n_mid) + pick(long_pool, n_long)

    # Pad or trim to exact count
    while len(freqs) < n_freqs:
        freqs.append(freqs[-1] + 1)
    freqs = freqs[:n_freqs]

    return np.array(freqs, dtype=np.float32)


def generate_geometric_frequencies(n_freqs: int, freq_base: float = 10000.0) -> np.ndarray:
    """Standard RoPE geometric frequency sequence: θ_j = base^(-2j/d)."""
    d = n_freqs * 2  # Each frequency pair covers 2 dimensions
    return np.array([
        freq_base ** (-2.0 * j / d) for j in range(n_freqs)
    ], dtype=np.float32)


def blend_frequencies(geometric: np.ndarray, lattice: np.ndarray,
                      alpha: float) -> np.ndarray:
    """
    Blend geometric and lattice frequencies at ratio α.

    result = (1 - α) * geometric + α * lattice_normalized

    The lattice frequencies are normalized to the same scale as the geometric
    frequencies before blending, so α=0 gives pure geometric and α=1 gives
    pure lattice (not recommended — diversity matters).

    Optimal α from paper: 0.15–0.22
    """
    # Normalize lattice to geometric scale
    # Map lattice freq range to geometric freq range
    geo_min, geo_max = geometric.min(), geometric.max()
    lat_min, lat_max = lattice.min(), lattice.max()

    if lat_max > lat_min:
        lattice_norm = geo_min + (lattice - lat_min) / (lat_max - lat_min) * (geo_max - geo_min)
    else:
        lattice_norm = geometric.copy()

    blended = (1.0 - alpha) * geometric + alpha * lattice_norm
    return blended.astype(np.float32)


def frequencies_to_freq_factors(blended: np.ndarray, geometric: np.ndarray) -> np.ndarray:
    """
    Convert blended frequencies to freq_factors (multipliers on the base frequency).

    llama.cpp's freq_factors mechanism: actual_freq[i] = base_freq[i] * freq_factor[i]
    So: freq_factor[i] = blended[i] / geometric[i]

    This is the mechanism used by models like CodeLlama for extended context,
    and it's what we use to inject lattice frequencies at inference time.
    """
    factors = np.where(geometric > 0, blended / geometric, 1.0)
    return factors.astype(np.float32)


# =============================================================================
# GGUF model info
# =============================================================================

def print_model_info(path: str):
    """Print relevant RoPE and architecture info from a GGUF file."""
    reader = GGUFReader(path)

    print(f"\nModel: {path}")
    print(f"{'─' * 60}")

    arch = None
    for field in reader.fields.values():
        name = field.name
        if name == 'general.architecture':
            arch = str(bytes(field.parts[-1]), 'utf-8')
            print(f"  Architecture:    {arch}")
        elif name == 'general.name':
            print(f"  Name:            {str(bytes(field.parts[-1]), 'utf-8')}")
        elif 'head_count_kv' in name:
            val = int(field.parts[-1][0])
            print(f"  KV heads:        {val}")
        elif 'head_count' in name and 'kv' not in name:
            val = int(field.parts[-1][0])
            print(f"  Attention heads: {val}")
        elif 'block_count' in name:
            val = int(field.parts[-1][0])
            print(f"  Layers:          {val}")
        elif 'embedding_length' in name:
            val = int(field.parts[-1][0])
            print(f"  Embedding dim:   {val}")
        elif 'rope.dimension_count' in name:
            val = int(field.parts[-1][0])
            print(f"  RoPE dimensions: {val}")
        elif 'rope.freq_base' in name:
            val = float(field.parts[-1][0])
            print(f"  RoPE freq_base:  {val}")
        elif 'context_length' in name:
            val = int(field.parts[-1][0])
            print(f"  Context length:  {val}")
        elif 'file_type' in name:
            val = int(field.parts[-1][0])
            print(f"  File type:       {val}")

    # Check for existing freq factors tensor
    has_freq_factors = False
    for tensor in reader.tensors:
        if 'rope_freqs' in tensor.name or 'freq_factors' in tensor.name.lower():
            has_freq_factors = True
            print(f"  Freq factors:    PRESENT ({tensor.name}, shape {tensor.shape})")

    if not has_freq_factors:
        print(f"  Freq factors:    not present (will use geometric default)")

    n_tensors = len(reader.tensors)
    print(f"  Total tensors:   {n_tensors}")
    print()


# =============================================================================
# GGUF frequency injection
# =============================================================================

def inject_frequencies(input_path: str, output_path: str,
                       alpha: float, tier_mode: str = 'composite',
                       verbose: bool = True):
    """
    Read a GGUF model, inject Shannon-Prime lattice frequencies,
    write a new GGUF file.

    The injection uses llama.cpp's freq_factors mechanism:
    a tensor named 'rope_freqs.weight' containing per-dimension multipliers.
    This is the same mechanism CodeLlama uses for extended context.
    """
    if not HAS_GGUF:
        print("ERROR: gguf package not installed. Run: pip install gguf")
        sys.exit(1)

    reader = GGUFReader(input_path)

    # Extract RoPE params
    rope_dim = None
    freq_base = 10000.0
    arch = 'llama'
    n_heads = None

    for field in reader.fields.values():
        if 'rope.dimension_count' in field.name:
            rope_dim = int(field.parts[-1][0])
        elif 'rope.freq_base' in field.name:
            freq_base = float(field.parts[-1][0])
        elif field.name == 'general.architecture':
            arch = str(bytes(field.parts[-1]), 'utf-8')
        elif 'head_count' in field.name and 'kv' not in field.name:
            n_heads = int(field.parts[-1][0])

    if rope_dim is None:
        print("ERROR: Could not find rope.dimension_count in model metadata.")
        sys.exit(1)

    n_freqs = rope_dim // 2  # Each freq covers a (cos, sin) pair

    if verbose:
        print(f"\nShannon-Prime Frequency Injection")
        print(f"{'─' * 50}")
        print(f"  Input:        {input_path}")
        print(f"  Output:       {output_path}")
        print(f"  Architecture: {arch}")
        print(f"  RoPE dim:     {rope_dim}")
        print(f"  Freq pairs:   {n_freqs}")
        print(f"  Freq base:    {freq_base}")
        print(f"  Alpha:        {alpha}")
        print(f"  Tier mode:    {tier_mode}")

    # Generate frequencies
    geometric = generate_geometric_frequencies(n_freqs, freq_base)
    lattice   = generate_tiered_frequencies(n_freqs, mode=tier_mode)
    blended   = blend_frequencies(geometric, lattice, alpha)
    factors   = frequencies_to_freq_factors(blended, geometric)

    if verbose:
        print(f"\n  Freq factor stats:")
        print(f"    min:    {factors.min():.4f}")
        print(f"    max:    {factors.max():.4f}")
        print(f"    mean:   {factors.mean():.4f}")
        print(f"    std:    {factors.std():.4f}")

    # v1.03: produce a real modified GGUF that inserts a single shared
    # `rope_freqs.weight` tensor (no `blk.<i>.` prefix) carrying the
    # blended factors. llama.cpp's LLM_TENSOR_ROPE_FREQS resolves to the
    # unqualified root name even though it's declared LAYER_REPEATING —
    # every layer reads the same tensor via `get_rope_factors`. No
    # runtime env or sidecar consumer is needed; loading the output
    # GGUF is enough.
    #
    # Fall back to the legacy byte-identical copy + sidecar path when either
    # the `gguf` package isn't importable or the block count cannot be
    # determined from metadata.

    factors_path = output_path.rsplit('.', 1)[0] + '.sp_freq_factors.bin'
    factors.tofile(factors_path)
    if verbose:
        print(f"\n  Freq factors sidecar: {factors_path}")
        print(f"    ({n_freqs} × float32 = {n_freqs * 4} bytes, kept for debugging)")

    n_layers = None
    for field in reader.fields.values():
        if field.name.endswith('.block_count'):
            n_layers = int(field.parts[-1][0])
            break

    if not HAS_GGUF or n_layers is None:
        shutil.copy2(input_path, output_path)
        if verbose:
            print(f"\n  Output written to: {output_path}")
            print(f"  (legacy byte-identical path — gguf package missing or")
            print(f"   block count not in metadata; load SHANNON_PRIME_SIDECAR={factors_path}")
            print(f"   to apply via the llama.cpp integration)")
        return factors

    _rewrite_gguf_with_rope_freqs(reader, input_path, output_path,
                                   arch=arch, n_layers=n_layers,
                                   factors=factors.astype(np.float32),
                                   verbose=verbose)
    return factors


def _rewrite_gguf_with_rope_freqs(reader, input_path, output_path,
                                  arch, n_layers, factors, verbose=True):
    """Rewrite `input_path` as `output_path` and inject `factors` as the
    shared `rope_freqs.weight` tensor (read by every layer via
    LLM_TENSOR_ROPE_FREQS — the tensor name carries no blk prefix even
    though the category is LAYER_REPEATING).

    Copies every field and existing tensor as-is; the only change is
    replacing (or adding) the shared rope_freqs.weight. Uses the `gguf`
    package that ships with llama.cpp / Hugging Face convert scripts.
    """
    writer = GGUFWriter(output_path, arch)

    def _scalar_from(field, idx):
        raw = field.parts[idx]
        if field.types[0] == GGUFValueType.STRING:
            return raw.tobytes().decode('utf-8', errors='replace')
        return raw.tolist()[0] if hasattr(raw, 'tolist') and raw.shape else raw.item()

    # --- copy metadata (skip architecture + GGUF-internal bookkeeping; the
    # writer supplies header/version/counts itself) -------------------------
    skip_keys = {'general.architecture', 'GGUF.version',
                 'GGUF.tensor_count', 'GGUF.kv_count'}
    for field in reader.fields.values():
        if field.name in skip_keys:
            continue
        if len(field.types) == 0 or len(field.data) == 0:
            continue
        try:
            if field.types[0] == GGUFValueType.ARRAY:
                inner = field.types[-1]
                if inner == GGUFValueType.STRING:
                    values = [field.parts[di].tobytes().decode('utf-8', errors='replace')
                              for di in field.data]
                else:
                    values = [field.parts[di].tolist()[0] if hasattr(field.parts[di], 'tolist') and field.parts[di].shape
                              else field.parts[di].item()
                              for di in field.data]
                writer.add_array(field.name, values)
            else:
                val = _scalar_from(field, field.data[0])
                writer.add_key_value(field.name, val, field.types[0])
        except Exception as e:
            if verbose:
                print(f"  warning: skipping metadata field '{field.name}': {e}")

    # --- copy tensors (streaming — don't load all into memory) -------------
    # GGUFReader.tensors[i].data is a np.uint8 array in the raw packed byte
    # layout the file uses. GGUFWriter.add_tensor with `raw_dtype=...` set
    # and no explicit raw_shape takes tensor.shape (=byte shape) and calls
    # quant_shape_from_byte_shape() internally to recover the logical shape.
    # This matches the GGUFReader inverse exactly and is correct for both
    # unquantised (F32/F16/BF16) and quantised (Q8_0/Q4_K_M/...) tensors.
    #
    # The special case: `rope_freqs.weight` is a single shared tensor (not
    # per-layer — llama.cpp's LLM_TENSOR_ROPE_FREQS maps to the root name
    # even though it's LAYER_REPEATING, so every layer reads the same
    # tensor). If the input already has it, we REPLACE the data with our
    # blended factors; otherwise we add it alongside the existing tensors.
    ROPE_NAME = "rope_freqs.weight"
    had_rope  = False
    for ti in reader.tensors:
        if ti.name == ROPE_NAME:
            had_rope = True
            writer.add_tensor(ROPE_NAME, factors,
                              raw_shape=[factors.shape[0]],
                              raw_dtype=GGMLQuantizationType.F32)
        else:
            writer.add_tensor(ti.name, ti.data, raw_dtype=ti.tensor_type)

    if not had_rope:
        writer.add_tensor(ROPE_NAME, factors,
                          raw_shape=[factors.shape[0]],
                          raw_dtype=GGMLQuantizationType.F32)

    # --- write -------------------------------------------------------------
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    if verbose:
        in_sz  = os.path.getsize(input_path)
        out_sz = os.path.getsize(output_path)
        delta  = out_sz - in_sz
        action = "replaced rope_freqs.weight" if had_rope else "added rope_freqs.weight"
        print(f"\n  Output GGUF: {output_path}")
        print(f"  {action} ({factors.shape[0]} × fp32, shared across all {n_layers} layers)")
        print(f"  Size: {in_sz:,} → {out_sz:,} bytes (delta {delta:+,})")
        print(f"  No runtime env required — llama.cpp picks up rope_freqs via the standard loader.")


# =============================================================================
# Standalone analysis tool
# =============================================================================

def analyze_frequencies(input_path: str, alpha: float = 0.17,
                        tier_mode: str = 'composite'):
    """Analyze what frequency injection would do, without modifying anything."""
    reader = GGUFReader(input_path)

    rope_dim = None
    freq_base = 10000.0
    for field in reader.fields.values():
        if 'rope.dimension_count' in field.name:
            rope_dim = int(field.parts[-1][0])
        elif 'rope.freq_base' in field.name:
            freq_base = float(field.parts[-1][0])

    if rope_dim is None:
        print("ERROR: No rope.dimension_count found.")
        return

    n_freqs = rope_dim // 2
    geometric = generate_geometric_frequencies(n_freqs, freq_base)
    lattice = generate_tiered_frequencies(n_freqs, mode=tier_mode)
    blended = blend_frequencies(geometric, lattice, alpha)
    factors = frequencies_to_freq_factors(blended, geometric)

    print(f"\nFrequency Analysis (α={alpha}, mode={tier_mode})")
    print(f"{'─' * 60}")
    print(f"  RoPE dim: {rope_dim}, freq pairs: {n_freqs}, base: {freq_base}")
    print(f"\n  {'Idx':>4} {'Geometric':>12} {'Lattice':>12} {'Blended':>12} {'Factor':>10}")
    print(f"  {'─'*4} {'─'*12} {'─'*12} {'─'*12} {'─'*10}")

    for i in range(min(n_freqs, 20)):
        print(f"  {i:4d} {geometric[i]:12.6f} {lattice[i]:12.2f} "
              f"{blended[i]:12.6f} {factors[i]:10.4f}")

    if n_freqs > 20:
        print(f"  ... ({n_freqs - 20} more)")

    print(f"\n  Factor range: [{factors.min():.4f}, {factors.max():.4f}]")
    print(f"  Expected PPL improvement: ~0.5–0.8% (varies by model/quant)")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Shannon-Prime: Inject lattice-aligned RoPE frequencies into GGUF models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show model RoPE info
  python sp_inject_freqs.py model.gguf --info

  # Analyze without modifying
  python sp_inject_freqs.py model.gguf --analyze --alpha 0.17

  # Inject at optimal alpha
  python sp_inject_freqs.py model.gguf model_sp.gguf --alpha 0.17

  # Use prime mode instead of composite
  python sp_inject_freqs.py model.gguf model_sp.gguf --alpha 0.22 --tier-mode prime

  # Alpha sweep (for finding optimal value)
  for a in 0.05 0.10 0.15 0.17 0.20 0.22 0.25; do
    python sp_inject_freqs.py model.gguf model_a${a}.gguf --alpha $a
  done
        """
    )

    parser.add_argument('input', help='Input GGUF model file')
    parser.add_argument('output', nargs='?', help='Output GGUF file (required unless --info/--analyze)')
    parser.add_argument('--alpha', type=float, default=0.17,
                        help='Blending ratio (0=pure geometric, 1=pure lattice). Default: 0.17')
    parser.add_argument('--tier-mode', choices=['prime', 'composite'], default='composite',
                        help='Frequency source: prime (lattice generators) or composite (lattice coordinates). Default: composite')
    parser.add_argument('--info', action='store_true',
                        help='Print model info and exit')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze frequency injection without modifying')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    if not HAS_GGUF:
        print("ERROR: gguf package not installed. Run: pip install gguf")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    if args.info:
        print_model_info(args.input)
        return

    if args.analyze:
        analyze_frequencies(args.input, args.alpha, args.tier_mode)
        return

    if not args.output:
        parser.error("Output file required (unless using --info or --analyze)")

    inject_frequencies(args.input, args.output,
                       alpha=args.alpha,
                       tier_mode=args.tier_mode,
                       verbose=not args.quiet)


if __name__ == '__main__':
    main()
