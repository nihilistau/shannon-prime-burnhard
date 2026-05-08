# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
sp_compress_model.py — Shannon-Prime Model Compression

Applies VHT2 spectral compression to transformer model weights.

The insight: W_K and W_Q weight matrices are trained with RoPE applied.
The learned weights implicitly encode the frequency patterns they were
optimized for. VHT2 exploits this structure for better compression
than generic quantization — on the specific tensors that have it.

Strategy:
  - W_K, W_Q: VHT2 spectral compression (VHT2 → Möbius → banded quant)
    These have RoPE-induced spectral structure in head_dim rows.
  - W_V, W_O, FFN: Standard quantization (no RoPE structure)
  - Norms, embeddings: Keep at higher precision (standard practice)

This produces a hybrid-quantized model: structure-aware on Q/K weights,
standard on everything else. The VHT2-compressed Q/K weights need a
custom dequantization path at inference time.

Supports:
  - HuggingFace safetensors → Shannon-Prime compressed format
  - GGUF → GGUF with VHT2-compressed Q/K tensors + freq injection
  - Analysis mode: measure spectral concentration without modifying

Usage:
    python sp_compress_model.py --analyze model_dir/     # Measure structure
    python sp_compress_model.py model_dir/ output_dir/   # Compress
    python sp_compress_model.py model.gguf model_sp.gguf # GGUF to GGUF
"""

import argparse
import sys
import os
import json
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backends', 'torch'))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from shannon_prime_torch import (
        vht2, MobiusMask, BandedQuantizer, correlation
    )


# =============================================================================
# Spectral analysis of weight tensors
# =============================================================================

def analyze_tensor_spectrum(tensor: 'torch.Tensor', name: str,
                           head_dim: int = 128) -> dict:
    """
    Analyze VHT2 spectral concentration of a weight tensor.

    For W_K/W_Q: reshape to (n_heads, head_dim, model_dim), analyze each
    head's projection vectors (each row of head_dim length).

    Returns spectral concentration metrics.
    """
    if not HAS_TORCH:
        return {}

    t = tensor.float()

    # Determine if this is a Q/K/V/O weight and reshape appropriately
    if t.dim() == 2:
        out_dim, in_dim = t.shape
        if out_dim % head_dim == 0:
            n_heads = out_dim // head_dim
            # Reshape to (n_heads * in_dim, head_dim) — each row is head_dim
            t_reshaped = t.view(n_heads, head_dim, in_dim)
            # Analyze along the head_dim axis
            vectors = t_reshaped.reshape(-1, head_dim)
        elif in_dim % head_dim == 0:
            n_heads = in_dim // head_dim
            t_reshaped = t.view(out_dim, n_heads, head_dim)
            vectors = t_reshaped.reshape(-1, head_dim)
        else:
            # Not a head_dim-aligned tensor
            return {'name': name, 'aligned': False}
    else:
        return {'name': name, 'aligned': False}

    if vectors.shape[-1] != head_dim or not (head_dim & (head_dim - 1) == 0):
        return {'name': name, 'aligned': False}

    # Sample up to 1000 vectors for analysis speed
    n_sample = min(vectors.shape[0], 1000)
    idx = torch.randperm(vectors.shape[0])[:n_sample]
    sample = vectors[idx].clone()

    # VHT2 transform (self-inverse; the band energies below are scale-invariant
    # so the 1/√N normalisation doesn't affect the concentration metric).
    sample = vht2(sample)

    # Compute per-band energy
    n_bands = 4
    band_size = head_dim // n_bands
    total_energy = (sample ** 2).sum(dim=-1)

    band_energies = []
    for b in range(n_bands):
        be = (sample[:, b*band_size:(b+1)*band_size] ** 2).sum(dim=-1)
        band_energies.append(be)

    # Concentration ratio: first half vs total
    first_half = (band_energies[0] + band_energies[1]) / total_energy.clamp(min=1e-12)
    concentration = first_half.mean().item()

    # Compare VHT2 vs flat quantization quality
    mask = MobiusMask(head_dim)
    bq_vht2 = BandedQuantizer(head_dim, [5, 5, 4, 3])
    bq_flat = BandedQuantizer(head_dim, [4, 4, 4, 4])

    vht2_corrs = []
    flat_corrs = []
    for i in range(min(n_sample, 200)):
        orig = vectors[idx[i]].clone()

        # VHT2 path (self-inverse: forward = inverse, no 1/N)
        w = vht2(orig.unsqueeze(0)).squeeze(0)
        w = mask.reorder(w)
        s, q = bq_vht2.quantize(w.unsqueeze(0))
        r = bq_vht2.dequantize(s, q).squeeze(0)
        r = mask.unreorder(r)
        r = vht2(r.unsqueeze(0)).squeeze(0)
        vht2_corrs.append(correlation(orig, r))

        # Flat path (same total bits, uniform allocation)
        w2 = vht2(orig.unsqueeze(0)).squeeze(0)
        s2, q2 = bq_flat.quantize(w2.unsqueeze(0))
        r2 = bq_flat.dequantize(s2, q2).squeeze(0)
        r2 = vht2(r2.unsqueeze(0)).squeeze(0)
        flat_corrs.append(correlation(orig, r2))

    vht2_mean = sum(vht2_corrs) / len(vht2_corrs)
    flat_mean = sum(flat_corrs) / len(flat_corrs)

    return {
        'name': name,
        'aligned': True,
        'shape': list(tensor.shape),
        'n_vectors': vectors.shape[0],
        'head_dim': head_dim,
        'concentration': concentration,
        'band_energy_pct': [
            (be / total_energy.clamp(min=1e-12)).mean().item() * 100
            for be in band_energies
        ],
        'vht2_corr': vht2_mean,
        'flat_corr': flat_mean,
        'vht2_advantage': vht2_mean - flat_mean,
    }


def analyze_model_directory(model_dir: str, head_dim: int = None):
    """
    Analyze all weight tensors in a HuggingFace model directory.
    Identifies which tensors have spectral structure suitable for VHT2.
    """
    if not HAS_TORCH:
        print("ERROR: torch not installed")
        return

    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip install safetensors")
        return

    # Read config for head_dim
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        if head_dim is None:
            n_heads = config.get('num_attention_heads', 32)
            hidden = config.get('hidden_size', 4096)
            head_dim = hidden // n_heads
        print(f"Model config: hidden={config.get('hidden_size')}, "
              f"heads={config.get('num_attention_heads')}, head_dim={head_dim}")
    elif head_dim is None:
        head_dim = 128
        print(f"No config.json found, assuming head_dim={head_dim}")

    # Find safetensor files
    st_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
    if not st_files:
        print("No .safetensors files found.")
        return

    print(f"\n{'═' * 70}")
    print(f"Shannon-Prime Spectral Analysis: {model_dir}")
    print(f"{'═' * 70}")
    print(f"\n{'Tensor':<45} {'Shape':<20} {'Conc%':>6} {'VHT2':>6} {'Flat':>6} {'Δ':>7}")
    print(f"{'─'*45} {'─'*20} {'─'*6} {'─'*6} {'─'*6} {'─'*7}")

    structured_tensors = []
    unstructured_tensors = []

    for st_file in sorted(st_files):
        path = os.path.join(model_dir, st_file)
        with safe_open(path, framework='pt') as f:
            for name in f.keys():
                tensor = f.get_tensor(name)

                # Only analyze Q/K/V/O projection weights
                is_attn = any(k in name.lower() for k in
                             ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                              'self_attn', 'query', 'key', 'value',
                              'qkv_proj', 'attn.weight'])

                if not is_attn or tensor.dim() != 2:
                    continue

                result = analyze_tensor_spectrum(tensor, name, head_dim)

                if not result.get('aligned', False):
                    continue

                shape_str = str(result['shape'])
                conc = result['concentration'] * 100
                vht2 = result['vht2_corr']
                flat = result['flat_corr']
                delta = result['vht2_advantage']

                marker = ' ◀' if delta > 0.001 else ''
                print(f"  {name[:43]:<43} {shape_str:<20} {conc:5.1f}% "
                      f"{vht2:5.3f} {flat:5.3f} {delta:+6.4f}{marker}")

                if delta > 0.001:
                    structured_tensors.append(result)
                else:
                    unstructured_tensors.append(result)

    print(f"\n{'─' * 70}")
    print(f"  Tensors with VHT2 advantage (Δ > 0.001): {len(structured_tensors)}")
    print(f"  Tensors without advantage:               {len(unstructured_tensors)}")

    if structured_tensors:
        avg_advantage = sum(t['vht2_advantage'] for t in structured_tensors) / len(structured_tensors)
        avg_conc = sum(t['concentration'] for t in structured_tensors) / len(structured_tensors) * 100
        print(f"  Average VHT2 advantage:                  {avg_advantage:+.4f} correlation")
        print(f"  Average spectral concentration:           {avg_conc:.1f}%")
        print(f"\n  Recommendation: Apply VHT2 to {len(structured_tensors)} attention tensors,")
        print(f"  standard quantization to the rest.")
    else:
        print(f"\n  No tensors show significant VHT2 advantage at this head_dim.")
        print(f"  Standard quantization recommended for all weights.")

    print()


# =============================================================================
# VHT2 weight compression
# =============================================================================

def compress_tensor_vht2(tensor: 'torch.Tensor', head_dim: int,
                         band_bits: list = [5, 5, 4, 3],
                         use_mobius: bool = True) -> dict:
    """
    Compress a weight tensor using VHT2 spectral quantization.

    Returns a dict with compressed data and metadata needed for reconstruction.
    """
    t = tensor.float()
    orig_shape = t.shape
    orig_dtype = tensor.dtype

    # Reshape so last dim = head_dim
    if t.dim() == 2:
        out_dim, in_dim = t.shape
        if out_dim % head_dim == 0:
            t = t.view(-1, head_dim)
        elif in_dim % head_dim == 0:
            t = t.view(-1, head_dim)
        else:
            raise ValueError(f"Tensor shape {orig_shape} not aligned to head_dim={head_dim}")
    else:
        raise ValueError(f"Expected 2D tensor, got {t.dim()}D")

    n_vectors = t.shape[0]
    mask = MobiusMask(head_dim) if use_mobius else None
    bq = BandedQuantizer(head_dim, band_bits)

    # Process all vectors: VHT2 forward on the (n_vectors, head_dim) batch
    work = vht2(t)

    if mask is not None:
        work = mask.reorder(work)

    scales, quants = bq.quantize(work)

    return {
        'scales': [s.cpu() for s in scales],
        'quants': [q.cpu() for q in quants],
        'orig_shape': list(orig_shape),
        'orig_dtype': str(orig_dtype),
        'head_dim': head_dim,
        'band_bits': band_bits,
        'use_mobius': use_mobius,
        'n_vectors': n_vectors,
        'bytes_compressed': bq.compressed_bytes_per_vec() * n_vectors,
        'bytes_original': tensor.nelement() * tensor.element_size(),
    }


def decompress_tensor_vht2(compressed: dict) -> 'torch.Tensor':
    """Reconstruct a weight tensor from VHT2 compressed format."""
    head_dim = compressed['head_dim']
    band_bits = compressed['band_bits']
    use_mobius = compressed['use_mobius']
    orig_shape = compressed['orig_shape']

    bq = BandedQuantizer(head_dim, band_bits)
    mask = MobiusMask(head_dim) if use_mobius else None

    scales = compressed['scales']
    quants = compressed['quants']

    work = bq.dequantize(scales, quants)

    if mask is not None:
        work = mask.unreorder(work)

    # Inverse VHT2 (self-inverse, no 1/N)
    work = vht2(work)

    return work.reshape(orig_shape)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Shannon-Prime Model Compression — VHT2-aware weight quantization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a HuggingFace model directory
  python sp_compress_model.py --analyze path/to/model/

  # Analyze with specific head_dim
  python sp_compress_model.py --analyze path/to/model/ --head-dim 64

  # Compress (future — currently analysis only)
  python sp_compress_model.py path/to/model/ path/to/output/
        """
    )

    parser.add_argument('input', help='Input model directory or GGUF file')
    parser.add_argument('output', nargs='?', help='Output path')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze spectral structure without compressing')
    parser.add_argument('--head-dim', type=int, default=None,
                        help='Head dimension (auto-detected from config.json)')
    parser.add_argument('--k-bits', type=str, default='5,5,4,3',
                        help='K/Q weight bit allocation. Default: 5,5,4,3')

    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: torch not installed. Run: pip install torch")
        sys.exit(1)

    if args.analyze:
        if os.path.isdir(args.input):
            analyze_model_directory(args.input, args.head_dim)
        else:
            print(f"ERROR: --analyze expects a model directory, got: {args.input}")
            sys.exit(1)
        return

    if not args.output:
        parser.error("Output path required (unless using --analyze)")

    print("Full model compression is a research feature.")
    print("Currently supported: --analyze (spectral structure analysis)")
    print("For KV cache compression at inference time, use the shannon-prime library directly.")
    print("For RoPE frequency injection, use sp_inject_freqs.py.")


if __name__ == '__main__':
    main()
