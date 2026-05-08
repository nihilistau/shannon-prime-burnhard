# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
sp_compress.py — Phase 12: KV Cache Compression Round-Trip Engine

The real test. This script:
  1. Loads a GGUF model
  2. Runs a prompt to populate the KV cache
  3. Extracts the full KV state (K and V vectors)
  4. Applies VHT2 transform + adaptive top-K skeleton compression
  5. Reconstructs compressed KV vectors via inverse VHT2
  6. Injects the compressed cache back into the model
  7. Generates continuation tokens with both original and compressed caches
  8. Compares: perplexity, token agreement, generation quality

This answers the only question that matters:
  "Does spectral KV compression actually preserve generation quality?"

Usage:
    # Basic compression test at 30% skeleton:
    python sp_compress.py --model model.gguf --skeleton-frac 0.30

    # Test multiple compression ratios:
    python sp_compress.py --model model.gguf --skeleton-frac 0.10,0.20,0.30,0.50

    # With long prompt and custom continuation:
    python sp_compress.py --model model.gguf --prompt-file long_prompt.txt --skeleton-frac 0.30

    # Verbose with per-layer error reporting:
    python sp_compress.py --model model.gguf --skeleton-frac 0.30 --verbose
"""

import sys
import os
import argparse
import ctypes
import json
import math
import time
from typing import Dict, List, Tuple, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# VHT2 Transform (same as sp_regime_analysis.py — self-inverse)
# ─────────────────────────────────────────────────────────────────────────────

def _hartley_kernel(p: int) -> np.ndarray:
    idx = np.arange(p)
    angles = (2.0 * np.pi / p) * np.outer(idx, idx)
    return (np.cos(angles) + np.sin(angles)) / np.sqrt(p)


def _factor_small_primes(n: int) -> List[int]:
    d = n
    primes = []
    for p in [2, 3, 5, 7, 11]:
        while d % p == 0:
            primes.append(p)
            d //= p
    if d != 1:
        raise ValueError(f"dim {n} has prime factor > 11 (residue {d})")
    return primes


def vht2_forward(x: np.ndarray) -> np.ndarray:
    n = len(x)
    primes = _factor_small_primes(n)
    out = x.copy().astype(np.float64)
    stride = 1
    for p in primes:
        H = _hartley_kernel(p)
        for block_start in range(0, n, stride * p):
            for s in range(stride):
                indices = [block_start + s + i * stride for i in range(p)]
                vals = out[indices]
                out[indices] = H @ vals
        stride *= p
    return out.astype(np.float32)


def vht2_pow2(x: np.ndarray) -> np.ndarray:
    n = len(x)
    out = x.copy().astype(np.float64)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a, b = out[j], out[j + h]
                out[j] = a + b
                out[j + h] = a - b
        h *= 2
    out /= np.sqrt(n)
    return out.astype(np.float32)


def is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def vht2(x: np.ndarray) -> np.ndarray:
    if is_power_of_2(len(x)):
        return vht2_pow2(x)
    return vht2_forward(x)


SQFREE_PAD = {64: 66, 96: 110, 128: 154, 256: 330}

def sqfree_pad_dim(head_dim: int) -> int:
    if head_dim in SQFREE_PAD:
        return SQFREE_PAD[head_dim]
    n = head_dim
    while n < head_dim * 2:
        d = n
        distinct = 0
        is_sq = True
        for p in [2, 3, 5, 7, 11]:
            if d % p == 0:
                distinct += 1
                d //= p
                if d % p == 0:
                    is_sq = False
                    break
        if is_sq and d == 1 and distinct >= 3:
            return n
        n += 1
    return head_dim


def sqfree_pad_vector(x: np.ndarray, pad_dim: int) -> np.ndarray:
    hd = len(x)
    if hd >= pad_dim:
        return x[:pad_dim].copy()
    out = np.empty(pad_dim, dtype=x.dtype)
    out[:hd] = x
    out[hd:] = np.mean(x)
    return out


def sqfree_unpad_vector(padded: np.ndarray, original_dim: int) -> np.ndarray:
    """Unpad: take only the first original_dim elements."""
    return padded[:original_dim].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Top-K Skeleton
# ─────────────────────────────────────────────────────────────────────────────

def build_adaptive_skeleton(spectra: np.ndarray, k: int) -> np.ndarray:
    """Variance-ranked skeleton: top-k indices by spectral variance across samples."""
    var = np.var(spectra, axis=0)
    return np.argsort(var)[::-1][:k]


def compress_spectrum(spectrum: np.ndarray, skeleton: np.ndarray,
                      dim: int) -> np.ndarray:
    """Zero non-skeleton coefficients."""
    sparse = np.zeros(dim, dtype=np.float32)
    for idx in skeleton:
        if idx < dim:
            sparse[idx] = spectrum[idx]
    return sparse


# ─────────────────────────────────────────────────────────────────────────────
# KV Cache Extraction and Injection
# ─────────────────────────────────────────────────────────────────────────────

def _kv_clear(ctx):
    """Clear the KV cache — handles API renames across llama.cpp versions."""
    from llama_cpp import llama_cpp
    for fn_name in ['llama_kv_self_clear', 'llama_kv_cache_clear']:
        if hasattr(llama_cpp, fn_name):
            try:
                getattr(llama_cpp, fn_name)(ctx)
                return
            except Exception:
                continue
    # Last resort: try llm-level reset if we can find it
    raise RuntimeError("No KV clear function found (tried llama_kv_self_clear, llama_kv_cache_clear)")


def extract_kv_state(ctx, n_layers: int, n_kv_heads: int, head_dim: int,
                     n_tokens: int, verbose: bool = False) -> Tuple[bytes, np.ndarray, np.ndarray]:
    """Extract the full KV state as raw bytes plus parsed K and V arrays.

    Returns:
        raw_state: the original binary state (for size reference)
        k_vectors: shape (n_layers, n_kv_heads, n_tokens, head_dim) float32
        v_vectors: shape (n_layers, n_kv_heads, n_tokens, head_dim) float32
    """
    from llama_cpp import llama_cpp

    # Get per-sequence state
    get_fn = None
    size_fn = None
    set_fn = None
    for get_name, size_name in [
        ('llama_state_seq_get_data', 'llama_state_seq_get_size'),
        ('llama_kv_self_seq_get_data', 'llama_kv_self_seq_get_size'),
    ]:
        if hasattr(llama_cpp, get_name) and hasattr(llama_cpp, size_name):
            get_fn = getattr(llama_cpp, get_name)
            size_fn = getattr(llama_cpp, size_name)
            break

    if get_fn is None:
        raise RuntimeError("No per-sequence state API found")

    seq_id = 0
    state_size = size_fn(ctx, seq_id)
    if state_size == 0:
        raise RuntimeError("State size is 0 — KV cache empty")

    buf = (ctypes.c_uint8 * state_size)()
    written = get_fn(ctx, buf, state_size, seq_id)
    raw_state = bytes(buf[:written])

    if verbose:
        print(f"    KV state: {written} bytes ({written/1024/1024:.2f} MB)")

    # Parse into K and V arrays
    kv_per_layer_f32 = n_tokens * n_kv_heads * head_dim * 4
    header_overhead = 4 + (n_tokens * 8)  # n_cells + cell descriptors

    k_vectors = None
    v_vectors = None

    # Try different header sizes and data types
    for dtype_name, cell_bytes, np_dtype in [('f32', 4, np.float32), ('f16', 2, np.float16)]:
        kv_per_layer = n_tokens * n_kv_heads * head_dim * cell_bytes
        total_kv = 2 * n_layers * kv_per_layer

        for header_size in [header_overhead, 0, 4, 8, 16, 32, 64, 128, 256]:
            if header_size + total_kv > len(raw_state):
                continue
            if abs((header_size + total_kv) - len(raw_state)) > 4096:
                continue

            try:
                k_arr = np.zeros((n_layers, n_kv_heads, n_tokens, head_dim), dtype=np.float32)
                v_arr = np.zeros((n_layers, n_kv_heads, n_tokens, head_dim), dtype=np.float32)
                offset = header_size

                valid = True
                for L in range(n_layers):
                    # K data
                    k_raw = np.frombuffer(raw_state[offset:offset + kv_per_layer], dtype=np_dtype)
                    if np_dtype == np.float16:
                        k_flat = k_raw.astype(np.float32)
                    else:
                        k_flat = k_raw.copy()

                    if np.isnan(k_flat).any():
                        valid = False
                        break

                    k_arr[L] = k_flat.reshape(n_tokens, n_kv_heads, head_dim).transpose(1, 0, 2)
                    offset += kv_per_layer

                    # V data
                    v_raw = np.frombuffer(raw_state[offset:offset + kv_per_layer], dtype=np_dtype)
                    if np_dtype == np.float16:
                        v_flat = v_raw.astype(np.float32)
                    else:
                        v_flat = v_raw.copy()

                    if np.isnan(v_flat).any():
                        valid = False
                        break

                    v_arr[L] = v_flat.reshape(n_tokens, n_kv_heads, head_dim).transpose(1, 0, 2)
                    offset += kv_per_layer

                if not valid:
                    continue

                # Validation
                l0_norm = float(np.linalg.norm(k_arr[0], axis=-1).mean())
                lN_norm = float(np.linalg.norm(k_arr[-1], axis=-1).mean())
                if abs(l0_norm) < 1e-10 or abs(l0_norm - lN_norm) < 1e-6:
                    continue

                k_vectors = k_arr
                v_vectors = v_arr

                if verbose:
                    print(f"    Parsed KV: dtype={dtype_name}, header={header_size}, "
                          f"L0 K norm={l0_norm:.4f}, L{n_layers-1} K norm={lN_norm:.4f}")
                break

            except (ValueError, IndexError):
                continue

        if k_vectors is not None:
            break

    if k_vectors is None:
        raise RuntimeError(
            f"Could not parse KV state ({len(raw_state)} bytes). "
            f"Expected ~{2 * n_layers * kv_per_layer_f32 + header_overhead} bytes for f32."
        )

    return raw_state, k_vectors, v_vectors


def inject_kv_state(ctx, raw_state: bytes, k_vectors: np.ndarray, v_vectors: np.ndarray,
                    n_layers: int, n_kv_heads: int, head_dim: int, n_tokens: int,
                    verbose: bool = False) -> bool:
    """Inject modified K and V vectors back into the model's KV cache.

    Reconstructs the binary state with the new KV data and uses
    llama_state_seq_set_data to load it back.

    Returns True on success.
    """
    from llama_cpp import llama_cpp

    # Find the set API
    set_fn = None
    for set_name in ['llama_state_seq_set_data', 'llama_kv_self_seq_set_data']:
        if hasattr(llama_cpp, set_name):
            set_fn = getattr(llama_cpp, set_name)
            break

    if set_fn is None:
        raise RuntimeError("No per-sequence state set API found")

    # We need to reconstruct the binary with the same format as the original
    # Parse the header to find where KV data starts
    kv_per_layer_f32 = n_tokens * n_kv_heads * head_dim * 4
    kv_per_layer_f16 = n_tokens * n_kv_heads * head_dim * 2
    total_kv_f32 = 2 * n_layers * kv_per_layer_f32
    total_kv_f16 = 2 * n_layers * kv_per_layer_f16

    # Detect which format the original state used
    state_len = len(raw_state)
    header_size = None
    kv_per_layer = None
    use_f16 = False

    for dtype_name, kv_pl, total in [('f32', kv_per_layer_f32, total_kv_f32),
                                      ('f16', kv_per_layer_f16, total_kv_f16)]:
        header_overhead = 4 + (n_tokens * 8)
        for hs in [header_overhead, 0, 4, 8, 16, 32, 64, 128, 256]:
            if abs((hs + total) - state_len) < 4096:
                header_size = hs
                kv_per_layer = kv_pl
                use_f16 = (dtype_name == 'f16')
                break
        if header_size is not None:
            break

    if header_size is None:
        raise RuntimeError(f"Cannot determine KV state format for injection "
                          f"(state_len={state_len})")

    # Reconstruct the binary state
    new_state = bytearray(raw_state[:header_size])  # Copy header verbatim

    for L in range(n_layers):
        # K data: (n_kv_heads, n_tokens, head_dim) → (n_tokens, n_kv_heads, head_dim)
        k_layer = k_vectors[L].transpose(1, 0, 2)  # → (n_tokens, n_kv_heads, head_dim)
        k_flat = k_layer.reshape(-1)

        if use_f16:
            new_state.extend(k_flat.astype(np.float16).tobytes())
        else:
            new_state.extend(k_flat.astype(np.float32).tobytes())

        # V data
        v_layer = v_vectors[L].transpose(1, 0, 2)
        v_flat = v_layer.reshape(-1)

        if use_f16:
            new_state.extend(v_flat.astype(np.float16).tobytes())
        else:
            new_state.extend(v_flat.astype(np.float32).tobytes())

    # Append any trailing bytes from the original state
    expected_end = header_size + 2 * n_layers * kv_per_layer
    if expected_end < state_len:
        new_state.extend(raw_state[expected_end:])

    new_state = bytes(new_state)

    if verbose:
        print(f"    Injecting: {len(new_state)} bytes "
              f"(original: {state_len}, delta: {len(new_state) - state_len})")

    # Inject via the API
    seq_id = 0
    buf = (ctypes.c_uint8 * len(new_state)).from_buffer_copy(new_state)
    result = set_fn(ctx, buf, len(new_state), seq_id)

    if verbose:
        print(f"    Injection result: {result} bytes consumed")

    return result > 0


# ─────────────────────────────────────────────────────────────────────────────
# Compression Engine
# ─────────────────────────────────────────────────────────────────────────────

def compress_kv_cache(k_vectors: np.ndarray, v_vectors: np.ndarray,
                      skeleton_frac: float, use_sqfree: bool = True,
                      compress_v: bool = True,
                      verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Compress K (and optionally V) vectors via VHT2 + adaptive top-K.

    For each layer:
      1. Pad vectors to sqfree dimension if requested
      2. VHT2 forward transform all vectors
      3. Build adaptive skeleton from spectral variance
      4. Zero non-skeleton coefficients
      5. VHT2 inverse (self-inverse) to reconstruct
      6. Unpad back to original dimension

    Returns:
        k_compressed: same shape as k_vectors, with compressed data
        v_compressed: same shape as v_vectors (compressed or original copy)
        stats: per-layer compression statistics
    """
    n_layers, n_kv_heads, n_pos, head_dim = k_vectors.shape

    if use_sqfree:
        analysis_dim = sqfree_pad_dim(head_dim)
    else:
        analysis_dim = head_dim

    k = max(1, int(skeleton_frac * analysis_dim))

    if verbose:
        print(f"\n  Compression settings:")
        print(f"    head_dim={head_dim}, analysis_dim={analysis_dim}")
        print(f"    skeleton_frac={skeleton_frac}, K={k}/{analysis_dim}")
        print(f"    compress_v={compress_v}")
        print(f"    Theoretical compression: {analysis_dim/k:.1f}x spectral")

    k_compressed = np.zeros_like(k_vectors)
    v_compressed = v_vectors.copy() if not compress_v else np.zeros_like(v_vectors)

    stats = {
        'skeleton_frac': skeleton_frac,
        'skeleton_k': k,
        'analysis_dim': analysis_dim,
        'compress_v': compress_v,
        'per_layer': [],
    }

    t_total = time.time()

    for L in range(n_layers):
        t0 = time.time()

        # ── Compress K vectors ────────────────────────────────────────────
        # Collect all spectra for this layer to build the skeleton
        k_spectra = []
        for H in range(n_kv_heads):
            for pos in range(n_pos):
                vec = k_vectors[L, H, pos]
                if use_sqfree and analysis_dim != head_dim:
                    padded = sqfree_pad_vector(vec, analysis_dim)
                    spec = vht2(padded)
                else:
                    spec = vht2(vec)
                k_spectra.append(spec)

        k_spectra = np.array(k_spectra)
        k_skeleton = build_adaptive_skeleton(k_spectra, k)

        # Reconstruct K
        k_errors = []
        idx = 0
        for H in range(n_kv_heads):
            for pos in range(n_pos):
                spec = k_spectra[idx]
                compressed_spec = compress_spectrum(spec, k_skeleton, analysis_dim)
                reconstructed = vht2(compressed_spec)  # Self-inverse

                if use_sqfree and analysis_dim != head_dim:
                    k_compressed[L, H, pos] = sqfree_unpad_vector(reconstructed, head_dim)
                else:
                    k_compressed[L, H, pos] = reconstructed

                # Track error
                orig = k_vectors[L, H, pos]
                recon = k_compressed[L, H, pos]
                orig_norm = np.linalg.norm(orig)
                if orig_norm > 1e-12:
                    k_errors.append(float(np.linalg.norm(orig - recon) / orig_norm))

                idx += 1

        # ── Compress V vectors (if requested) ─────────────────────────────
        v_errors = []
        if compress_v:
            v_spectra = []
            for H in range(n_kv_heads):
                for pos in range(n_pos):
                    vec = v_vectors[L, H, pos]
                    if use_sqfree and analysis_dim != head_dim:
                        padded = sqfree_pad_vector(vec, analysis_dim)
                        spec = vht2(padded)
                    else:
                        spec = vht2(vec)
                    v_spectra.append(spec)

            v_spectra = np.array(v_spectra)
            v_skeleton = build_adaptive_skeleton(v_spectra, k)

            idx = 0
            for H in range(n_kv_heads):
                for pos in range(n_pos):
                    spec = v_spectra[idx]
                    compressed_spec = compress_spectrum(spec, v_skeleton, analysis_dim)
                    reconstructed = vht2(compressed_spec)

                    if use_sqfree and analysis_dim != head_dim:
                        v_compressed[L, H, pos] = sqfree_unpad_vector(reconstructed, head_dim)
                    else:
                        v_compressed[L, H, pos] = reconstructed

                    orig = v_vectors[L, H, pos]
                    recon = v_compressed[L, H, pos]
                    orig_norm = np.linalg.norm(orig)
                    if orig_norm > 1e-12:
                        v_errors.append(float(np.linalg.norm(orig - recon) / orig_norm))

                    idx += 1

        elapsed = time.time() - t0
        layer_stats = {
            'layer': L,
            'k_mean_error': float(np.mean(k_errors)) if k_errors else 0.0,
            'k_max_error': float(np.max(k_errors)) if k_errors else 0.0,
            'v_mean_error': float(np.mean(v_errors)) if v_errors else 0.0,
            'time': elapsed,
        }
        stats['per_layer'].append(layer_stats)

        if verbose:
            v_str = f", V err={layer_stats['v_mean_error']:.4f}" if compress_v else ""
            print(f"    L{L:02d}: K err={layer_stats['k_mean_error']:.4f} "
                  f"(max={layer_stats['k_max_error']:.4f}){v_str}  [{elapsed:.1f}s]")

    stats['total_time'] = time.time() - t_total
    stats['mean_k_error'] = float(np.mean([s['k_mean_error'] for s in stats['per_layer']]))
    if compress_v:
        stats['mean_v_error'] = float(np.mean([s['v_mean_error'] for s in stats['per_layer']]))

    return k_compressed, v_compressed, stats


# ─────────────────────────────────────────────────────────────────────────────
# Generation and Perplexity
# ─────────────────────────────────────────────────────────────────────────────

def generate_tokens(llm, n_tokens: int = 64, temperature: float = 0.0) -> Tuple[List[int], List[float]]:
    """Generate tokens from the current KV cache state.

    Returns:
        tokens: list of generated token IDs
        logprobs: list of log-probabilities for each generated token
    """
    from llama_cpp import llama_cpp

    tokens = []
    logprobs = []

    for i in range(n_tokens):
        # Get logits for the last position
        output = llm.create_completion(
            "",  # Empty — we're continuing from cached state
            max_tokens=1,
            temperature=temperature,
            logprobs=1,
        )

        if output and 'choices' in output and output['choices']:
            choice = output['choices'][0]
            token_text = choice.get('text', '')

            # Extract logprob
            lp = None
            if 'logprobs' in choice and choice['logprobs']:
                lp_data = choice['logprobs']
                if 'token_logprobs' in lp_data and lp_data['token_logprobs']:
                    lp = lp_data['token_logprobs'][0]

            tokens.append(token_text)
            logprobs.append(lp if lp is not None else 0.0)
        else:
            break

    return tokens, logprobs


def _get_vocab_size(llm, verbose: bool = False) -> int:
    """Detect vocabulary size through multiple API paths."""
    from llama_cpp import llama_cpp

    model = llm._model.model

    # Path 1: llama_model_n_vocab(model) — older API
    for fn_name in ['llama_model_n_vocab']:
        if hasattr(llama_cpp, fn_name):
            try:
                n = getattr(llama_cpp, fn_name)(model)
                if n and n > 0:
                    return n
            except Exception:
                pass

    # Path 2: llama_model_get_vocab(model) → llama_vocab_n_tokens(vocab) — newer API
    try:
        if hasattr(llama_cpp, 'llama_model_get_vocab'):
            vocab = llama_cpp.llama_model_get_vocab(model)
            if hasattr(llama_cpp, 'llama_vocab_n_tokens'):
                n = llama_cpp.llama_vocab_n_tokens(vocab)
                if n and n > 0:
                    return n
    except Exception:
        pass

    # Path 3: high-level Python method
    if hasattr(llm, 'n_vocab'):
        try:
            n = llm.n_vocab()
            if n and n > 0:
                return n
        except Exception:
            pass

    # Path 4: scores buffer shape
    if hasattr(llm, '_scores') and llm._scores is not None:
        n = llm._scores.shape[-1]
        if n and n > 0:
            return n

    # Absolute fallback
    if verbose:
        print(f"    WARNING: Could not determine vocab size, using 32064")
    return 32064


def _compute_perplexity_lowlevel(llm, eval_tokens: List[int],
                                  n_vocab: int, verbose: bool = False) -> List[float]:
    """Low-level perplexity via C API: llama_batch_get_one + llama_decode + llama_get_logits."""
    from llama_cpp import llama_cpp

    ctx = llm._ctx.ctx
    n_eval = len(eval_tokens)
    per_token_lp = []

    # Check that the C functions we need actually exist
    for fn_name in ['llama_batch_get_one', 'llama_decode', 'llama_get_logits']:
        if not hasattr(llama_cpp, fn_name):
            raise AttributeError(f"llama_cpp.{fn_name} not found")

    for i in range(n_eval):
        token = eval_tokens[i]

        batch = llama_cpp.llama_batch_get_one(
            (ctypes.c_int32 * 1)(token),
            1
        )

        ret = llama_cpp.llama_decode(ctx, batch)
        if ret != 0:
            if verbose:
                print(f"    Decode error at eval token {i}: ret={ret}")
            break

        if i < n_eval - 1:
            logits_ptr = llama_cpp.llama_get_logits(ctx)
            if logits_ptr is None:
                continue

            logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy()

            # Log-softmax
            max_logit = float(np.max(logits))
            shifted = logits - max_logit
            log_sum_exp = max_logit + float(np.log(np.sum(np.exp(shifted))))
            log_probs = logits - log_sum_exp

            next_token = eval_tokens[i + 1]
            if 0 <= next_token < n_vocab:
                per_token_lp.append(float(log_probs[next_token]))

    return per_token_lp


def _compute_perplexity_highlevel(llm, eval_tokens: List[int],
                                   n_vocab: int, verbose: bool = False) -> List[float]:
    """High-level perplexity via llm.eval() + internal logits buffer.

    Fallback for when the C API functions have been renamed.
    """
    n_eval = len(eval_tokens)
    per_token_lp = []

    for i in range(n_eval):
        # Feed one token at a time through the Python API
        llm.eval([eval_tokens[i]])

        if i < n_eval - 1:
            # Try to get logits from the internal buffer
            logits = None

            # Method 1: llm._scores — stored after eval in some versions
            if hasattr(llm, '_scores') and llm._scores is not None:
                try:
                    # _scores shape is typically (n_tokens_processed, n_vocab)
                    # After single-token eval, last row is what we want
                    if llm._scores.ndim == 2 and llm._scores.shape[-1] >= n_vocab:
                        logits = llm._scores[-1, :n_vocab].copy()
                    elif llm._scores.ndim == 1 and len(llm._scores) >= n_vocab:
                        logits = llm._scores[:n_vocab].copy()
                except Exception:
                    pass

            # Method 2: direct C logits pointer as last resort
            if logits is None:
                try:
                    from llama_cpp import llama_cpp
                    ctx = llm._ctx.ctx
                    # Try all known logits function names
                    for fn_name in ['llama_get_logits', 'llama_get_logits_ith']:
                        if hasattr(llama_cpp, fn_name):
                            try:
                                if fn_name == 'llama_get_logits_ith':
                                    logits_ptr = getattr(llama_cpp, fn_name)(ctx, -1)
                                else:
                                    logits_ptr = getattr(llama_cpp, fn_name)(ctx)
                                if logits_ptr is not None:
                                    logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy()
                                    break
                            except Exception:
                                continue
                except Exception:
                    pass

            if logits is None:
                continue

            # Log-softmax
            max_logit = float(np.max(logits))
            shifted = logits - max_logit
            log_sum_exp = max_logit + float(np.log(np.sum(np.exp(shifted))))
            log_probs = logits - log_sum_exp

            next_token = eval_tokens[i + 1]
            if 0 <= next_token < n_vocab:
                per_token_lp.append(float(log_probs[next_token]))

    return per_token_lp


def compute_perplexity(llm, eval_text: str, verbose: bool = False) -> Tuple[float, List[float]]:
    """Compute perplexity on evaluation text using the model's current state.

    Tries two approaches:
      1. Low-level C API (llama_batch_get_one + llama_decode + llama_get_logits)
      2. High-level Python API (llm.eval + internal logits buffer)

    Returns:
        perplexity: exp(-mean(log_probs))
        per_token_logprobs: list of per-token log-probabilities
    """
    eval_tokens = llm.tokenize(eval_text.encode('utf-8'), add_bos=False)
    n_eval = len(eval_tokens)

    if verbose:
        print(f"    Eval tokens: {n_eval}")

    if n_eval < 2:
        return 1.0, []

    n_vocab = _get_vocab_size(llm, verbose)
    if verbose:
        print(f"    Vocabulary size: {n_vocab}")

    per_token_lp = []

    # ── Attempt 1: Low-level C API ──
    try:
        if verbose:
            print(f"    Trying low-level C API path...")
        per_token_lp = _compute_perplexity_lowlevel(llm, eval_tokens, n_vocab, verbose)
        if per_token_lp:
            if verbose:
                print(f"    Low-level path succeeded: {len(per_token_lp)} logprobs")
    except Exception as e:
        if verbose:
            print(f"    Low-level path failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Attempt 2: High-level Python API ──
    if not per_token_lp:
        try:
            if verbose:
                print(f"    Trying high-level Python API path...")
            per_token_lp = _compute_perplexity_highlevel(llm, eval_tokens, n_vocab, verbose)
            if per_token_lp:
                if verbose:
                    print(f"    High-level path succeeded: {len(per_token_lp)} logprobs")
        except Exception as e:
            if verbose:
                print(f"    High-level path failed: {e}")
                import traceback
                traceback.print_exc()

    if not per_token_lp:
        if verbose:
            print(f"    WARNING: No logprobs collected from either path")
        return float('inf'), []

    mean_lp = float(np.mean(per_token_lp))
    ppl = float(np.exp(-mean_lp))

    if verbose:
        print(f"    Collected {len(per_token_lp)} logprobs, mean={mean_lp:.4f}")

    return ppl, per_token_lp


# ─────────────────────────────────────────────────────────────────────────────
# Main: The Full Round-Trip
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROMPT = (
    "The fundamental theorem of arithmetic states that every integer greater "
    "than one can be uniquely represented as a product of prime numbers. For "
    "example, 30 = 2 × 3 × 5, and 1001 = 7 × 11 × 13. This factorization "
    "is the basis of the Vilenkin group structure on the integers."
)

DEFAULT_EVAL = (
    "The relationship between prime factorization and algebraic structures "
    "extends beyond the integers. In ring theory, unique factorization domains "
    "generalize this property to polynomial rings and number fields. The "
    "Fundamental Theorem of Algebra guarantees that every non-constant "
    "polynomial with complex coefficients has at least one complex root, "
    "which combined with the factor theorem means every polynomial of degree "
    "n has exactly n roots counted with multiplicity. This connects number "
    "theory with complex analysis through the zeros of the Riemann zeta "
    "function, whose distribution encodes deep information about the primes."
)


def run_compression_test(model_path: str, prompt: str, eval_text: str,
                         skeleton_fracs: List[float],
                         n_gpu_layers: int = 0, n_ctx: int = 2048,
                         use_sqfree: bool = True, compress_v: bool = True,
                         verbose: bool = False) -> Dict:
    """Run the full compression round-trip test.

    For each skeleton fraction:
      1. Load model fresh (clean state)
      2. Process prompt → baseline KV cache
      3. Measure baseline perplexity on eval text
      4. Extract KV state
      5. Compress via VHT2 + adaptive top-K
      6. Inject compressed cache
      7. Measure compressed perplexity on eval text
      8. Compare
    """
    from llama_cpp import Llama, llama_cpp

    results = {
        'model': os.path.basename(model_path),
        'prompt_length': len(prompt),
        'eval_length': len(eval_text),
        'use_sqfree': use_sqfree,
        'compress_v': compress_v,
        'tests': [],
    }

    for frac in skeleton_fracs:
        print(f"\n{'='*72}")
        print(f"  COMPRESSION TEST: skeleton_frac={frac}")
        print(f"{'='*72}")

        # ── Load model fresh ──────────────────────────────────────────────
        print(f"\n  Loading model...")
        t0 = time.time()
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            type_k=0,  # F32 KV cache
            type_v=0,
        )
        model = llm._model.model
        n_layers = llama_cpp.llama_model_n_layer(model)
        n_heads = llama_cpp.llama_model_n_head(model)
        n_kv_heads = llama_cpp.llama_model_n_head_kv(model)
        n_embd = llama_cpp.llama_model_n_embd(model)
        head_dim = n_embd // n_heads
        print(f"  Loaded in {time.time()-t0:.1f}s: {n_layers}L, {n_heads}H "
              f"({n_kv_heads} KV), dim={head_dim}")

        # ── Process prompt ────────────────────────────────────────────────
        tokens = llm.tokenize(prompt.encode('utf-8'), add_bos=True)
        n_tokens = len(tokens)
        print(f"  Prompt: {n_tokens} tokens")

        print(f"  Forward pass (prompt)...")
        t0 = time.time()
        llm.eval(tokens)
        print(f"  Done in {time.time()-t0:.1f}s")

        ctx = llm._ctx.ctx

        # ── Baseline perplexity ───────────────────────────────────────────
        print(f"\n  Computing baseline perplexity...")
        baseline_ppl, baseline_lp = compute_perplexity(llm, eval_text, verbose)
        print(f"  Baseline PPL: {baseline_ppl:.4f}")
        baseline_n_eval = len(baseline_lp)

        # ── Extract KV state ──────────────────────────────────────────────
        # We need to reload and re-eval because perplexity eval consumed
        # additional KV slots. Reset and re-process just the prompt.
        print(f"\n  Reloading for compression...")

        # Clear and re-eval
        _kv_clear(ctx)
        llm.eval(tokens)

        print(f"  Extracting KV state...")
        raw_state, k_vectors, v_vectors = extract_kv_state(
            ctx, n_layers, n_kv_heads, head_dim, n_tokens, verbose
        )
        print(f"  K shape: {k_vectors.shape}, V shape: {v_vectors.shape}")

        # ── Compress ──────────────────────────────────────────────────────
        print(f"\n  Compressing at skeleton_frac={frac}...")
        k_comp, v_comp, comp_stats = compress_kv_cache(
            k_vectors, v_vectors, frac,
            use_sqfree=use_sqfree,
            compress_v=compress_v,
            verbose=verbose,
        )
        print(f"  Mean K error: {comp_stats['mean_k_error']:.4f}")
        if compress_v:
            print(f"  Mean V error: {comp_stats['mean_v_error']:.4f}")
        print(f"  Compression time: {comp_stats['total_time']:.1f}s")

        # ── Inject compressed cache ───────────────────────────────────────
        print(f"\n  Injecting compressed KV cache...")
        _kv_clear(ctx)

        success = inject_kv_state(
            ctx, raw_state, k_comp, v_comp,
            n_layers, n_kv_heads, head_dim, n_tokens, verbose
        )

        if not success:
            print(f"  ERROR: KV injection failed!")
            results['tests'].append({
                'skeleton_frac': frac,
                'error': 'injection_failed',
            })
            del llm
            continue

        print(f"  Injection successful")

        # ── Compressed perplexity ─────────────────────────────────────────
        print(f"\n  Computing compressed perplexity...")
        compressed_ppl, compressed_lp = compute_perplexity(llm, eval_text, verbose)
        print(f"  Compressed PPL: {compressed_ppl:.4f}")

        # ── Compare ──────────────────────────────────────────────────────
        ppl_ratio = compressed_ppl / baseline_ppl if baseline_ppl > 0 else float('inf')
        ppl_increase_pct = (ppl_ratio - 1.0) * 100

        # Per-token logprob comparison
        n_compare = min(len(baseline_lp), len(compressed_lp))
        if n_compare > 0:
            lp_deltas = [compressed_lp[i] - baseline_lp[i] for i in range(n_compare)]
            mean_lp_delta = float(np.mean(lp_deltas))
            max_lp_drop = float(np.min(lp_deltas))  # Most negative = worst degradation
        else:
            mean_lp_delta = 0.0
            max_lp_drop = 0.0

        test_result = {
            'skeleton_frac': frac,
            'skeleton_k': comp_stats['skeleton_k'],
            'analysis_dim': comp_stats['analysis_dim'],
            'compression_ratio': comp_stats['analysis_dim'] / comp_stats['skeleton_k'],
            'baseline_ppl': baseline_ppl,
            'compressed_ppl': compressed_ppl,
            'ppl_ratio': ppl_ratio,
            'ppl_increase_pct': ppl_increase_pct,
            'mean_k_error': comp_stats['mean_k_error'],
            'mean_v_error': comp_stats.get('mean_v_error', 0.0),
            'mean_logprob_delta': mean_lp_delta,
            'max_logprob_drop': max_lp_drop,
            'n_eval_tokens': n_compare,
            'compression_time': comp_stats['total_time'],
            'per_layer_k_error': [s['k_mean_error'] for s in comp_stats['per_layer']],
        }
        results['tests'].append(test_result)

        print(f"\n  ────────────────────────────────────────")
        print(f"  RESULT: skeleton={frac} ({comp_stats['analysis_dim']/comp_stats['skeleton_k']:.1f}x)")
        print(f"    Baseline PPL:   {baseline_ppl:.4f}")
        print(f"    Compressed PPL: {compressed_ppl:.4f}")
        print(f"    PPL increase:   {ppl_increase_pct:+.2f}%")
        print(f"    Mean logprob Δ: {mean_lp_delta:+.4f}")
        print(f"    Max logprob ↓:  {max_lp_drop:+.4f}")
        print(f"  ────────────────────────────────────────")

        del llm

    # ── Final Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  COMPRESSION SUMMARY: {results['model']}")
    print(f"{'='*72}")
    print(f"  {'Skel%':>6} {'Ratio':>6} {'Base PPL':>10} {'Comp PPL':>10} {'PPL Δ%':>8} {'K err':>8} {'V err':>8}")
    print(f"  {'─'*6} {'─'*6} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    for t in results['tests']:
        if 'error' in t:
            print(f"  {t['skeleton_frac']:>5.0%}  FAILED: {t['error']}")
        else:
            print(f"  {t['skeleton_frac']:>5.0%} {t['compression_ratio']:>5.1f}x "
                  f"{t['baseline_ppl']:>10.4f} {t['compressed_ppl']:>10.4f} "
                  f"{t['ppl_increase_pct']:>+7.2f}% "
                  f"{t['mean_k_error']:>7.4f} {t['mean_v_error']:>7.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 12: KV Cache Compression Round-Trip Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--model', type=str, required=True,
                        help='GGUF model path')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Prompt text')
    parser.add_argument('--prompt-file', type=str, default=None,
                        help='Read prompt from file')
    parser.add_argument('--eval-text', type=str, default=None,
                        help='Text to evaluate perplexity on')
    parser.add_argument('--eval-file', type=str, default=None,
                        help='Read eval text from file')
    parser.add_argument('--skeleton-frac', type=str, default='0.30',
                        help='Skeleton fraction(s), comma-separated (default: 0.30)')
    parser.add_argument('--no-sqfree', action='store_true',
                        help='Disable sqfree padding')
    parser.add_argument('--k-only', action='store_true',
                        help='Only compress K vectors, leave V untouched')
    parser.add_argument('--n-gpu-layers', type=int, default=0,
                        help='GPU layers (0=CPU)')
    parser.add_argument('--n-ctx', type=int, default=2048,
                        help='Context size')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save results as JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Resolve prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPT

    # Resolve eval text
    if args.eval_file:
        with open(args.eval_file) as f:
            eval_text = f.read().strip()
    elif args.eval_text:
        eval_text = args.eval_text
    else:
        eval_text = DEFAULT_EVAL

    # Parse skeleton fractions
    skeleton_fracs = [float(x.strip()) for x in args.skeleton_frac.split(',')]

    print(f"Shannon-Prime Phase 12: KV Cache Compression Engine")
    print(f"  Model: {args.model}")
    print(f"  Prompt: {len(prompt)} chars")
    print(f"  Eval text: {len(eval_text)} chars")
    print(f"  Skeleton fractions: {skeleton_fracs}")
    print(f"  Sqfree: {not args.no_sqfree}")
    print(f"  Compress V: {not args.k_only}")

    results = run_compression_test(
        model_path=args.model,
        prompt=prompt,
        eval_text=eval_text,
        skeleton_fracs=skeleton_fracs,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        use_sqfree=not args.no_sqfree,
        compress_v=not args.k_only,
        verbose=args.verbose,
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {args.output}")


if __name__ == '__main__':
    main()
