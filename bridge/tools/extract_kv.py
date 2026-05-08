# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
extract_kv.py — Extract K vectors from models for Prime Chord analysis.

Supports:
  - GGUF models via llama-cpp-python (recommended for local models)
  - HuggingFace models via transformers + PyTorch

Runs a prompt through the model, captures K projections (post-RoPE),
and writes them as .npz for sp_chord_diagnostic.py.

Output format (consumed by sp_chord_diagnostic.py):
  .npz with key "k_vectors" → shape (n_layers, n_heads, n_positions, head_dim)
  Plus metadata: model_name, n_layers, n_heads, head_dim, n_positions

Usage:
    # GGUF model (auto-detected by .gguf extension):
    python extract_kv.py --model path/to/model.gguf --output kv.npz

    # GGUF with GPU offload:
    python extract_kv.py --model path/to/model.gguf --n-gpu-layers 99 --output kv.npz

    # HuggingFace model:
    python extract_kv.py --model Qwen/Qwen2-0.5B --output kv.npz --device cuda

    # Then run the diagnostic:
    python sp_chord_diagnostic.py --input kv.npz --sqfree --plot
"""

import sys
import os
import argparse
import time
import json
import ctypes
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Default calibration prompt — covers diverse syntax to activate many heads
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROMPT = (
    "The fundamental theorem of arithmetic states that every integer greater "
    "than one can be uniquely represented as a product of prime numbers. For "
    "example, 30 = 2 × 3 × 5, and 1001 = 7 × 11 × 13. This factorization "
    "is the basis of the Vilenkin group structure on the integers, where each "
    "prime p contributes a cyclic factor Z/pZ to the infinite product group. "
    "In a transformer's attention mechanism, the key vectors at each position "
    "encode both syntactic structure (word order, dependencies) and semantic "
    "content (meaning, topic). When we apply the Vilenkin-Hartley Transform "
    "to these vectors, we decompose the positional encoding into prime-frequency "
    "components on the 5-torus T^5 = S^1(2) × S^1(3) × S^1(5) × S^1(7) × S^1(11). "
    "The hypothesis is that individual attention heads specialize on specific "
    "prime 'chords' — subsets of these frequencies that they use to track "
    "linguistic features at different scales."
)


# ─────────────────────────────────────────────────────────────────────────────
# Layer/Head range parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_range(spec: str, maximum: int) -> List[int]:
    """Parse a range spec like '0-7', '0,2,4', '0-3,8-11' into a list of ints."""
    if spec is None:
        return list(range(maximum))
    result = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            lo, hi = int(lo), int(hi)
            result.extend(range(lo, min(hi + 1, maximum)))
        else:
            idx = int(part)
            if idx < maximum:
                result.append(idx)
    return sorted(set(result))


# ─────────────────────────────────────────────────────────────────────────────
# GGUF extraction via llama-cpp-python
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_gguf(model_path: str, prompt: str,
                      n_gpu_layers: int = 0,
                      n_ctx: int = 2048,
                      max_length: Optional[int] = None,
                      verbose: bool = False) -> Tuple[np.ndarray, dict]:
    """Extract K vectors from a GGUF model using llama-cpp-python.

    Uses the low-level llama.cpp API to access KV cache cells directly
    after processing the prompt.

    Returns:
        k_vectors: shape (n_layers, n_kv_heads, n_positions, head_dim)
        metadata: dict with model info
    """
    try:
        from llama_cpp import Llama, llama_cpp
    except ImportError:
        print("ERROR: llama-cpp-python is required for GGUF models.")
        print("Install with: python -m pip install llama-cpp-python")
        print("For GPU:  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --force-reinstall --no-cache-dir")
        sys.exit(1)

    print(f"  Loading GGUF: {os.path.basename(model_path)}")
    t0 = time.time()

    # Load model with F32 KV cache for clean extraction
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
        type_k=0,  # GGML_TYPE_F32
        type_v=0,  # GGML_TYPE_F32
    )

    # Get model metadata via the low-level C API
    model = llm._model.model
    n_layers = llama_cpp.llama_model_n_layer(model)
    n_heads = llama_cpp.llama_model_n_head(model)
    n_kv_heads = llama_cpp.llama_model_n_head_kv(model)
    n_embd = llama_cpp.llama_model_n_embd(model)
    head_dim = n_embd // n_heads

    dt_load = time.time() - t0
    print(f"  Model loaded in {dt_load:.1f}s")
    print(f"  Layers: {n_layers}, Heads: {n_heads}, KV heads: {n_kv_heads}, Head dim: {head_dim}")

    # Tokenize the prompt
    tokens = llm.tokenize(prompt.encode('utf-8'), add_bos=True)
    n_tokens = len(tokens)
    if max_length and n_tokens > max_length:
        tokens = tokens[:max_length]
        n_tokens = max_length
    print(f"  Prompt tokenized to {n_tokens} tokens")

    # Process all tokens in a single batch to populate the KV cache
    print(f"  Running forward pass...")
    t0 = time.time()
    llm.eval(tokens)
    dt_eval = time.time() - t0
    print(f"  Forward pass complete in {dt_eval:.1f}s")

    # Extract K vectors from the KV cache
    # The KV cache in llama.cpp stores K and V contiguously.
    # We use llama_kv_self_seq_data_get() or direct memory access.
    print(f"  Extracting K vectors from KV cache...")

    ctx = llm._ctx.ctx

    # Try the modern API first (llama.cpp recent versions)
    k_vectors = _extract_kv_cache_direct(llm, ctx, model, n_layers, n_kv_heads, head_dim, n_tokens, verbose)

    if k_vectors is None:
        raise RuntimeError("Could not extract K vectors from llama.cpp KV cache")

    metadata = {
        'model_name': os.path.basename(model_path),
        'model_path': model_path,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_kv_heads': n_kv_heads,
        'head_dim': head_dim,
        'n_positions': n_tokens,
        'n_embd': n_embd,
        'method': 'gguf',
        'load_time': dt_load,
        'eval_time': dt_eval,
    }

    return k_vectors, metadata


def _extract_kv_cache_direct(llm, ctx, model, n_layers, n_kv_heads, head_dim, n_tokens, verbose):
    """Extract K vectors by reading the KV cache via llama.cpp C API.

    llama.cpp stores the KV cache as contiguous memory. The layout depends
    on the backend but for CPU/CUDA the K cache for each layer is stored as
    a tensor of shape (n_kv_heads, n_ctx, head_dim) or transposed.

    We use llama_kv_self_get to read it out.
    """
    from llama_cpp import llama_cpp

    # Approach: use llama_kv_self_get if available (llama.cpp >= b5000+)
    # This copies K/V data for a specific sequence into a provided buffer.

    # First check what API is available
    has_kv_get = hasattr(llama_cpp, 'llama_kv_self_get')
    has_kv_cache_view = hasattr(llama_cpp, 'llama_kv_cache_view_init')

    if verbose:
        print(f"    API check: llama_kv_self_get={has_kv_get}, kv_cache_view={has_kv_cache_view}")

    # Method: Read KV cache via the save/load state API
    # llama_state_seq_get_data copies the KV state for a sequence to a buffer.
    # We can parse the binary format to extract K vectors.

    # Alternative: Just use the Python-level token evaluation and extract via
    # repeated single-token evals while capturing internal state.

    # Most reliable method: use llama_kv_self_seq_get_data to dump KV for seq 0
    # The format is documented in llama.cpp:
    #   For each layer: [K data (n_tokens * n_kv_heads * head_dim * sizeof(float))]
    #                   [V data (same)]

    # Try using the state API
    try:
        return _extract_via_state_api(llm, ctx, n_layers, n_kv_heads, head_dim, n_tokens, verbose)
    except Exception as e:
        if verbose:
            print(f"    State API failed: {e}")

    # Fallback: compute K vectors manually from weights
    print("  Falling back to weight-based K extraction...")
    return _extract_via_weights(llm, model, n_layers, n_kv_heads, head_dim, n_tokens, verbose)


def _extract_via_state_api(llm, ctx, n_layers, n_kv_heads, head_dim, n_tokens, verbose):
    """Extract KV cache using llama_state_seq_get_data / llama_kv_self_seq_get_data."""
    from llama_cpp import llama_cpp

    # Determine which API function to use
    get_fn = None
    size_fn = None

    # Try modern API names (llama.cpp keeps renaming these...)
    for get_name, size_name in [
        ('llama_state_seq_get_data', 'llama_state_seq_get_size'),
        ('llama_kv_self_seq_get_data', 'llama_kv_self_seq_get_size'),
    ]:
        if hasattr(llama_cpp, get_name) and hasattr(llama_cpp, size_name):
            get_fn = getattr(llama_cpp, get_name)
            size_fn = getattr(llama_cpp, size_name)
            if verbose:
                print(f"    Using {get_name}")
            break

    if get_fn is None:
        raise RuntimeError("No state seq get API found")

    # Get size needed
    seq_id = 0
    state_size = size_fn(ctx, seq_id)
    if verbose:
        print(f"    State size for seq {seq_id}: {state_size} bytes")

    if state_size == 0:
        raise RuntimeError("State size is 0 — KV cache may be empty")

    # Allocate buffer and get data
    buf = (ctypes.c_uint8 * state_size)()
    written = get_fn(ctx, buf, state_size, seq_id)
    if verbose:
        print(f"    Got {written} bytes")

    # Parse the binary format
    # The format from llama.cpp (as of recent versions):
    # Header: cell_count (uint32), then for each layer K data then V data
    # K data per layer: n_tokens * n_kv_heads * head_dim values (f32 or f16)
    data = bytes(buf[:written])

    # Try to parse — the exact format varies by llama.cpp version
    # Expected total K data: n_layers * n_tokens * n_kv_heads * head_dim * 4 bytes (f32)
    expected_k_bytes = n_layers * n_tokens * n_kv_heads * head_dim * 4
    expected_kv_bytes = expected_k_bytes * 2  # K + V

    if verbose:
        print(f"    Expected KV bytes: {expected_kv_bytes}, got total: {len(data)}")
        print(f"    Data header (first 32 bytes): {data[:32].hex()}")

    # Try different parsing strategies
    k_vectors = _parse_kv_state(data, n_layers, n_kv_heads, head_dim, n_tokens, verbose)
    return k_vectors


def _parse_kv_state(data: bytes, n_layers: int, n_kv_heads: int, head_dim: int,
                    n_tokens: int, verbose: bool) -> Optional[np.ndarray]:
    """Parse the binary KV state dump into K vectors.

    The binary format from llama_state_seq_get_data varies by version.
    We try multiple parsing strategies.
    """
    # Strategy 1: Skip small header, then alternating K/V per layer as f32
    # Common header: n_cells (4 bytes) + cell metadata (variable)
    # Each cell: pos (4 bytes) + seq_mask (4 bytes) = 8 bytes per cell
    header_overhead = 4 + (n_tokens * 8)  # n_cells + cell descriptors
    kv_data_per_layer = n_tokens * n_kv_heads * head_dim * 4  # f32

    for header_size in [header_overhead, 0, 4, 8, 16, 32]:
        remaining = len(data) - header_size
        needed = 2 * n_layers * kv_data_per_layer  # K + V for all layers

        if remaining >= needed and remaining < needed + 1024:
            if verbose:
                print(f"    Trying header_size={header_size}, remaining={remaining}, needed={needed}")
            try:
                k_vectors = np.zeros((n_layers, n_kv_heads, n_tokens, head_dim), dtype=np.float32)
                offset = header_size
                for L in range(n_layers):
                    # K data for this layer
                    k_bytes = data[offset:offset + kv_data_per_layer]
                    k_flat = np.frombuffer(k_bytes, dtype=np.float32)
                    k_vectors[L] = k_flat.reshape(n_tokens, n_kv_heads, head_dim).transpose(1, 0, 2)
                    offset += kv_data_per_layer
                    # Skip V data
                    offset += kv_data_per_layer

                # Sanity check: K vectors should not be all zeros or NaN
                if not np.isnan(k_vectors).any() and np.abs(k_vectors).max() > 1e-6:
                    if verbose:
                        print(f"    Parse successful with header_size={header_size}")
                        print(f"    K norm range: {np.linalg.norm(k_vectors[0], axis=-1).mean():.4f}")
                    return k_vectors
            except (ValueError, IndexError):
                continue

    # Strategy 2: K/V might be stored as f16
    kv_data_per_layer_f16 = n_tokens * n_kv_heads * head_dim * 2

    for header_size in [header_overhead, 0, 4, 8, 16, 32]:
        remaining = len(data) - header_size
        needed = 2 * n_layers * kv_data_per_layer_f16

        if remaining >= needed and remaining < needed + 1024:
            if verbose:
                print(f"    Trying f16 with header_size={header_size}")
            try:
                k_vectors = np.zeros((n_layers, n_kv_heads, n_tokens, head_dim), dtype=np.float32)
                offset = header_size
                for L in range(n_layers):
                    k_bytes = data[offset:offset + kv_data_per_layer_f16]
                    k_flat = np.frombuffer(k_bytes, dtype=np.float16).astype(np.float32)
                    k_vectors[L] = k_flat.reshape(n_tokens, n_kv_heads, head_dim).transpose(1, 0, 2)
                    offset += kv_data_per_layer_f16
                    offset += kv_data_per_layer_f16

                if not np.isnan(k_vectors).any() and np.abs(k_vectors).max() > 1e-6:
                    if verbose:
                        print(f"    Parse successful (f16) with header_size={header_size}")
                    return k_vectors
            except (ValueError, IndexError):
                continue

    if verbose:
        print(f"    Could not parse KV state. Data length: {len(data)}")
        print(f"    Expected f32 KV: {2 * n_layers * kv_data_per_layer + header_overhead}")
        print(f"    Expected f16 KV: {2 * n_layers * kv_data_per_layer_f16 + header_overhead}")

    return None


def _extract_via_weights(llm, model, n_layers, n_kv_heads, head_dim, n_tokens, verbose):
    """Fallback: extract K vectors by manually computing K = W_K × embeddings.

    This reads the K projection weights from the GGUF and the token embeddings,
    then computes K vectors. Note: this does NOT include RoPE, so the K vectors
    will differ from the actual cached ones. Still useful for spectral analysis
    since RoPE is a rotation that preserves the prime-frequency structure.
    """
    # This is complex and model-architecture-dependent.
    # For now, raise an error pointing users to the HF path.
    print("  Weight-based extraction not yet implemented for GGUF.")
    print("  Workaround: use the HuggingFace version of this model instead:")
    print("    python extract_kv.py --model <hf_model_name> --output kv.npz")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Alternative GGUF approach: high-level eval + cache dump
# ─────────────────────────────────────────────────────────────────────────────

def extract_gguf_simple(model_path: str, prompt: str,
                        n_gpu_layers: int = 0,
                        n_ctx: int = 2048,
                        max_length: Optional[int] = None,
                        verbose: bool = False) -> Tuple[np.ndarray, dict]:
    """Simpler GGUF extraction using llama-cpp-python's high-level API.

    Processes the prompt, then uses the internal state save/restore API
    to dump the KV cache contents.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python is required for GGUF models.")
        print("Install with: python -m pip install llama-cpp-python")
        sys.exit(1)

    print(f"  Loading GGUF: {os.path.basename(model_path)}")
    t0 = time.time()

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
        type_k=0,  # GGML_TYPE_F32 — store K cache in full precision
        type_v=0,  # GGML_TYPE_F32 — store V cache in full precision
    )

    # Get model params
    from llama_cpp import llama_cpp
    lmodel = llm._model.model
    n_layers = llama_cpp.llama_model_n_layer(lmodel)
    n_heads = llama_cpp.llama_model_n_head(lmodel)
    n_kv_heads = llama_cpp.llama_model_n_head_kv(lmodel)
    n_embd = llama_cpp.llama_model_n_embd(lmodel)
    head_dim = n_embd // n_heads

    dt_load = time.time() - t0
    print(f"  Loaded in {dt_load:.1f}s")
    print(f"  Layers: {n_layers}, Heads: {n_heads}, KV heads: {n_kv_heads}, Head dim: {head_dim}")

    # Tokenize
    tokens = llm.tokenize(prompt.encode('utf-8'), add_bos=True)
    n_tokens = len(tokens)
    if max_length and n_tokens > max_length:
        tokens = tokens[:max_length]
        n_tokens = max_length
    print(f"  Prompt: {n_tokens} tokens")

    # Process tokens
    print(f"  Forward pass...")
    t0 = time.time()
    llm.eval(tokens)
    dt_eval = time.time() - t0
    print(f"  Done in {dt_eval:.1f}s")

    # ── Extract K vectors from the KV cache ────────────────────────────
    print(f"  Extracting K vectors from KV cache...")
    ctx = llm._ctx.ctx
    k_vectors = None
    method = 'unknown'

    # Strategy 1: Per-sequence state API (cleanest — returns only used cells)
    try:
        print(f"  Trying per-sequence state API...")
        k_vectors = _extract_via_seq_api(ctx, n_layers, n_kv_heads, head_dim, n_tokens, verbose)
        if k_vectors is not None:
            method = 'gguf_seq_state'
            print(f"  Per-sequence extraction succeeded: {k_vectors.shape}")
    except Exception as e:
        if verbose:
            print(f"    Per-sequence API failed: {e}")

    # Strategy 2: Full state dump with smart scanning
    if k_vectors is None:
        try:
            print(f"  Trying full state dump...")
            state_size = llama_cpp.llama_state_get_size(ctx)
            print(f"    State size: {state_size / 1024 / 1024:.1f} MB")

            buf = (ctypes.c_uint8 * state_size)()
            written = llama_cpp.llama_state_get_data(ctx, buf, state_size)
            print(f"    Written: {written / 1024 / 1024:.1f} MB")

            state_data = bytes(buf[:written])
            del buf

            k_vectors = _scan_for_kv_data(
                state_data, n_layers, n_kv_heads, head_dim, n_tokens, verbose
            )
            if k_vectors is not None:
                method = 'gguf_state_scan'
                print(f"  State dump scan succeeded: {k_vectors.shape}")
        except Exception as e:
            if verbose:
                print(f"    State dump failed: {e}")

    # Strategy 3: Use the other extraction path's direct cache reader
    if k_vectors is None:
        try:
            print(f"  Trying direct cache access...")
            lmodel = llm._model.model
            k_vectors = _extract_kv_cache_direct(
                llm, ctx, lmodel, n_layers, n_kv_heads, head_dim, n_tokens, verbose
            )
            if k_vectors is not None:
                method = 'gguf_direct'
                print(f"  Direct cache access succeeded: {k_vectors.shape}")
        except Exception as e:
            if verbose:
                print(f"    Direct cache access failed: {e}")

    if k_vectors is not None:
        metadata = {
            'model_name': os.path.basename(model_path),
            'model_path': model_path,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_kv_heads': n_kv_heads,
            'head_dim': head_dim,
            'n_positions': k_vectors.shape[2],
            'method': method,
            'load_time': dt_load,
            'eval_time': dt_eval,
        }
        return k_vectors, metadata

    raise RuntimeError(
        "Could not extract KV cache from GGUF model. Tried: "
        "per-sequence API, full state dump scan, and direct cache access. "
        "Try the HuggingFace extraction path instead."
    )


def _extract_via_seq_api(ctx, n_layers: int, n_kv_heads: int, head_dim: int,
                         n_tokens: int, verbose: bool) -> Optional[np.ndarray]:
    """Extract KV cache using the per-sequence state API.

    Uses llama_state_seq_get_data or llama_kv_self_seq_get_data, which
    returns only the used cells for a given sequence (cleaner than the
    full state dump).
    """
    from llama_cpp import llama_cpp

    # Find the right API function
    get_fn = None
    size_fn = None
    for get_name, size_name in [
        ('llama_state_seq_get_data', 'llama_state_seq_get_size'),
        ('llama_kv_self_seq_get_data', 'llama_kv_self_seq_get_size'),
    ]:
        if hasattr(llama_cpp, get_name) and hasattr(llama_cpp, size_name):
            get_fn = getattr(llama_cpp, get_name)
            size_fn = getattr(llama_cpp, size_name)
            if verbose:
                print(f"    Using {get_name}")
            break

    if get_fn is None:
        raise RuntimeError("No per-sequence state API found in this llama-cpp-python version")

    seq_id = 0
    state_size = size_fn(ctx, seq_id)
    if verbose:
        print(f"    Seq state size: {state_size} bytes ({state_size/1024/1024:.2f} MB)")

    if state_size == 0:
        raise RuntimeError("Seq state size is 0 — KV cache may be empty")

    buf = (ctypes.c_uint8 * state_size)()
    written = get_fn(ctx, buf, state_size, seq_id)
    if verbose:
        print(f"    Got {written} bytes")

    data = bytes(buf[:written])
    del buf

    # Parse the per-sequence data (same scanner, but should be cleaner)
    k_vectors = _scan_for_kv_data(data, n_layers, n_kv_heads, head_dim, n_tokens, verbose)
    return k_vectors


def _scan_for_kv_data(data: bytes, n_layers: int, n_kv_heads: int, head_dim: int,
                      n_tokens: int, verbose: bool) -> Optional[np.ndarray]:
    """Scan the full state dump for the KV cache block.

    The state dump format varies by llama.cpp version. The KV section may
    store only used cells (n_tokens) or all allocated cells (n_ctx). We try
    multiple cell counts and both f32/f16 dtypes.
    """
    data_len = len(data)

    # Bytes per cell per layer for K (or V): n_kv_heads * head_dim * sizeof
    cell_layer_f32 = n_kv_heads * head_dim * 4
    cell_layer_f16 = n_kv_heads * head_dim * 2

    # Infer possible cell counts from data size
    # Total KV for N cells: 2 * n_layers * N * cell_layer_bytes + header
    # So N ≈ (data_len - header) / (2 * n_layers * cell_layer_bytes)
    possible_n_cells = set()
    possible_n_cells.add(n_tokens)  # The prompt length

    for header_est in [0, 64, 256, 512, 1024, 4096, 8192]:
        for cell_bytes, dtype_name in [(cell_layer_f32, 'f32'), (cell_layer_f16, 'f16')]:
            divisor = 2 * n_layers * cell_bytes
            if divisor > 0:
                n_cells_est = (data_len - header_est) / divisor
                # Accept if it's close to an integer
                n_cells_round = round(n_cells_est)
                if n_cells_round > 0 and abs(n_cells_est - n_cells_round) < 0.01:
                    possible_n_cells.add(n_cells_round)

    # Also try common n_ctx values
    for n_ctx_try in [128, 256, 512, 1024, 2048, 4096, 8192]:
        possible_n_cells.add(n_ctx_try)

    if verbose:
        print(f"    Data size: {data_len} bytes ({data_len/1024/1024:.1f} MB)")
        print(f"    Trying cell counts: {sorted(possible_n_cells)}")

    best_k = None
    best_score = 0
    best_n_cells = 0

    for n_cells in sorted(possible_n_cells):
        for dtype, cell_bytes, dtype_np in [
            ('f32', cell_layer_f32, np.float32),
            ('f16', cell_layer_f16, np.float16),
        ]:
            k_bytes_per_layer = n_cells * cell_bytes
            total_kv = 2 * n_layers * k_bytes_per_layer

            if total_kv > data_len or total_kv == 0:
                continue

            # Build candidate start offsets
            candidates = set()
            exact_header = data_len - total_kv
            # Try exact fit and nearby
            for delta in range(0, min(64, exact_header + 1), 4):
                candidates.add(exact_header - delta)
            # Also try small fixed headers
            candidates.update([0, 4, 8, 16, 32, 64, 128, 256, 512])

            # Look for uint32 markers in the first 4K
            for off in range(0, min(data_len - 4, 4096), 4):
                val = int.from_bytes(data[off:off+4], 'little')
                if val in (n_cells, n_tokens, n_layers):
                    for d in [4, 8, 12, 16, 20, 24]:
                        candidates.add(off + d)

            candidates = sorted(c for c in candidates if 0 <= c and c + total_kv <= data_len)

            for offset in candidates:
                try:
                    # Quick test: read first layer K
                    test_raw = np.frombuffer(
                        data[offset:offset + k_bytes_per_layer],
                        dtype=dtype_np
                    )
                    if dtype_np == np.float16:
                        test_k = test_raw.astype(np.float32)
                    else:
                        test_k = test_raw

                    if np.isnan(test_k).any() or np.isinf(test_k).any():
                        continue
                    if np.abs(test_k).max() > 1000 or np.abs(test_k).max() < 1e-10:
                        continue

                    score = float(test_k.std())
                    if score < 0.01 or score > 100:
                        continue

                    # Try to parse all layers
                    k_vectors_full = np.zeros(
                        (n_layers, n_kv_heads, n_cells, head_dim), dtype=np.float32
                    )
                    valid = True
                    off = offset
                    for L in range(n_layers):
                        raw = np.frombuffer(data[off:off + k_bytes_per_layer], dtype=dtype_np)
                        if dtype_np == np.float16:
                            k_flat = raw.astype(np.float32)
                        else:
                            k_flat = raw
                        if np.isnan(k_flat).any():
                            valid = False
                            break

                        # Try two possible layouts
                        try:
                            k_vectors_full[L] = k_flat.reshape(
                                n_cells, n_kv_heads, head_dim
                            ).transpose(1, 0, 2)
                        except ValueError:
                            valid = False
                            break

                        off += k_bytes_per_layer
                        off += k_bytes_per_layer  # Skip V data

                    if not valid:
                        continue

                    # Validation: layers should differ from each other
                    l0_norm = float(np.linalg.norm(k_vectors_full[0], axis=-1).mean())
                    lN_norm = float(np.linalg.norm(k_vectors_full[-1], axis=-1).mean())
                    if abs(l0_norm - lN_norm) < 0.001:
                        continue

                    # Check that non-zero data exists in later layers too
                    mid = n_layers // 2
                    lM_max = float(np.abs(k_vectors_full[mid]).max())
                    if lM_max < 1e-8:
                        continue

                    if verbose:
                        print(f"    Found KV at offset {offset}, n_cells={n_cells}, "
                              f"dtype={dtype} (score={score:.4f})")
                        print(f"    L0 norm={l0_norm:.4f}, L{n_layers-1} norm={lN_norm:.4f}")

                    if score > best_score:
                        best_score = score
                        best_n_cells = n_cells
                        # Trim to only the actual used tokens if n_cells > n_tokens
                        if n_cells > n_tokens:
                            # Cells 0..n_tokens-1 are the actual cached data
                            best_k = k_vectors_full[:, :, :n_tokens, :].copy()
                        else:
                            best_k = k_vectors_full.copy()

                except (ValueError, IndexError):
                    continue

    if best_k is not None and verbose:
        print(f"    Best parse: n_cells={best_n_cells}, score={best_score:.4f}, "
              f"output shape={best_k.shape}")

    return best_k


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_to_np(t):
    """Convert a (batch, n_heads, seq, hd) tensor to (n_heads, seq, hd) numpy."""
    import torch
    if isinstance(t, torch.Tensor):
        return t[0].detach().float().cpu().numpy()
    return np.array(t[0], dtype=np.float32)


def _extract_k_list(past_kv) -> List[np.ndarray]:
    """Extract per-layer K tensors from any HuggingFace cache format."""
    k_list = []

    # Strategy 1: transformers 5.x DynamicCache — .layers[i].keys
    if hasattr(past_kv, 'layers'):
        layers = past_kv.layers
        if hasattr(layers, '__len__') and len(layers) > 0:
            sample = layers[0]
            if hasattr(sample, 'keys') and hasattr(sample.keys, 'shape'):
                for layer_cache in layers:
                    k_list.append(_tensor_to_np(layer_cache.keys))
                if k_list:
                    return k_list
            if hasattr(sample, 'key_cache') and hasattr(sample.key_cache, 'shape'):
                for layer_cache in layers:
                    k_list.append(_tensor_to_np(layer_cache.key_cache))
                if k_list:
                    return k_list

    # Strategy 2: transformers 4.36+ DynamicCache — .key_cache list
    if hasattr(past_kv, 'key_cache'):
        cache = past_kv.key_cache
        if isinstance(cache, list) and len(cache) > 0:
            import torch
            if isinstance(cache[0], torch.Tensor):
                for k_tensor in cache:
                    k_list.append(_tensor_to_np(k_tensor))
                if k_list:
                    return k_list

    # Strategy 3: Indexable — past_kv[i] returns (K, V) tuple
    if hasattr(past_kv, '__getitem__') and hasattr(past_kv, '__len__'):
        try:
            n = len(past_kv)
            for i in range(n):
                item = past_kv[i]
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    k_list.append(_tensor_to_np(item[0]))
                elif hasattr(item, 'detach'):
                    k_list.append(_tensor_to_np(item))
            if k_list:
                return k_list
        except (TypeError, IndexError, KeyError):
            k_list = []

    # Strategy 4: Plain list/tuple of (K, V) pairs
    if isinstance(past_kv, (list, tuple)):
        for layer_kv in past_kv:
            if isinstance(layer_kv, (list, tuple)) and len(layer_kv) >= 1:
                k_list.append(_tensor_to_np(layer_kv[0]))
        if k_list:
            return k_list

    # Debug dump
    attrs = [a for a in dir(past_kv) if not a.startswith('_')]
    print(f"  [debug] past_key_values type: {type(past_kv)}")
    print(f"  [debug] attributes: {attrs}")
    if hasattr(past_kv, 'layers'):
        layers = past_kv.layers
        print(f"  [debug] .layers len: {len(layers) if hasattr(layers, '__len__') else '?'}")
        if hasattr(layers, '__len__') and len(layers) > 0:
            s = layers[0]
            print(f"  [debug] layers[0] type: {type(s)}")
            print(f"  [debug] layers[0] attrs: {[a for a in dir(s) if not a.startswith('_')]}")
    raise RuntimeError(f"Could not extract K from cache (type: {type(past_kv).__name__})")


def extract_from_hf_cache(model, tokenizer, prompt: str,
                          max_length: Optional[int] = None) -> np.ndarray:
    """Extract K vectors from HuggingFace model's KV cache."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    n_tokens = input_ids.shape[1]
    if max_length and n_tokens > max_length:
        input_ids = input_ids[:, :max_length]
        n_tokens = max_length

    print(f"  Prompt tokenized to {n_tokens} tokens")

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, output_attentions=False)

    past_kv = outputs.past_key_values
    if past_kv is None:
        raise RuntimeError("Model did not return past_key_values.")

    k_list = _extract_k_list(past_kv)

    n_layers = len(k_list)
    n_heads, n_pos, head_dim = k_list[0].shape
    print(f"  Captured: {n_layers} layers × {n_heads} heads × {n_pos} pos × {head_dim} dim")

    return np.stack(k_list, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract K vectors from models for Prime Chord analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # GGUF model (recommended for local models):
    python extract_kv.py --model Phi-3.1-mini-4k-instruct-Q8_0.gguf --output kv.npz
    python extract_kv.py --model model.gguf --n-gpu-layers 99 --output kv.npz

    # HuggingFace model:
    python extract_kv.py --model Qwen/Qwen2-0.5B --output kv.npz --device cuda

    # Then run the diagnostic:
    python sp_chord_diagnostic.py --input kv.npz --sqfree --plot
        """
    )

    parser.add_argument('--model', type=str, required=True,
                        help='Model path (.gguf) or HuggingFace model name')
    parser.add_argument('--output', '-o', type=str, default='kv_vectors.npz',
                        help='Output path (.npz or .pt)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Prompt text (default: calibration prompt)')
    parser.add_argument('--prompt-file', type=str, default=None,
                        help='Read prompt from a text file')
    parser.add_argument('--max-length', type=int, default=None,
                        help='Maximum number of tokens (truncate prompt if longer)')
    parser.add_argument('--layers', type=str, default=None,
                        help='Layer range to extract (e.g., "0-7", "0,4,8,12")')
    parser.add_argument('--heads', type=str, default=None,
                        help='Head range to extract (e.g., "0-3", "0,2,4")')

    # GGUF-specific options
    parser.add_argument('--n-gpu-layers', type=int, default=0,
                        help='Number of layers to offload to GPU (GGUF only, 0=CPU, 99=all)')
    parser.add_argument('--n-ctx', type=int, default=2048,
                        help='Context size (GGUF only, default: 2048)')

    # HuggingFace-specific options
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cpu, cuda, auto (HF only)')
    parser.add_argument('--dtype', type=str, default='auto',
                        choices=['auto', 'float32', 'float16', 'bfloat16'],
                        help='Model precision (HF only)')
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='Trust remote code (HF only)')

    parser.add_argument('--per-head', action='store_true',
                        help='Also save per-head arrays (k_layer{L}_head{H})')
    parser.add_argument('--layer-period', type=int, default=None,
                        help='Repeating attention-type period for hybrid models. '
                             'Saves layer_types array to .npz. '
                             'Known: Qwen3.6 (27B, 35B-A3B) = 4, Gemma4 = 4')
    parser.add_argument('--global-offset', type=int, default=3,
                        help='Layer index within the period that is full/global attention '
                             '(default: 3). With --layer-period 4: layers 3,7,11,… are global.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # ── Resolve prompt ───────────────────────────────────────────────────
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read().strip()
        print(f"Loaded prompt from {args.prompt_file} ({len(prompt)} chars)")
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPT
        print(f"Using default calibration prompt ({len(prompt)} chars)")

    # ── Detect model type and extract ────────────────────────────────────
    is_gguf = args.model.lower().endswith('.gguf') or os.path.isfile(args.model) and '.gguf' in args.model.lower()

    if is_gguf:
        # ── GGUF path ────────────────────────────────────────────────────
        if not os.path.isfile(args.model):
            print(f"ERROR: GGUF file not found: {args.model}")
            sys.exit(1)

        print(f"\nExtracting K vectors from GGUF model...")
        t0 = time.time()

        # Try the simple state-dump approach first
        try:
            k_vectors, metadata = extract_gguf_simple(
                args.model, prompt,
                n_gpu_layers=args.n_gpu_layers,
                n_ctx=args.n_ctx,
                max_length=args.max_length,
                verbose=args.verbose,
            )
        except RuntimeError as e:
            print(f"\n  Simple extraction failed: {e}")
            print(f"  Trying alternative approach...")
            k_vectors, metadata = extract_from_gguf(
                args.model, prompt,
                n_gpu_layers=args.n_gpu_layers,
                n_ctx=args.n_ctx,
                max_length=args.max_length,
                verbose=args.verbose,
            )

        dt_total = time.time() - t0

    else:
        # ── HuggingFace path ─────────────────────────────────────────────
        print("Loading dependencies...")
        try:
            import torch
        except ImportError:
            print("ERROR: PyTorch required. Install: python -m pip install torch")
            sys.exit(1)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        except ImportError:
            print("ERROR: transformers required. Install: python -m pip install transformers")
            sys.exit(1)

        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device

        dtype_map = {
            'auto': None, 'float32': torch.float32,
            'float16': torch.float16, 'bfloat16': torch.bfloat16,
        }
        torch_dtype = dtype_map[args.dtype]
        if args.dtype == 'auto' and device == 'cpu':
            torch_dtype = torch.float32

        print(f"Device: {device}, Dtype: {torch_dtype or 'auto'}")
        print(f"\nLoading model: {args.model}")
        t0 = time.time()

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        n_layers_total = getattr(config, 'num_hidden_layers', None)
        n_heads_total = getattr(config, 'num_attention_heads', None)
        n_kv_heads_cfg = getattr(config, 'num_key_value_heads', n_heads_total)
        head_dim_cfg = getattr(config, 'head_dim', None)
        if head_dim_cfg is None and hasattr(config, 'hidden_size') and n_heads_total:
            head_dim_cfg = config.hidden_size // n_heads_total

        print(f"  Architecture: {config.model_type}")
        print(f"  Layers: {n_layers_total}, Heads: {n_heads_total}, KV heads: {n_kv_heads_cfg}, Head dim: {head_dim_cfg}")

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        model_kwargs = {'trust_remote_code': args.trust_remote_code,
                        'device_map': device if device != 'cpu' else None}
        if torch_dtype:
            model_kwargs['torch_dtype'] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        if device == 'cpu' or model_kwargs.get('device_map') is None:
            model = model.to(device)
        model.eval()

        dt_load = time.time() - t0
        print(f"  Loaded in {dt_load:.1f}s")

        print(f"\nExtracting K vectors...")
        t0_extract = time.time()
        k_vectors = extract_from_hf_cache(model, tokenizer, prompt, args.max_length)
        dt_extract = time.time() - t0_extract
        dt_total = dt_load + dt_extract

        metadata = {
            'model_name': args.model,
            'n_layers': k_vectors.shape[0],
            'n_heads': k_vectors.shape[1],
            'n_kv_heads': n_kv_heads_cfg or k_vectors.shape[1],
            'head_dim': k_vectors.shape[3],
            'n_positions': k_vectors.shape[2],
            'device': device,
            'dtype': str(torch_dtype),
            'method': 'hf_cache',
        }

    # ── Apply layer/head filtering ───────────────────────────────────────
    n_layers, n_heads, n_pos, head_dim = k_vectors.shape

    if args.layers:
        layer_indices = parse_range(args.layers, n_layers)
        k_vectors = k_vectors[layer_indices]
        n_layers = k_vectors.shape[0]

    if args.heads:
        head_indices = parse_range(args.heads, n_heads)
        k_vectors = k_vectors[:, head_indices]
        n_heads = k_vectors.shape[1]

    print(f"\n  Final shape: ({n_layers} layers, {n_heads} heads, {n_pos} positions, {head_dim} head_dim)")

    # ── K vector statistics ──────────────────────────────────────────────
    if args.verbose:
        print("\n  K vector statistics:")
        for L in range(min(n_layers, 4)):
            norms = np.linalg.norm(k_vectors[L], axis=-1)
            print(f"    Layer {L}: norm mean={norms.mean():.4f}, std={norms.std():.4f}, "
                  f"min={norms.min():.4f}, max={norms.max():.4f}")
        if n_layers > 4:
            print(f"    ... ({n_layers - 4} more layers)")

    # ── Detect layer types ───────────────────────────────────────────────
    layer_types = None
    lt_source   = None

    if args.layer_period:
        # User-specified periodic pattern
        period = args.layer_period
        offset = args.global_offset
        layer_types = np.array(
            ['global' if L % period == offset else 'local' for L in range(n_layers)]
        )
        lt_source = f'period={period},offset={offset}'

    else:
        # Auto-detect from GGUF metadata: look for sliding_window key
        try:
            if is_gguf:
                from llama_cpp import Llama, llama_cpp as _lc
                import ctypes as _ct
                _m = metadata  # already have model metadata
                # Try to read GGUF metadata keys for SWA
                sw_key_hits = 0
                try:
                    lm = Llama(model_path=args.model, n_ctx=8, verbose=False)._model.model
                    n_meta = _lc.llama_model_meta_count(lm)
                    sw_val = None
                    for idx in range(n_meta):
                        kbuf = (_ct.c_char * 256)()
                        vbuf = (_ct.c_char * 256)()
                        _lc.llama_model_meta_key_by_index(lm, idx, kbuf, 256)
                        _lc.llama_model_meta_val_str_by_index(lm, idx, vbuf, 256)
                        k_str = kbuf.value.decode('utf-8', errors='replace')
                        v_str = vbuf.value.decode('utf-8', errors='replace')
                        if 'sliding_window' in k_str.lower() and 'layer' not in k_str.lower():
                            try:
                                sw_val = int(v_str.strip())
                                sw_key_hits += 1
                            except ValueError:
                                pass
                    if sw_key_hits > 0 and sw_val and sw_val > 0:
                        # SWA model detected: use the standard 3:1 pattern (period=4, offset=3)
                        # This matches Qwen3.6 and Gemma4 architectures
                        period, offset = 4, 3
                        layer_types = np.array(
                            ['global' if L % period == offset else 'local'
                             for L in range(n_layers)]
                        )
                        lt_source = f'auto-detected(sw={sw_val},period={period},offset={offset})'
                        print(f"  Auto-detected SWA (sliding_window={sw_val}): "
                              f"period={period}, offset={offset} "
                              f"-> {np.sum(layer_types=='global')} global, "
                              f"{np.sum(layer_types=='local')} local layers")
                except Exception:
                    pass
        except Exception:
            pass

    # ── Save output ──────────────────────────────────────────────────────
    print(f"\nSaving to {args.output}...")

    save_dict = {'k_vectors': k_vectors}
    if args.per_head:
        for L in range(n_layers):
            for H in range(n_heads):
                save_dict[f'k_layer{L}_head{H}'] = k_vectors[L, H]

    if layer_types is not None:
        save_dict['layer_types'] = layer_types
        n_global = int(np.sum(layer_types == 'global'))
        n_local  = int(np.sum(layer_types == 'local'))
        metadata['layer_types_source'] = lt_source
        metadata['n_global_layers']    = n_global
        metadata['n_local_layers']     = n_local
        print(f"  Layer types: {n_global} global, {n_local} local ({lt_source})")

    save_dict['metadata'] = np.array(json.dumps(metadata))
    np.savez_compressed(args.output, **save_dict)

    file_size = os.path.getsize(args.output)
    size_mb = file_size / (1024 * 1024)
    print(f"  Saved: {size_mb:.1f} MB")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Model:      {metadata.get('model_name', args.model)}")
    print(f"  Shape:      {n_layers}L × {n_heads}H × {n_pos}P × {head_dim}D")
    print(f"  File:       {args.output} ({size_mb:.1f} MB)")
    print(f"  Time:       {dt_total:.1f}s")
    print(f"{'='*60}")
    print(f"\nNext step:")
    print(f"  python sp_chord_diagnostic.py --input {args.output} --sqfree --plot")


if __name__ == '__main__':
    main()
