#!/usr/bin/env python3
"""
calibrate_sp_flash.py — Calibrate Vilenkin W matrices for SpFlashOracleNative.

Usage
-----
# 1. Capture hidden states from the target model:
#    Run sp-engine with SP_FLASH_DUMP_CAPTURE=<dir> to write per-prompt
#    capture files. Each file is a numpy .npy with shape
#    [num_target_layers, hidden_size] (one file per calibration step).
#
# 2. Fit W matrices:
#    python calibrate_sp_flash.py --capture-dir captures/ --out calibration/
#
# 3. Outputs:
#    calibration/layer_<N>.bin  — float32 array [hidden_size × 14], one per layer
#    calibration/meta.json      — hidden_size, target_layer_ids, n_skel
#
# W matrices are loaded by SpFlashOracleNative and used by
# sp_flash_vilenkin_predict() for context-free draft generation.

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# VHT2 — Vilenkin-Hartley Transform (Python reference implementation)
# ---------------------------------------------------------------------------
# VHT2 is a staged Hartley transform over small primes {2, 3, 5, 7, 11}.
# Each stage applies a p-point Hartley butterfly, normalised by 1/sqrt(p).
# The transform is self-inverse: VHT2(VHT2(x)) == x.

def _hartley_stage(data: np.ndarray, p: int) -> None:
    """In-place p-point Hartley butterfly stage over a float64 array."""
    n = len(data)
    if n % p != 0:
        raise ValueError(f"length {n} not divisible by {p}")
    step = n // p
    # Process each block of `step` positions
    for start in range(step):
        block = data[start::step].copy()  # [p] elements
        out = np.zeros(p, dtype=np.float64)
        for k in range(p):
            s = 0.0
            for j in range(p):
                angle = 2.0 * math.pi * j * k / p
                s += block[j] * (math.cos(angle) + math.sin(angle))
            out[k] = s
        data[start::step] = out / math.sqrt(p)


_VHT2_PRIMES = [2, 3, 5, 7, 11]


def vht2(x: np.ndarray) -> np.ndarray:
    """
    Apply VHT2 to a 1-D float array. Length must factor into {2, 3, 5, 7, 11}.
    Returns a new float64 array. Self-inverse: vht2(vht2(x)) ≈ x.
    """
    data = x.astype(np.float64).copy()
    n = len(data)
    for p in _VHT2_PRIMES:
        while n % p == 0:
            _hartley_stage(data, p)
            n //= p
    if n != 1:
        raise ValueError(f"VHT2: length {len(x)} has prime factors outside {{2,3,5,7,11}}")
    return data


def vht2_batch(X: np.ndarray) -> np.ndarray:
    """Apply VHT2 to each row of a 2-D array [N, D]."""
    return np.array([vht2(row) for row in X], dtype=np.float64)


# ---------------------------------------------------------------------------
# Skeleton extraction
# ---------------------------------------------------------------------------

N_SKEL = 14  # H_2 ⊗ H_7 = 2 × 7 Kronecker Vilenkin skeleton


def extract_skeleton(hidden: np.ndarray) -> np.ndarray:
    """
    Apply VHT2 and return the first N_SKEL (14) coefficients.
    hidden: [hidden_size] float array.
    """
    transformed = vht2(hidden)
    return transformed[:N_SKEL]


def extract_skeleton_batch(hiddens: np.ndarray) -> np.ndarray:
    """
    hiddens: [N, hidden_size] — returns [N, N_SKEL].
    """
    transformed = vht2_batch(hiddens)
    return transformed[:, :N_SKEL]


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------

def fit_W(skeletons: np.ndarray, targets: np.ndarray,
          lam: float = 1e-3) -> np.ndarray:
    """
    Fit W such that W @ skel ≈ hidden for all calibration samples.

    skeletons : [N, N_SKEL]   — VHT2 skeleton of target hidden states
    targets   : [N, hidden_size] — original target hidden states (ground truth)
    lam       : ridge regularisation strength

    Returns W : [hidden_size, N_SKEL] (row-major, as expected by
                sp_flash_vilenkin_predict which does W[i,:] @ skel).
    """
    X = skeletons   # [N, N_SKEL]
    Y = targets     # [N, hidden_size]

    # Normal equations: W^T = (X^T X + λI)^{-1} X^T Y
    # → W = Y^T X (X^T X + λI)^{-1}    [hidden_size, N_SKEL]
    A = X.T @ X + lam * np.eye(N_SKEL)   # [N_SKEL, N_SKEL]
    B = X.T @ Y                           # [N_SKEL, hidden_size]
    W_T = np.linalg.solve(A, B)           # [N_SKEL, hidden_size]
    return W_T.T.astype(np.float32)       # [hidden_size, N_SKEL]


# ---------------------------------------------------------------------------
# Capture file loading
# ---------------------------------------------------------------------------

def load_captures(capture_dir: str) -> dict:
    """
    Load all capture files from capture_dir.
    Each .npy file has shape [num_target_layers, hidden_size].
    Returns dict: layer_idx → list of [hidden_size] arrays.
    """
    capture_dir = Path(capture_dir)
    files = sorted(capture_dir.glob("capture_*.npy"))
    if not files:
        print(f"[calibrate] No capture_*.npy files found in {capture_dir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[calibrate] Found {len(files)} capture files in {capture_dir}")

    # Peek at first file to get dimensions
    sample = np.load(str(files[0]))
    if sample.ndim != 2:
        print(f"[calibrate] ERROR: expected 2-D capture (num_layers, hidden_size), "
              f"got shape {sample.shape}", file=sys.stderr)
        sys.exit(1)

    num_layers, hidden_size = sample.shape
    print(f"[calibrate] num_target_layers={num_layers} hidden_size={hidden_size}")

    layer_hiddens = {li: [] for li in range(num_layers)}
    for f in files:
        cap = np.load(str(f))  # [num_layers, hidden_size]
        if cap.shape != (num_layers, hidden_size):
            print(f"[calibrate] WARN: {f.name} has unexpected shape {cap.shape}, skipping")
            continue
        for li in range(num_layers):
            layer_hiddens[li].append(cap[li])

    return layer_hiddens, hidden_size, num_layers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Vilenkin W matrices for SP-Flash Phase 3")
    parser.add_argument("--capture-dir", required=True,
                        help="Directory containing capture_*.npy files from sp-engine")
    parser.add_argument("--out", default="calibration",
                        help="Output directory for calibration binaries (default: calibration/)")
    parser.add_argument("--lam", type=float, default=1e-3,
                        help="Ridge regularisation λ (default: 1e-3)")
    parser.add_argument("--target-layer-ids", type=str, default=None,
                        help="Comma-separated target layer IDs (e.g. 1,10,19,28,37). "
                             "Written to meta.json; informational only.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_hiddens, hidden_size, num_layers = load_captures(args.capture_dir)

    target_layer_ids = None
    if args.target_layer_ids:
        target_layer_ids = [int(x) for x in args.target_layer_ids.split(",")]

    print(f"[calibrate] Fitting W for {num_layers} layers, "
          f"hidden_size={hidden_size}, N_skel={N_SKEL}, λ={args.lam}")

    for li in range(num_layers):
        hiddens = np.array(layer_hiddens[li], dtype=np.float64)  # [N, hidden_size]
        N = hiddens.shape[0]
        print(f"  Layer {li}: {N} samples", end="", flush=True)

        # Skeletons: apply VHT2 and take first 14 coefficients
        skeletons = extract_skeleton_batch(hiddens)  # [N, N_SKEL]

        # Regression targets: the original hidden states
        # We want W × skel ≈ VHT2^{-1}(hidden_vht2) = hidden,
        # but actually we want to predict hidden in the VHT2 basis
        # (sp_flash_vilenkin_predict applies VHT2 at the end to go back to spatial).
        # So targets = hiddens (spatial domain).
        W = fit_W(skeletons, hiddens.astype(np.float64), lam=args.lam)  # [hidden_size, N_SKEL]

        # Report reconstruction error
        pred = (skeletons @ W.T.astype(np.float64))  # [N, hidden_size]
        err = np.sqrt(np.mean((pred - hiddens) ** 2))
        print(f"  RMSE={err:.6f}")

        # Save as binary float32 [hidden_size × N_SKEL] row-major
        out_path = out_dir / f"layer_{li}.bin"
        W.astype(np.float32).tofile(str(out_path))
        print(f"    → {out_path}  ({W.nbytes:,} bytes)")

    # Write metadata
    meta = {
        "hidden_size":       hidden_size,
        "n_skel":            N_SKEL,
        "num_target_layers": num_layers,
        "target_layer_ids":  target_layer_ids,
        "lam":               args.lam,
        "W_shape":           [hidden_size, N_SKEL],
        "W_dtype":           "float32",
        "W_layout":          "row_major",
        "note": (
            "Load with: np.fromfile(path, dtype=np.float32).reshape(hidden_size, n_skel). "
            "C: float W[hidden_size * n_skel]; passed to sp_flash_vilenkin_pred_init()."
        )
    }
    meta_path = out_dir / "meta.json"
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[calibrate] Metadata → {meta_path}")
    print("[calibrate] Done.")


if __name__ == "__main__":
    main()
