"""
Shannon-Prime / Phase 2.2 — transformer attention block ONNX builder.

Builds a self-contained ONNX graph for ONE Qwen3-4B-shape attention block
(input x -> Q/K/V projections -> multi-head scaled-dot-product attention
-> output projection). Weights are inlined as Initializers so the graph is
deployable as-is via Qualcomm AI Hub's compile_job → qnn_context_binary
pipeline (same flow as compile job j5wm6m34g in commit e5b3963).

Why fp32 first: this is a baseline run. We want the V69 HTP latency for
a transformer-shaped graph at our actual target dimensions before we
introduce w4a16 quantization (Phase 2.2.1) or KV-cache integration
(Phase 2.3). fp32 will compile cleanly via QNN's fp16-fallback path on
HTP and give us an upper-bound latency.

Scale knobs: build_attention_block_onnx(seq_len=64, d_model=2048,
n_heads=16, head_dim=128) is the Qwen3-4B-Coder default. Smaller defaults
let us iterate quickly without paying the 64MB-per-projection upload.

ir_version=8 + opset 13 is the AI Hub-accepted combination per Phase 2.1
findings (commit e5b3963).
"""
from __future__ import annotations

import math
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def _const_weight(name: str, shape: tuple[int, ...], scale: float = 0.01) -> onnx.TensorProto:
    """Initializer-backed constant weight.

    Deterministic seeded values so the compiled graph is reproducible —
    we don't actually care about the values for a perf run, but want the
    graph to be sound (no NaNs, distinguishable activations) so QNN's
    auto-quantize calibration step doesn't degenerate.
    """
    rng = np.random.default_rng(seed=hash(name) & 0xFFFFFFFF)
    arr = (rng.standard_normal(size=shape).astype(np.float32) * scale)
    return numpy_helper.from_array(arr, name=name)


def build_attention_block_onnx(
    seq_len: int = 64,
    d_model: int = 2048,
    n_heads: int = 16,
    head_dim: int = 128,
):
    """Single transformer attention block, fp32, self-contained ONNX.

    Shapes:
        x:        (1, seq_len, d_model)            input
        W_Q,K,V:  (d_model, d_model)                projections
        W_O:      (d_model, d_model)                output projection
        Q,K,V:    (1, seq_len, d_model)             after projection
        Q_h:      (1, n_heads, seq_len, head_dim)   after reshape+transpose
        K_h:      (1, n_heads, seq_len, head_dim)   ditto
        V_h:      (1, n_heads, seq_len, head_dim)   ditto
        scores:   (1, n_heads, seq_len, seq_len)    Q_h @ K_h^T / sqrt(head_dim)
        probs:    (1, n_heads, seq_len, seq_len)    softmax(scores)
        attn:     (1, n_heads, seq_len, head_dim)   probs @ V_h
        out:      (1, seq_len, d_model)             reshape, then @ W_O
    """
    assert n_heads * head_dim == d_model, \
        f"n_heads*head_dim ({n_heads}*{head_dim}={n_heads*head_dim}) must equal d_model ({d_model})"

    inv_sqrt_dh = 1.0 / math.sqrt(head_dim)

    # Initializers (weights + reshape constants)
    W_Q = _const_weight("W_Q", (d_model, d_model))
    W_K = _const_weight("W_K", (d_model, d_model))
    W_V = _const_weight("W_V", (d_model, d_model))
    W_O = _const_weight("W_O", (d_model, d_model))

    # Reshape constants
    shape_to_heads = numpy_helper.from_array(
        np.array([1, seq_len, n_heads, head_dim], dtype=np.int64),
        name="shape_to_heads",
    )
    shape_back = numpy_helper.from_array(
        np.array([1, seq_len, d_model], dtype=np.int64),
        name="shape_back",
    )

    # Scale constant for attention logits
    scale = numpy_helper.from_array(
        np.array(inv_sqrt_dh, dtype=np.float32),
        name="scale",
    )

    nodes = []

    # 1. Q/K/V projections: y = x @ W
    nodes += [
        helper.make_node("MatMul", ["x", "W_Q"], ["Q"]),
        helper.make_node("MatMul", ["x", "W_K"], ["K"]),
        helper.make_node("MatMul", ["x", "W_V"], ["V"]),
    ]

    # 2. Reshape (1, seq_len, d_model) -> (1, seq_len, n_heads, head_dim)
    nodes += [
        helper.make_node("Reshape", ["Q", "shape_to_heads"], ["Q_h_pre"]),
        helper.make_node("Reshape", ["K", "shape_to_heads"], ["K_h_pre"]),
        helper.make_node("Reshape", ["V", "shape_to_heads"], ["V_h_pre"]),
    ]

    # 3. Transpose (1, seq_len, n_heads, head_dim) -> (1, n_heads, seq_len, head_dim)
    nodes += [
        helper.make_node("Transpose", ["Q_h_pre"], ["Q_h"], perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", ["K_h_pre"], ["K_h"], perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", ["V_h_pre"], ["V_h"], perm=[0, 2, 1, 3]),
    ]

    # 4. K^T over the last two dims: (1, n_heads, seq_len, head_dim) -> (1, n_heads, head_dim, seq_len)
    nodes += [
        helper.make_node("Transpose", ["K_h"], ["K_h_T"], perm=[0, 1, 3, 2]),
    ]

    # 5. Attention scores: Q_h @ K_h_T -> (1, n_heads, seq_len, seq_len)
    nodes += [
        helper.make_node("MatMul", ["Q_h", "K_h_T"], ["scores_raw"]),
        helper.make_node("Mul", ["scores_raw", "scale"], ["scores"]),
    ]

    # 6. Softmax over last axis
    nodes += [
        helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
    ]

    # 7. Apply attention to V: probs @ V_h -> (1, n_heads, seq_len, head_dim)
    nodes += [
        helper.make_node("MatMul", ["probs", "V_h"], ["attn_h"]),
    ]

    # 8. Transpose back and reshape to (1, seq_len, d_model)
    nodes += [
        helper.make_node("Transpose", ["attn_h"], ["attn_h_t"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["attn_h_t", "shape_back"], ["attn_flat"]),
    ]

    # 9. Output projection
    nodes += [
        helper.make_node("MatMul", ["attn_flat", "W_O"], ["y"]),
    ]

    # I/O
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, seq_len, d_model])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, seq_len, d_model])

    initializers = [W_Q, W_K, W_V, W_O, shape_to_heads, shape_back, scale]
    graph = helper.make_graph(nodes, "attn_block", [x], [y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    # Quick local validation — no AI Hub submission, just shape-check.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=2048)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--out", default=None, help="optional path to save the .onnx")
    args = ap.parse_args()

    m = build_attention_block_onnx(
        seq_len=args.seq, d_model=args.d_model,
        n_heads=args.n_heads, head_dim=args.head_dim,
    )
    n_init = len(m.graph.initializer)
    n_nodes = len(m.graph.node)
    total_init_bytes = sum(
        np.prod(list(t.dims)).item() * (4 if t.data_type == TensorProto.FLOAT else 8)
        for t in m.graph.initializer
    )
    print(f"attn_block: seq={args.seq} d_model={args.d_model} "
          f"heads={args.n_heads} head_dim={args.head_dim}")
    print(f"  nodes:        {n_nodes}")
    print(f"  initializers: {n_init} ({total_init_bytes/1024/1024:.1f} MB)")
    print(f"  ir_version:   {m.ir_version}, opset: 13")
    if args.out:
        onnx.save(m, args.out)
        import os
        print(f"  saved: {args.out} ({os.path.getsize(args.out)/1024/1024:.1f} MB)")
