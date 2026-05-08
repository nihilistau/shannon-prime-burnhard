"""
Phase 2.3.1 — submit attention-block ONNX through AI Hub's quantize +
compile pipeline. Produces a w4a16 (INT4 weights, INT16 activations)
QNN context binary for V69 HTP.

This is the AIMET-equivalent baseline. SP-spectral-bands → HTP-int4
direct mapping is a separate Phase 2.3.2 follow-up.
"""
from __future__ import annotations
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(__file__))
from attention_block import build_attention_block_onnx

import qai_hub as hub
import numpy as np
import onnx


def poll(job, label, max_seconds=300):
    start = time.time()
    last = None
    while time.time() - start < max_seconds:
        s = job.get_status()
        if s.code != last:
            print(f"  [{label}] t={int(time.time()-start):>3}s code={s.code}")
            last = s.code
        if s.finished:
            return s
        time.sleep(10)
    return job.get_status()


def main(seq_len=8, d_model=256, n_heads=4, head_dim=64, name="attn_w4a16_small"):
    print(f"=== build attention block ONNX ===")
    m = build_attention_block_onnx(seq_len, d_model, n_heads, head_dim)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(m, f.name)
        onnx_path = f.name
    print(f"ONNX: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")

    # Build calibration data — a handful of realistic input samples.
    # AI Hub's quantize uses these to determine activation ranges.
    print("\n=== build calibration data (8 random samples) ===")
    rng = np.random.default_rng(seed=42)
    cal_inputs = [
        rng.standard_normal(size=(1, seq_len, d_model)).astype(np.float32)
        for _ in range(8)
    ]
    cal_data = {"x": cal_inputs}

    print(f"\n=== submit quantize (w4a16) ===")
    qj = hub.submit_quantize_job(
        model=onnx_path,
        calibration_data=cal_data,
        weights_dtype=hub.QuantizeDtype.INT4,
        activations_dtype=hub.QuantizeDtype.INT16,
        name=f"{name}_quantize",
    )
    print(f"quantize job: {qj.job_id}  url: {qj.url}")
    poll(qj, "quantize", max_seconds=300)
    if not qj.get_status().success:
        print(f"QUANTIZE FAILED: {qj.get_status().message}")
        return 1

    qmodel = qj.get_target_model()
    print(f"quantized model handle: {qmodel}")

    print(f"\n=== submit compile (quantized -> qnn_context_binary) ===")
    dev = next(d for d in hub.get_devices() if d.name == "Samsung Galaxy S22 Ultra 5G")
    cj = hub.submit_compile_job(
        model=qmodel, device=dev,
        name=f"{name}_compile",
        options="--target_runtime qnn_context_binary",
    )
    print(f"compile job: {cj.job_id}  url: {cj.url}")
    poll(cj, "compile", max_seconds=300)
    if not cj.get_status().success:
        print(f"COMPILE FAILED: {cj.get_status().message}")
        return 1

    out_dir = os.environ.get("SP_AIHUB_OUT_DIR", "/tmp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"v69_{name}.bin")
    cj.get_target_model().download(filename=out_path)
    print(f"\nartifact: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

    # Optional profile
    print(f"\n=== submit profile ===")
    pj = hub.submit_profile_job(model=cj.get_target_model(), device=dev,
                                name=f"{name}_profile")
    print(f"profile job: {pj.job_id}  url: {pj.url}")
    poll(pj, "profile", max_seconds=300)
    if pj.get_status().success:
        r = pj.download_profile()
        ex = r.get("execution_summary", {})
        for k in ('estimated_inference_time','first_load_time','warm_load_time'):
            print(f"  {k}: {ex.get(k)}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--head_dim", type=int, default=64)
    ap.add_argument("--name", default="attn_w4a16_small")
    args = ap.parse_args()
    sys.exit(main(args.seq, args.d_model, args.n_heads, args.head_dim, args.name))
