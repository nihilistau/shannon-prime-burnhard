"""
Phase 2.2 driver — submit attention-block ONNX to Qualcomm AI Hub for
V69 HTP compile + profile, mirroring v69_workflow.py's flow.
"""
from __future__ import annotations
import argparse
import os
import sys
import tempfile
import time

import qai_hub as hub
import onnx

from attention_block import build_attention_block_onnx


def poll(job, label, max_seconds=600):
    start = time.time()
    last = None
    while time.time() - start < max_seconds:
        s = job.get_status()
        if s.code != last:
            print(f"  [{label}] t={int(time.time()-start):>3}s: {s.code}")
            last = s.code
        if s.finished:
            return s
        time.sleep(8)
    raise RuntimeError(f"{label} {job.job_id} did not finish in {max_seconds}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=2048)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--name", default=None,
                    help="job name suffix; defaults to 'attn_<seq>_<d>_<h>x<hd>'")
    ap.add_argument("--device", default="Samsung Galaxy S22 Ultra 5G")
    ap.add_argument("--profile", action="store_true",
                    help="also submit a profile job after compile")
    args = ap.parse_args()

    name = args.name or f"attn_{args.seq}_{args.d_model}_{args.n_heads}x{args.head_dim}"

    print(f"=== build attention block ONNX ===")
    model = build_attention_block_onnx(
        seq_len=args.seq, d_model=args.d_model,
        n_heads=args.n_heads, head_dim=args.head_dim,
    )
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        path = f.name
    sz_mb = os.path.getsize(path) / 1024 / 1024
    print(f"ONNX: {path} ({sz_mb:.1f} MB)")

    dev = next(d for d in hub.get_devices() if d.name == args.device)
    print(f"Target: {dev.name}, OS={dev.os}")

    print(f"\n=== submit compile job ({name}) ===")
    cjob = hub.submit_compile_job(
        model=path, device=dev, name=f"{name}_compile",
        options="--target_runtime qnn_context_binary",
    )
    print(f"compile job: {cjob.job_id}  url: {cjob.url}")
    poll(cjob, "compile")
    s = cjob.get_status()
    if not s.success:
        print(f"COMPILE FAILED: {s.message}")
        return 1

    out_dir = os.environ.get("SP_AIHUB_OUT_DIR", "/tmp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"v69_{name}.bin")
    cjob.get_target_model().download(filename=out_path)
    sz_kb = os.path.getsize(out_path) / 1024
    print(f"compiled artifact: {out_path} ({sz_kb:.1f} KB)")

    if args.profile:
        print(f"\n=== submit profile job ===")
        pjob = hub.submit_profile_job(
            model=cjob.get_target_model(), device=dev,
            name=f"{name}_profile",
        )
        print(f"profile job: {pjob.job_id}  url: {pjob.url}")
        poll(pjob, "profile")
        ps = pjob.get_status()
        if ps.success:
            r = pjob.download_profile()
            ex = r.get("execution_summary", {})
            print(f"  estimated_inference_time: {ex.get('estimated_inference_time')} µs")
            print(f"  first_load_time:          {ex.get('first_load_time')} µs")
            times = ex.get("all_inference_times", [])
            if times:
                steady = sorted(times[5:])
                print(f"  median over {len(steady)}: {steady[len(steady)//2]} µs")
        else:
            print(f"PROFILE FAILED: {ps.message}")

    print(f"\n=== summary ===")
    print(f"  compile job: {cjob.job_id}")
    print(f"  artifact:    {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
