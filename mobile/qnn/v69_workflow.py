"""
Shannon-Prime: validated V69 + QNN HTP compile + profile workflow via Qualcomm AI Hub.

Submits a small reference matmul ONNX to Qualcomm AI Hub, targets
"Samsung Galaxy S22 Ultra 5G" (sm8450 / V69), compiles to QNN context
binary, profiles on a real lab device, returns timing data.

VALIDATED 2026-05-01 (jobs j5wm6m34g compile + jp4jrjd2p profile):
  - 256x256 fp32 MatMul -> 81 µs/inference on V69 NPU/HTP
  - Run on a real Samsung Galaxy S22 Ultra 5G in Qualcomm's lab
  - Stable across 100 trials (83-87 µs steady-state, one 435 µs warmup)

Pre-requisites:
  pip install qai-hub onnx
  qai-hub configure --api_token <YOUR_AIHUB_API_TOKEN>

Notes on what tripped me up the first time:
  - AI Hub's ONNX checker rejects ir_version > 12. Our local onnx
    package defaults to ir_version=13, so we must explicitly set
    model.ir_version = 8 (works with opset 13).
  - --target_runtime qnn_context_binary is the option that produces
    a V69 HTP-runnable .bin. The Genie pre-built artifacts use the
    same format.

Usage:
  python v69_workflow.py            # submit + poll + download
  python v69_workflow.py --check    # poll status of an existing job

Returns:
  /tmp/v69_<name>.bin               # compiled QNN context binary
  Profile data printed to stdout (latency, memory, op breakdown).
"""
import argparse, os, sys, tempfile, time

try:
    import onnx
    from onnx import helper, TensorProto
    import qai_hub as hub
except ImportError as e:
    print(f"Missing dep: {e}. Run: pip install qai-hub onnx")
    sys.exit(1)


def build_tiny_matmul_onnx(N=256):
    """N×N fp32 MatMul. Tiny upload (256 KB at N=256), gives a clean
    perf data point on V69's tensor accelerator."""
    W = helper.make_tensor("W", TensorProto.FLOAT, [N, N], [0.01] * (N * N))
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, N])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, N])
    node = helper.make_node("MatMul", ["x", "W"], ["y"])
    graph = helper.make_graph([node], "tiny_matmul", [x], [y], initializer=[W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    # AI Hub checker rejects ir_version > 12. Pin to 8 (mapped from opset 13).
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model


def submit_and_run(name="sp_v69_baseline", N=256, device_name="Samsung Galaxy S22 Ultra 5G"):
    model = build_tiny_matmul_onnx(N)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        path = f.name
    print(f"ONNX: {path} ({os.path.getsize(path) / 1024:.1f} KB)")

    dev = next(d for d in hub.get_devices() if d.name == device_name)
    print(f"Target: {dev.name}, OS={dev.os}")

    print("\n--- compile ---")
    cjob = hub.submit_compile_job(
        model=path, device=dev, name=f"{name}_compile",
        options="--target_runtime qnn_context_binary",
    )
    print(f"compile job: {cjob.job_id}  url: {cjob.url}")
    poll_until_done(cjob)
    if not cjob.get_status().success:
        print("Compile FAILED:")
        print(f"  {cjob.get_status().message}")
        return None

    print("\n--- download .bin ---")
    # Honour SP_AIHUB_OUT_DIR for Windows-host runs; default to /tmp
    # for legacy sandbox compatibility.
    out_dir = os.environ.get("SP_AIHUB_OUT_DIR", "/tmp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"v69_{name}.bin")
    cjob.get_target_model().download(filename=out_path)
    print(f"compiled artifact: {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

    print("\n--- profile (runs on real lab device) ---")
    pjob = hub.submit_profile_job(
        model=cjob.get_target_model(), device=dev,
        name=f"{name}_profile",
    )
    print(f"profile job: {pjob.job_id}  url: {pjob.url}")
    poll_until_done(pjob)
    if not pjob.get_status().success:
        print("Profile FAILED:")
        print(f"  {pjob.get_status().message}")
        return None

    print("\n--- profile data ---")
    r = pjob.download_profile()
    ex = r.get("execution_summary", {})
    print(f"  estimated_inference_time:    {ex.get('estimated_inference_time'):>10} µs")
    print(f"  first_load_time:             {ex.get('first_load_time'):>10} µs")
    print(f"  warm_load_time:              {ex.get('warm_load_time'):>10} µs")
    print(f"  peak_memory_inference:       {ex.get('estimated_inference_peak_memory'):>10} bytes")
    times = ex.get("all_inference_times", [])
    if times:
        sorted_t = sorted(times[5:])  # drop warmup
        median = sorted_t[len(sorted_t) // 2]
        print(f"  median over {len(sorted_t)} runs:        {median:>10} µs")
    print()
    print("  ops:")
    for op in r.get("execution_detail", []):
        unit = op.get("compute_unit", op.get("runtime_unit", "?"))
        print(f"    {op.get('op_name', op.get('name', '?'))[:30]:30} | {unit}")
    return r


def poll_until_done(job, max_seconds=300):
    start = time.time()
    last_state = None
    while time.time() - start < max_seconds:
        s = job.get_status()
        state = s.code
        if state != last_state:
            print(f"  t={int(time.time() - start):>3}s: {state}")
            last_state = state
        if s.finished:
            return
        time.sleep(8)
    raise RuntimeError(f"Job {job.job_id} did not finish in {max_seconds}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", help="job_id to poll (skip submission)")
    ap.add_argument("--name", default="sp_v69_baseline")
    ap.add_argument("--n", type=int, default=256, help="matmul dim N (NxN)")
    args = ap.parse_args()

    if args.check:
        job = hub.get_job(args.check)
        s = job.get_status()
        print(f"job {args.check}: {s.code} (success={s.success}, finished={s.finished})")
        if s.success:
            # If it's a compile job, re-download the .bin to SP_AIHUB_OUT_DIR.
            try:
                tm = job.get_target_model()
                if tm is not None:
                    out_dir = os.environ.get("SP_AIHUB_OUT_DIR", "/tmp")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"v69_{args.name}.bin")
                    tm.download(filename=out_path)
                    print(f"re-downloaded artifact: {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")
            except Exception as e:
                print(f"(no target model on this job: {e})")
            # If it's a profile job, dump the perf summary.
            if hasattr(job, "download_profile"):
                try:
                    r = job.download_profile()
                    print(r.get("execution_summary", {}))
                except Exception:
                    pass
    else:
        submit_and_run(name=args.name, N=args.n)
