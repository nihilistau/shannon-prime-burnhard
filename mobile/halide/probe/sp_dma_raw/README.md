# Mode D Stage 1 — SP-packed-bytes UBWCDMA probe

This is the first of three probes documented in
[`../../mode_d_isp_probe_design.md`](../../mode_d_isp_probe_design.md).

## What this answers

> Does the Halide UBWCDMA RAW path on V69 accept non-camera-source
> bytes as input *and* fp32 RAW as output, end-to-end, on a real
> Samsung Galaxy S22 Ultra?

If yes → the foundation for Mode C (Halide-on-DMA + HTP matmul) is
proven, and Stage 2/3 (UBWC compression on the bus / CV-ISP MFNR
handle) are reachable. If no → we hit a layer earlier than expected
and fall back to a plain rpcmem write path for the output (slower,
but still functional).

## Layout (mirrors the Halide SDK's `dma_raw_blur_rw_async` example)

```
sp_dma_raw/
├── halide/sp_dma_raw_generator.cpp    # 2D Halide generator (per-band dequant)
├── dsp/sp_dma_raw_run.cpp             # cDSP-side FastRPC entry
├── host/main.cpp                      # ARM Android host driver + verify
├── rpc/sp_dma_raw.idl                 # FastRPC interface
└── Makefile                           # SDK Makefile.common/rules wrapper
```

## Build sequence (validated path from `../../Makefile` notes)

This expects to live *inside* the Halide SDK example tree at runtime —
same as the upstream `dma_raw_blur_rw_async` build flow. The intended
deployment is to symlink (or copy) this directory into:

```
C:\Qualcomm\HALIDE_Tools\2.4.07\Halide\Examples\standalone\device\apps\sp_dma_raw\
```

…then run from a Hexagon SDK environment shell:

```
setup_sdk_env.cmd            (sets HALIDE_ROOT, HEXAGON_SDK_ROOT, etc.)
cd ...\apps\sp_dma_raw
mingw32-make
```

Outputs (mirrors what the dma_raw_blur_rw_async build produces):
- `host/test-sp-dma-raw`              (ARM Android binary)
- `dsp/libsp_dma_raw_skel.so`         (cDSP skel)
- `host/libsp_dma_raw.so`             (FastRPC stub)

## Run on device (S22 Ultra)

```
adb push test-sp-dma-raw libsp_dma_raw.so libsp_dma_raw_skel.so /data/local/tmp/sp_dma_raw_probe/
adb shell
cd /data/local/tmp/sp_dma_raw_probe
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./test-sp-dma-raw 1000
```

Expected pass output (from `host/main.cpp::main`):
```
band 0 max_err = 0.000e+00
band 1 max_err = 0.000e+00
band 2 max_err = 0.000e+00
band 3 max_err = 0.000e+00
overall: max_err=0.000e+00 rms=0.000e+00 mismatches=0/128
DSP avg_time = ... us over 1000 iterations
[Mode] Device Standalone [Result] Pass
```

Expected fail output (probe FAILED — captured for triage):
```
PROBE FAIL: sp_dma_raw_run returned <code>
Most likely culprits in order of probability:
  1. prepare_for_copy_to_device rejects fp32 RAW (output type)
  2. prepare_for_copy_to_host rejects packed-bytes RAW (input layout)
  3. UBWCDMA descriptor issue (frame size / stride alignment)
  4. ION allocation alignment (need 128-byte aligned for HVX)
Check FARF logs from DSP for the failing call site.
```

If failure mode 1 fires (most likely), the workaround is to allocate
output as `uint8` 4× wider and reinterpret on the host. That doesn't
unblock Stage 3 (CVP MFNR — likely also fp16/int16 only) but it does
mean Mode C streaming is still on the table.

## Stage 2 add-on (if Stage 1 passes)

Re-run with `is_input_ubwc=1` in `host/main.cpp`. Should require zero
generator/dsp changes — UBWC is a transport-level concern. Measures
whether stacking SP compression + UBWC compression on the bus gives
the bonus 1.3-2× bandwidth multiplier the design doc projects.

## Stage 3 (separate scaffold — to be created if Stage 1+2 pass)

`backends/halide/probe/sp_isp_mfnr/` will reuse this directory's
packing code but call `cvpHandle_create` / `cvpMfnr_*` instead of
the Halide pipeline. CVP API surface lookup: see Stage 3 of the
parent design doc.
