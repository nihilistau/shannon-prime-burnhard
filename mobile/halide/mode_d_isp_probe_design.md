# Mode D ISP Probe — Bayer-Pack CVP Handle Test

**Goal:** binary determination of whether the Spectra 680 MFNR / CV-ISP
pipeline is accessible to non-camera-source buffers. If yes, Mode D
(ISP as 18-bit spectral fusion engine) is reachable. If no, we fall
back to Mode C (Halide-on-DMA + HTP matmul) which is still a major
architectural lift.

## What we're packing

A single SP-encoded K row at hd=128 with the 4-band {5,5,4,3} config:
- Total: 76 bytes
- Layout: 4 × (2-byte fp16 scale + N×bits/8 packed bytes)

For ISP consumption, we pack this as 4 "frames" (one per band), each:
- Width: hd elements treated as "pixels"
- Height: 1 row (1D image)
- Format: `eDmaFmt_RawData` (per Halide DMA example) — most permissive
  format that doesn't impose Bayer-pattern layout constraints

## Probe sequence

### Stage 1 — Halide DMA via `eDmaFmt_RawData` (proven path)

This is reachable today using `dma_raw_blur_rw_async` as the template.
Modify the dsp/_run.cpp to:

1. Allocate ION buffer with `halide_hexagon_dma_device_wrap_native()`
2. Configure as RAW frame: `eDmaFmt_RawData`, w=128, h=4, stride=128
3. Run our `sp_band_dequantize_generator` (already committed at
   `backends/halide/sp_band_dequantize_generator.cpp`) as the kernel
4. Compare output to scalar `sp_band_dequantize` reference

**Expected:** works. Validates Mode C streaming path. Lets us measure
DMA throughput from rpcmem-backed RAW input → HVX-resident VTCM.

### Stage 2 — UBWC compression on the bus (bonus)

Same as Stage 1 but with `getUBWCFrameSize` instead of `getLinearFrameSize`.
Tells us whether stacking SP compression + UBWC compression on the same
weight stream gives us a 2nd-order bandwidth multiplier. (SP gives 6×;
UBWC typically adds 1.3-2× on top depending on entropy.)

**Expected:** works. The UBWC engine doesn't care what the bytes mean.

### Stage 3 — CVP handle for MFNR (the real probe)

Locate the CVP API surface. Likely candidates:
- `C:\Qualcomm\HALIDE_Tools\2.4.07\Halide\include\HalideRuntimeHexagonHost.h`
  — primary candidate; if `cvp_*` functions appear here, easy entry.
- `C:\Qualcomm\Hexagon_SDK\5.5.6.0\incs\` — search for `cvp.h`,
  `cvp_handle.h`, `eva.h`.
- `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\include\` — QNN's CV-ISP
  bindings if any.
- Worst case: separate Snapdragon Camera SDK (not currently installed)
  or extracted from `vendor/lib/libcvp.so` on the device itself.

Once located, the probe is:

```c
cvp_handle h;
cvp_status rc = cvpHandle_create(&h, /*config*/);
//  -> if rc != CVP_OK at this stage: Mode D blocked at API level
//     (probably UNSUPPORTED_OPERATION or SECURE_REGION_VIOLATION).
//     Stop here; fall back to Mode C.

// Configure MFNR-equivalent: 4 frames, weighted accumulation.
cvp_mfnr_config cfg = {
    .frame_count   = 4,
    .precision     = CVP_18BIT,
    .alignment     = CVP_INDEX_ALIGNED,    // no motion compensation
    .frame_weights = {1.0f, scale_b1/scale_b0, ...},
};
rc = cvpMfnr_configure(h, &cfg);
//  -> BAD_PARAMETER on frame_count: minimum may be larger; pad with
//     zero-weight frames OR run nested 2-frame fusions.
//  -> BAD_PARAMETER on alignment: motion vectors may be required;
//     supply zero-vector with matching format.
//  -> BAD_PARAMETER on precision: 18-bit might be camera-only; try
//     CVP_INT16 or CVP_FP16.

// Submit our 4 SP bands (already RAW-formatted from Stage 1).
rc = cvpMfnr_submit(h, packed_band_buffers, 4);
rc = cvpHandle_wait(h);
//  -> success here = Mode D works. Read output, compare to scalar
//     reference, record per-element error and throughput.
```

## Decision tree from probe outcome

| Stage 1 | Stage 2 | Stage 3 | Decision |
|---|---|---|---|
| ✓ | — | — | Mode C with Halide-DMA is real; commit to that path |
| ✓ | ✓ | — | Mode C + UBWC = 1.5-2× bandwidth on weight streaming |
| ✓ | ✓ | ✓ | Mode D works; offload band fusion to ISP, free HVX cycles |
| ✓ | ✓ | UNSUPPORTED | Mode D blocked, ship Mode C, document as research path |
| ✗ | — | — | Halide DMA itself broken on V69; deeper investigation |

## What Stage 3 tells us regardless of outcome

If `BAD_PARAMETER` at config but Stage 1+2 work, we know the silicon is
reachable but our packing is wrong — we iterate on the packing format
until MFNR accepts it. This is an engineering loop, not an architecture
question.

If `UNSUPPORTED_OPERATION` at handle creation, the camera HAL is gating
us. We could still potentially reach MFNR via:
- Reverse-engineering the camera service IPC (Snapdragon Camera HAL is
  Android open-source; userspace bridge is documented)
- Using QNN's CV-ISP backend if it has a non-camera path
- Asking Qualcomm AI Hub support team directly (they may have a
  developer-mode flag we don't have docs for)

These are research moves, not engineering moves. Mode C ships first.

## Build path

1. Copy `dma_raw_blur_rw_async/` template to `backends/halide/probe/`
2. Replace `dma_raw_blur_rw_async_generator.cpp` with our existing
   `sp_band_dequantize_generator.cpp`
3. Modify `host/main.cpp` to allocate SP-formatted RAW buffer
4. Run on phone: `adb push test-dma; adb shell ./test-dma`
5. Output: pass/fail + throughput numbers
6. If Stage 1 passes, write Stage 3 probe similarly using `cvp_*` API
   (location TBD per Stage 3 above)

## Estimated effort

- Stage 1 + 2: 1-2 days. Standard Halide DMA pattern, well-documented.
- Stage 3 (locate APIs): 0.5-2 days. Highly variable depending on
  whether CVP headers are present.
- Stage 3 (run probe): 0.5 days once APIs found.

Total: 2-4 days for the binary "Mode D works on V69" answer.

## Pre-condition for this work

Mode C must already be running with HTP as the matmul executor. The
ISP win is a *delta* on top of Mode C — it shaves the dequant glue
overhead. If Mode C isn't validated first, we don't have a baseline
to measure the ISP delta against.
