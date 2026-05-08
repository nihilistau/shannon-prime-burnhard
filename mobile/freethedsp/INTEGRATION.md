# freethedsp integration — how every backend picks up the shim

## Default-loaded, opt-in-active

The harness is designed so the shim is *always linked* into every
on-device run path, but only *activates* when `SP_FREETHEDSP=1`. With
the env var unset, the LD_PRELOAD shim does nothing — single ioctl
pass-through, no patching. With the env var set, it patches
`is_test_enabled()` in the cDSP shell on first FastRPC handle open.

This decouples "having the shim available" (always true) from
"using the shim" (per-run, controlled by env). Means we never have to
remember to push it for any specific backend or experiment.

## Backends that pick it up

### `backends/halide/build-example.ps1` `-Run` flag

When invoked with `-Run`, the script does an `adb push` of the
artifact bundle and an `adb shell` to launch. We add `libfreethedsp.so`
to the push set and prepend `LD_PRELOAD=./libfreethedsp.so` to the
shell command line. Activation gated on env: if the calling shell has
`SP_FREETHEDSP=1` set, that propagates through `adb shell env -i ...`.

Concretely the run line becomes:
```
adb shell "cd /data/local/tmp/<test>; chmod 755 main-<test>.out; \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  LD_PRELOAD=./libfreethedsp.so SP_FREETHEDSP=${SP_FREETHEDSP:-0} \
  ./main-<test>.out <args>"
```

### `backends/hexagon/scaffold/build.cmd`

The production shannon-prime-hexagon FastRPC build. Same pattern: push
`libfreethedsp.so` next to `libsp_hex.so`, and any `adb shell` runner
script (manual `sp_hex_main` invocations, our existing `compress_f32`
benchmarks) wraps with `LD_PRELOAD=./libfreethedsp.so`.

### `backends/qnn_aihub/` Phase 2 on-device runs

The QNN-on-device runner (`qnn-net-run`) goes through FastRPC for HTP
context-binary loading. Adding `libfreethedsp.so` to the push bundle
and `LD_PRELOAD` to the shell command means if QNN ever needs a
permission that's signed-only, the shim covers it without us having to
diagnose the failure first.

### `shannon-prime-llama` integration

When we wire QNN execute() as the matmul kernel inside
shannon-prime-llama (Phase 2.3), the same LD_PRELOAD applies to the
`llama-cli` binary. Since the cDSP context is process-scoped, the
shim's patch propagates to every FastRPC handle the llama process
opens — both the SP-Hexagon bridge and the QNN HTP backend.

## When `SP_FREETHEDSP=1` is required

- **Mode D Stage 1** Halide DMA engine allocation (validated 2026-05-01
  to fail without it).
- **Mode D Stage 3** CV-ISP `cvpHandle_create` (presumed gated; not yet
  probed).
- Any QNN HTP API surface that requires test-mode permission (TBD —
  most don't; we'll know if Phase 2 surfaces one).
- Custom hardware coprocessor entitlements (VTCM sharing in non-default
  modes, NSP-direct memory access, etc).

## When `SP_FREETHEDSP=1` is NOT required

- The existing `shannon-prime-hexagon` rpcmem + FastRPC + HVX path.
  This works in unsigned PD without freethedsp and ships every day.
- Standard QNN HTP execution of compiled context binaries (almost
  certainly — Qualcomm's own runtime doesn't need test-mode for normal
  graph execution, otherwise their commercial SDK wouldn't deploy).
- The Halide-on-CPU Hexagon target (no DMA, no signed-PD-gated APIs).

## Drift detection

The shim refuses to patch if the bytes at `PATCH_ADDR` don't match
`PATCH_OLD`. The error message is explicit:
```
[freethedsp] expected bytes at PATCH_ADDR 0x... do not match.
             Found: AA BB CC DD
             ... Re-run Phase D.1 + D.2 to find the new offset.
```
That's our drift signal — happens when Samsung pushes a OneUI update
that ships a different `fastrpc_shell_3`. Recovery: re-run discover.so
+ find_is_test_enabled.py, update the constants, rebuild.

## Why we don't gate every backend on `SP_FREETHEDSP=1` by default

Two reasons:
1. **Forensic clarity.** When a future bench shows a regression, we
   want to know whether the shim was active so we can rule it in or
   out. Default-on hides a load-bearing variable.
2. **Defense in depth.** If a future Samsung OTA detects the page
   modification (currently they don't, but the threat model could
   change), default-off means we only burn the shim's stealth budget
   on the experiments that actually need it.

Tradeoff is one extra env var on the run line for experiments that do
need it. Worth it.
