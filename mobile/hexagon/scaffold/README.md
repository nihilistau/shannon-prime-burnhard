# Shannon-Prime Hexagon DSP scaffold

Self-contained FastRPC project that compiles to the cDSP on Snapdragon and exercises the SP VHT2 round-trip on real hardware. Forked from the canonical Qualcomm `calculator`/S22U sample shape because that's the layout `qaic` and `build_cmake` expect; the topology is preserved verbatim, only the targets and the IDL contents differ.

This is **not** wired into the engine build (the engine still links `backends/hexagon/shannon_prime_hexagon.{h,c}`, the host-side stub). The scaffold builds independently — the goal is to get a working DSP-side library in isolation, then graft its kernels into the engine path once they're real.

## Layout

```
scaffold/
  build.cmd            # CLI build wrapper (defaults DSP_ARCH=v69)
  CMakeLists.txt       # HLOS+Hexagon split; build_idl + link_custom_library
  inc/
    sp_hex.idl         # FastRPC interface (qaic input)
  src_app/             # ARM-side
    sp_hex_main.c      # CLI + arg parsing
    sp_hex_ext.{c,h}   # rpcmem alloc + FastRPC handshake + smoke test
  src_dsp/             # DSP-side
    sp_hex_imp.c       # IDL method impls + open/close lifecycle
    sp_hex_kernels.{c,h}  # scalar reference VHT2 (HVX kernels go in a sibling .c)
```

## What it does today

- Open/close lifecycle on the cDSP via `remote_handle64`.
- `sp_hex_round_trip_f32(in, head_dim, out)` — applies VHT2 forward twice on the DSP. VHT2 is self-inverse, so `out ≈ in` to fp32 epsilon. The ARM-side smoke test verifies max-abs-error < 1e-4.
- `sp_hex_vht2_forward(data, n)` — exposes the butterfly directly for benchmarking.
- `sp_hex_band_quantize` / `sp_hex_band_dequantize` — IDL slots are wired through to scalar reference *stubs* that currently return -1. Wire to the math core (or implement on-DSP) when band-IO smoke tests are needed.

## What's intentionally not done

- HVX intrinsics. The scalar VHT2 in `sp_hex_kernels.c` is the reference. Once perf work begins, drop a sibling `sp_hex_kernels_hvx.c` gated on `#ifdef __HEXAGON_HVX__` and switch the imp file to call the HVX path when the macro is defined.
- HMX (V73+ tensor cores). Out of scope until V73-class hardware is in the validation rotation.
- Banded quantize/dequantize on-DSP. Stubs return -1 today.
- CI integration. The scaffold builds locally only; no GH Actions runner has the Hexagon SDK.

## Build

```cmd
cd backends\hexagon\scaffold
build.cmd            REM both DSP + HLOS (default)
build.cmd dsp        REM DSP skel only
build.cmd android    REM ARM exe + stub only
build.cmd sim        REM build + run on Hexagon simulator
build.cmd clean
```

`build.cmd` auto-sources `setup_sdk_env.cmd` once per shell. Defaults `DSP_ARCH=v69` (S22 Ultra), `BUILD_TYPE=Debug`, `HLOS_ARCH=64`. Override via env vars: `set DSP_ARCH=v75 && build.cmd`.

Outputs:
- `hexagon_Debug_toolv87_v69/ship/libsp_hex_skel.so` — DSP-side, push to phone
- `android_Debug_aarch64/ship/sp_hex` — ARM-side executable, push to phone
- `android_Debug_aarch64/ship/libsp_hex.so` — ARM-side FastRPC stub, push to phone

## Push + run on phone

```powershell
$adb = 'D:\Files\Android\pt-latest\platform-tools\adb.exe'
$dst = '/data/local/tmp/sp_hex'
& $adb shell "mkdir -p $dst"
& $adb push hexagon_Debug_toolv87_v69\ship\libsp_hex_skel.so $dst/
& $adb push android_Debug_aarch64\ship\libsp_hex.so          $dst/
& $adb push android_Debug_aarch64\ship\sp_hex                $dst/
& $adb shell "chmod +x $dst/sp_hex"
& $adb shell "cd $dst && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./sp_hex -n 128"
```

Expected output:

```
[sp_hex] Shannon-Prime Hexagon DSP scaffold smoke test
[sp_hex] Domain: 3  PD: unsigned  head_dim: 128
[sp_hex] calling round_trip_f32 on the DSP...
[sp_hex] head_dim=128   max_abs_err=<small>   in[0]=0.125000   out[0]=0.125000
[sp_hex] Success
```

`max_abs_err` will be on the order of 1e-7 (fp32 epsilon × log2(N)) for the VHT2 self-inverse round trip.

## Forking this into a real backend

When the kernels are ready to graduate, the path is:
1. Fold the IDL methods into `shannon-prime/backends/hexagon/shannon_prime_hexagon.h`'s API surface (currently the host-side stub).
2. Move the DSP-side .c files into a `dsp/` subdirectory under `backends/hexagon/` that the engine's CMake target invokes.
3. Replace `sp_hex_kernels.c` (scalar) with `sp_hex_kernels_hvx.c` (vectorized).
4. Rip out the standalone scaffold `CMakeLists.txt` and `build.cmd` once the engine build owns the DSP-side compile.

Until then: the scaffold is the iteration loop. Kernel changes ship here first, get validated on the phone, then graduate.
