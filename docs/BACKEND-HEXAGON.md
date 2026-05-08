# Backend: Qualcomm Hexagon DSP

This document is the implementation roadmap for the Hexagon DSP backend. The header (`backends/hexagon/shannon_prime_hexagon.h`) declares the API surface; the .c file is a scaffolding stub that returns "unavailable" so bridges fall back cleanly until the kernels exist.

Target hardware: Snapdragon 8 Gen 1+ (Hexagon V69 with HVX). Primary validation device: Samsung Galaxy S22 Ultra. SDK requirement: Qualcomm Hexagon SDK 5.x with `toolv87` and DSP_ARCH=v69.

---

## What Hexagon brings to Shannon-Prime

The Hexagon DSP is a third compute resource on Snapdragon SoCs alongside the ARM CPU cores and the Adreno GPU. Three things make it interesting for SP:

- **Always-on, low-power.** The DSP runs at ~500mW even under load. Compression / decompression of KV cache vectors can happen continuously without affecting CPU/GPU thermal budget. Important for sustained-load deployments where the phone otherwise thermal-throttles.

- **HVX vector instructions.** 1024-bit SIMD with int8/int16/fp16 pipes. The banded quantize/dequantize loops are exactly what HVX is designed for: load a vector of int8 values, apply scale, write fp16. Theoretical throughput on V69: 128 int8 ops/cycle.

- **Shared physical memory with CPU/GPU.** The DSP, ARM, and Adreno all access the same DRAM through the SoC's SMMU. Round-trips don't need PCIe-class transfers; they happen through page tables. The "stack and pop" memory pattern from earlier phone work is implemented natively in the SoC's memory subsystem.

What it doesn't bring on V69: HMX matrix instructions (those are V73+, Snapdragon 8 Gen 2+). The VHT2 butterfly will run on HVX scalar+vector ops on V69, then can pivot to HMX matmul on later hardware.

---

## Where the SDK pieces live

Standard install layout for Hexagon SDK 5.5.6.0 (Windows):

```
C:\Qualcomm\Hexagon_SDK\5.5.6.0\
  setup_sdk_env.cmd            <- source from a fresh cmd shell to set
                                  HEXAGON_SDK_ROOT, DEFAULT_DSP_ARCH,
                                  PATH, etc. Use this as the entry point
                                  for any build session.
  build/                       <- CMake/make machinery for FastRPC stubs
  examples/                    <- canonical sample projects
    calculator/                <- "hello world" of FastRPC (CPU<->DSP IPC)
    qhl_hvx/                   <- HVX intrinsics demo (the closest analog
                                  to what SP needs)
    multithreading/            <- DSP thread management
  incs/                        <- Hexagon-specific headers (HAP_*, etc.)
  ipc/                         <- FastRPC plumbing
  libs/                        <- Qualcomm-supplied DSP libraries
  tools/
    HEXAGON_Tools/8.7.06/      <- the actual compiler (hexagon-clang) +
                                  HVX/HMX intrinsics headers
    android-ndk-r25c/          <- ARM-side build chain
    qhcg/                      <- Hexagon Graph Compiler (high-level)
    cmake-3.22.2-windows-x86_64/  <- bundled CMake (no need to install
                                  separately)
    wrapperTools/              <- qaic, etc.
```

Auxiliary Qualcomm tooling that does NOT need to be installed for SP work, in case it's already on the system:

- `C:\Program Files (x86)\Qualcomm\QPM3` - Qualcomm Package Manager. Just the installer; not needed at build/run time.
- `C:\Program Files\Qualcomm\QUTS` - Qualcomm Universal Trace System. Useful for performance profiling later, not for getting started.
- `%LOCALAPPDATA%\Qualcomm\PCAT` - PC Configuration / Analysis Tool. Not relevant.

If any of those are missing, ignore. They're orthogonal to building DSP code.

---

## Setup checklist (one-time)

**Strong recommendation: use WSL2 with Ubuntu 20.04 LTS**, not native Windows. The Hexagon SDK is primarily Linux-targeted; the Windows build pipeline has multiple known issues (documented below). Most Qualcomm engineers develop on Ubuntu, so the SDK's Linux paths are well-trodden while the Windows paths trip on small bugs. WSL2 lives next to Windows, shares the filesystem via `/mnt/c`, and lets adb-over-USB or adb-over-WiFi reach the phone identically to native Windows.

If you must work on native Windows, see the "Windows setup gotchas" section below for the workarounds.

### WSL2 / Ubuntu 20.04 path (recommended)

#### One-time check: do you already have WSL + Ubuntu?

```powershell
wsl --status
wsl --list --verbose
```

If you see **Ubuntu-20.04 — Stopped — WSL2** (or similar) you're good. Skip to step 2. If not, install:

```powershell
wsl --install -d Ubuntu-20.04
# (one-time first launch will prompt for username + password)
```

#### Step 1 — Start the distro and update apt

```powershell
wsl -d Ubuntu-20.04
```

You should land at a `username@host:~$` prompt. Update apt and install the dependencies the Hexagon SDK install scripts assume:

```bash
sudo apt update
sudo apt install -y build-essential cmake make python3 python3-pip default-jdk \
    git curl wget xz-utils libncurses5 libncursesw5 lib32stdc++6 zlib1g lib32z1
```

`libncurses5` and `lib32stdc++6` are the legacy 32-bit libs the bundled Hexagon Tools sometimes need on modern Ubuntu. Worth installing once even though you may not hit the dependency until later milestones.

#### Step 2 — Download the Hexagon SDK Linux variant

Two ways:

**A. Via Qualcomm Package Manager (QPM3)** — recommended, handles the EULA flow automatically:

```bash
# Download from https://qpm.qualcomm.com (requires free Qualcomm account login)
# Pick "Hexagon SDK 5.5.6.0 Linux" - delivered as an installer .bin
chmod +x qualcomm_hexagon_sdk_5.5.6.0.bin
./qualcomm_hexagon_sdk_5.5.6.0.bin
# Default install path: ~/Qualcomm/Hexagon_SDK/5.5.6.0/
```

**B. Reuse the Windows install via /mnt/c** — *attractive in theory, doesn't work in practice* (verified empirically on this machine 2026-04-29). Keeping this section so you don't waste time re-discovering why:

```bash
# Inside WSL — what you'd LIKE to do:
export HEXAGON_SDK_ROOT="/mnt/c/Qualcomm/Hexagon_SDK/5.5.6.0"
ls $HEXAGON_SDK_ROOT/setup_sdk_env.source    # MISSING on Windows installs
```

What's actually in a Windows-installed SDK:

| Component                                | Bundled? | Notes                                  |
|------------------------------------------|----------|-----------------------------------------|
| `setup_sdk_env.cmd` / `_power.ps1`       | yes      | Windows-only entry points              |
| `setup_sdk_env.source` (Linux entry)     | **no**   | Not shipped with Windows SDK           |
| `tools/HEXAGON_Tools/8.7.06/Tools/bin`   | yes      | Contains `.exe` ONLY (e.g. `hexagon-clang.exe`) — no Linux ELF binaries |
| `ipc/fastrpc/qaic/WinNT/qaic.exe`        | yes      | Windows qaic                           |
| `ipc/fastrpc/qaic/Ubuntu20/qaic`         | yes      | Linux qaic — the *one* cross-OS binary  |

So although `qaic/Ubuntu20/qaic` is callable from WSL, the compiler (`hexagon-clang`) is shipped as Windows .exe only — there are no Linux ELF copies to run. The calculator Makefile also guards on `SDK_SETUP_ENV=Done`, which the missing `setup_sdk_env.source` is the only thing that sets, so even running `make` directly is blocked.

**Conclusion: Option B is documentation only. Use Option A (QPM3 download of the Linux SDK installer) for any real WSL work.** Plan budget: ~1 GB download, plus the `setup_sdk_env.source` ships with the Linux installer and doesn't have the broken-`make` problem the Windows path has.

#### Step 3 — Source the Linux env script

```bash
source $HEXAGON_SDK_ROOT/setup_sdk_env.source
```

This sets `HEXAGON_SDK_ROOT`, `DEFAULT_DSP_ARCH`, prepends the toolchain to PATH, and (unlike the Windows version) actually completes without trying to rebuild qaic.

Override the DSP arch for V69 (the S22 Ultra's chip):

```bash
export DEFAULT_DSP_ARCH=v69
```

#### Step 4 — Verify the toolchain

```bash
hexagon-clang --version           # should print QuIC LLVM Hexagon Clang version 8.7.06
qaic                              # should print qaic IDL compiler usage
make --version                    # should print GNU Make 4.x
```

If any of these "command not found", the env didn't take. Common cause: setup_sdk_env.source needs to be sourced (`source`/`.`), not executed (`./`). Re-run with `source`.

#### Step 5 — Build the calculator example

```bash
cd $HEXAGON_SDK_ROOT/examples/calculator
make tree V=hexagon_Release_dynamic_toolv87_v69
```

If both the ARM-side and DSP-side `.so` files produce without errors, the SDK is functional. Output goes to `hexagon_Release_toolv87_v69/ship/`.

#### Step 6 — Push to the S22 Ultra

You can use Windows-side adb from inside WSL by referencing it via `/mnt/c`:

```bash
alias adb='/mnt/c/Users/Knack/AppData/Local/Android/Sdk/platform-tools/adb.exe'
adb connect 192.168.8.110:42441   # or whatever port wireless ADB is using
adb devices                       # should show the SM-S908E
```

Or install adb inside WSL (`sudo apt install adb`) — both work; using the Windows one means you don't have two adb daemons running on different ports.

Push the built binaries:

```bash
# DSP-side .so goes to /vendor/lib/rfsa/dsp/sdk/ on the device
# Host-side .so goes wherever your Android app expects them
adb push hexagon_Release_toolv87_v69/ship/libcalculator_skel.so /data/local/tmp/
adb push hexagon_Release_toolv87_v69/ship/libcalculator.so /data/local/tmp/
adb shell ls -la /data/local/tmp/lib*.so
```

(Production deployment needs a signed DSP binary; dev mode allows the SDK's default test signature.)

For phone deployment from this point forward:
   - S22 Ultra in USB Debugging mode (or wireless ADB on a known port — current session was 192.168.8.110:42441, port rotates per pairing)
   - Optional: Qualcomm signature for the DSP binary if running outside dev mode (the SDK ships a default dev signature in `tools/elfsigner/`)

### Windows setup gotchas (if you must)

The following issues were observed on Hexagon SDK 5.5.6.0 + Windows 10/11. Workarounds inline.

1. **`setup_sdk_env.cmd` fails with "No rule to make target Ubuntu18/qaic".**
   The setup script tries to "install" qaic by running its Makefile, but the Makefile's Linux branch is hardcoded to Ubuntu18 (the SDK only ships Ubuntu20 + WinNT prebuilts), and the Windows branch isn't picked up because `make` (gow's bundled GNU Make 3.81) doesn't see `OS=Windows_NT` in some shell contexts.
   **Workaround**: pre-create `ipc/fastrpc/qaic/bin/` and copy `WinNT/qaic.exe` into it (and a copy named just `qaic` without extension, because some downstream rules drop the `.exe`). Skip the setup script, set env vars manually instead — see step 3 below.

2. **gow-0.8.0/bin/bash.exe is too minimal for the example Makefiles.**
   The bundled GNU-on-Windows shell ships bash but not `ls`, `head`, `grep`, etc. — those are separate `.exe` files in the same directory but they don't get picked up because the bash startup PATH is short. Result: every Makefile invocation prints `bash.exe: warning: could not find /tmp` and downstream commands fail with "command not found".
   **Workaround**: create `C:\tmp` (the warning goes away with `/tmp` resolvable). For missing utilities, install Git for Windows or MSYS2 and prepend that bash to PATH before gow's.

3. **`make tree V=...` silently exits.**
   The example Makefiles include shell expansions that break under gow's bash. PowerShell's stateless invocation model also doesn't help — env vars set in one command don't persist to the next.
   **Workaround**: assemble the entire env+make invocation as a single `cmd /c` call so all state lives in one shell.

4. **PowerShell `$env:HEXAGON_SDK_ROOT` doesn't substitute in `Set-Location` correctly when PATH is mid-mutation.**
   Race condition between PATH update and path resolution.
   **Workaround**: use literal paths or do the cd before mutating PATH.

If, after applying all of the above, you still hit issues — switch to WSL2. The Linux path "just works" with the SDK's primary support tier.

### Manual Windows env setup (if you've worked around all of the above)

```powershell
$sdk = "C:\Qualcomm\Hexagon_SDK\5.5.6.0"
$env:HEXAGON_SDK_ROOT = $sdk
$env:DEFAULT_HEXAGON_TOOLS_ROOT = "$sdk/tools/HEXAGON_Tools/8.7.06"
$env:DEFAULT_DSP_ARCH = "v69"
$env:DEFAULT_BUILD = "ReleaseG"
$env:DEFAULT_HLOS_ARCH = "64"
$env:DEFAULT_TOOLS_VARIANT = "toolv87"
$env:DEFAULT_NO_QURT_INC = "0"
$env:DEFAULT_TREE = "1"
$env:SDK_SETUP_ENV = "1"
$env:PATH = "$sdk\tools\HEXAGON_Tools\8.7.06\Tools\bin;" +
            "$sdk\ipc\fastrpc\qaic\bin;" +
            "$sdk\tools\utils\gow-0.8.0\bin;" +
            "$sdk\tools\cmake-3.22.2-windows-x86_64\bin;" +
            "$sdk\tools\wrapperTools;" +
            "$sdk\build\cmake\WinNT;" +
            "C:\Python311;" +
            $env:PATH

# Verify (these three should all succeed):
& hexagon-clang --version    # QuIC LLVM Hexagon Clang version 8.7.06
& qaic                       # qaic IDL compiler
& make --version             # GNU Make 3.81
```

---

## SP-specific architecture

Three layers compose end-to-end:

```
+---------------------+    +---------------------+    +---------------------+
| ARM Host (Android)  |    | FastRPC IPC         |    | Hexagon DSP V69     |
|---------------------|    |---------------------|    |---------------------|
| llama.cpp + bridge  |    | qaic-generated stub |    | qaic-generated skel |
| sp_hexagon_init()   |--->| sp_hexagon_stub.c   |--->| sp_hexagon_skel.c   |
| sp_hexagon_alloc()  |    |                     |    |                     |
| sp_hexagon_         |    | rpcmem_alloc()      |    | shared phys page    |
|   round_trip_k() ---|--->| FastRPC dispatch    |--->| HVX kernel:         |
|                     |    |                     |    |  promote fp16->fp32 |
|                     |    |                     |    |  VHT2 butterfly     |
|                     |    |                     |    |  Mobius reorder     |
|                     |    |                     |    |  band quantize/     |
|                     |    |                     |    |   dequantize        |
|                     |    |                     |    |  inverse VHT2       |
|                     |    |                     |    |  demote fp32->fp16  |
| out_fp16 in shared  |<---| return value        |<---| write back to       |
| mem - no copy       |    |                     |    | shared region       |
+---------------------+    +---------------------+    +---------------------+
```

The IDL file (`shannon_prime_hexagon.idl`, not yet written) declares the FastRPC interface. `qaic` consumes it and generates `sp_hexagon_stub.c` (linked into the Android app) and `sp_hexagon_skel.c` (linked into the DSP .so). Each function call goes through the stub, marshals arguments via FastRPC, executes the skel on the DSP, and unmarshals the return.

---

## Compute mapping

Which SP operations live where:

| Operation | Where it runs |
|---|---|
| `sp_band_quantize` (encode) | Hexagon HVX scalar + vector |
| `sp_band_dequantize` (decode) | Hexagon HVX scalar + vector |
| `sp_band_dequantize_partial` | Hexagon HVX, with band-count parameter |
| `sp_vht2_forward_f32` (transform) | Hexagon HVX butterfly on V69; HMX matmul on V73+ |
| `sp_mobius_reorder` / `_unreorder` | Hexagon scalar (small, cache-friendly) |
| `sp_shadow_cache_*` (allocator + dispatch) | ARM host, calls into Hexagon |
| `sp_shadow_cache_load_partial` (disk IO) | ARM host, only the dequantize step on Hexagon |

The dispatch glue stays on ARM because UFS / file system access is host-side. The Hexagon side only sees compressed band buffers in shared memory once the host has read them off disk.

---

## Implementation milestones

### M0: Scaffolding (this commit)

- `shannon_prime_hexagon.h` with the API surface
- `shannon_prime_hexagon.c` stub returning -1 / NULL
- This roadmap doc
- README.md in the backend dir pointing at this doc

What works: nothing runtime; the SP build stays green and other backends still function.

### M1: Hello-world FastRPC round-trip

- Write `shannon_prime_hexagon.idl` declaring a single function:
  `int hexagon_echo(in uint8_t* buf, in size_t n, out uint8_t* result);`
- Run `qaic` to generate stubs.
- DSP side: copy input to output. No HVX. Just verify the IPC works.
- ARM side: `sp_hexagon_init` opens the FastRPC session and exposes the
  echo function via a private API.
- Smoke test: push to S22 Ultra via adb, run, verify output matches.

What works: FastRPC IPC is verified, signing flow is verified, build pipeline produces working binaries.

### M2: Single-vector round-trip

- IDL gains `hexagon_round_trip_k(in *uint16_t in, out *uint16_t out, ...)`.
- DSP side implements the full pipeline IN SCALAR HVX (no vectorisation):
  fp16->fp32 promote, VHT2 butterfly, Mobius reorder, band quantize +
  dequantize, Mobius unreorder, VHT2 inverse, fp32->fp16 demote.
- Compare the output against the CPU reference path (Adreno backend's
  scalar fallback) and assert correlation > 0.99 across 100 random inputs.

What works: SP round-trip is functionally correct on Hexagon. No
performance claims yet - scalar HVX is slow; the win comes in M3.

### M3: HVX-vectorised kernel

- Refactor M2's scalar code to use HVX intrinsics from
  `hvx_hexagon_protos.h`. The inner loops of band_quantize / dequantize
  vectorise cleanly: 128 int8 lanes per HVX vector, scale broadcast,
  multiply-accumulate.
- VHT2 butterfly (1/sqrt(2) scaling) runs as paired HVX
  add/subtract + multiply-by-constant.
- Bench: round-trip throughput in vectors-per-second on the S22 Ultra
  vs the ARM NEON Adreno backend. Target: >= 2x ARM NEON.

What works: full Hexagon backend, runtime-selectable via
`SHANNON_PRIME_BACKEND=hexagon` env var (or auto-detected when DSP is
present and HVX is available).

### M4: Async / overlapped IO with phase 3

- Hook the Hexagon backend into the partial-load path
  (`sp_shadow_cache_load_partial`). Disk reads happen on the ARM side;
  the Hexagon DSP receives band 0 first, processes it, then band 1
  while the next attention query is being computed.
- Pair with phase 3 attention short-circuit: most queries terminate
  after band 0 + 1, the rest walk further down the band stack.

This is the milestone where the architectural win actually shows up
in user-perceived latency.

---

## Validation plan

End-to-end smoke test for each milestone:

1. Build on the dev machine with the SDK set up.
2. `adb push` the `.so` files to the S22 Ultra.
3. Run a tiny integration test from a host Android app or via
   `adb shell` if signing allows.
4. Compare output against the CPU reference within tolerance.
5. Record cycle counts via `HEXAGON_REG_TIMER` and report
   in `archive/eval/bench/hexagon-<milestone>-<date>.csv`.

Per-milestone, the smoke test should be small enough to fit in a
single FastRPC call (microseconds-class). Performance comparisons go
through the existing bench harnesses (`bench_disk_partial`, the
speculative-decoding bench) once the backend is dispatchable from the
bridge.

---

## Open questions

These come up during M1-M3 and need decisions:

- **Signing for production builds.** Dev builds use the included
  testsig; shipping requires either Qualcomm signing or a custom-key
  approach if the user has access to a manufacturer-line signing
  workflow. Out of scope for the math/kernel work but blocks public
  distribution.

- **Vendor library reuse.** Qualcomm ships `qhl` (Qualcomm HVX
  Library) with BLAS-style kernels. SP's banded quantize is too
  specialised to map onto qhl directly, but the inverse VHT2
  butterfly might benefit from qhl's batched fp16 multiply
  routines. Worth measuring once M3 lands.

- **Multithreading on the DSP.** V69 has 6 hardware threads. The
  obvious mapping: one thread per layer, batching across heads. Not
  required for correctness; pure throughput optimisation.

- **Memory budget per FastRPC call.** Shared memory is constrained
  (typically a few MB). For long-context inference, one layer's worth
  of K (8 heads x 2048 positions x 76 bytes ~= 1.2 MB) fits, but
  multi-layer batches need streaming. Design decision: per-call
  granularity = single layer, all heads + positions.

---

## See also

- `BACKEND-ADRENO.md` - the existing ARM/NEON backend that the
  Hexagon work composes with on the same Snapdragon SoC.
- `DISK-TIER-ARCHITECTURE.md` - end-to-end storage tier story; the
  Hexagon DSP is the compute side that pairs with the storage tiering.
- `PHASE-3-ATTENTION-DESIGN.md` - where the partial-band-read primitive
  meets attention; M4 of this roadmap is what actually wires it up.
- `position_is_arithmetic_v8.md` - VHT2 + Mobius math underlying what
  the HVX kernels are computing.
