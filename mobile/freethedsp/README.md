# Shannon-Prime: freethedsp shim — universal cDSP unblock harness

## STATUS 2026-05-01 — Finding from Phase D.1 dry run

**The freethedsp technique as upstream-designed does not apply on stock
production S22 Ultra (Android 14, OneUI 6.1, Snapdragon 8 Gen 1).**

Evidence captured during D.1 (`backends/freethedsp/discover.so`
LD_PRELOAD against the validated `S22U` test and our `sp_dma_raw`
probe):

  - LD_PRELOAD intercepts every ioctl correctly. The constructor logs
    `[discover] LD_PRELOAD active (pid=N)`, and we trace 30+ ioctls
    per FastRPC session.
  - **Zero `FASTRPC_IOCTL_INIT` (request 6) or `FASTRPC_IOCTL_INIT_CREATE`
    (request 5) calls** — not in our trace, not in any of the FastRPC
    sessions we exercised. The shell ELF never enters our userspace.
  - Samsung's kernel renumbered ioctl request 4 to be `_IOWR` (size
    24, matching the upstream `fastrpc_init_create` struct shape) but
    that 24-byte struct fires 18+ times per session and the `file`
    pointer it carries does NOT point at an ELF (we observed `ff ff
    ff ff` at the target address). So request 4 isn't init either —
    it's a renumbered different operation.
  - Logcat from prior Stage 1 work confirmed the kernel did open
    `/vendor/dsp/cdsp/fastrpc_shell_3` during our process's run, BUT
    the load happens through a path that doesn't surface the ELF in
    our address space. Most likely: cdsprpcd (system user) preloaded
    the shell at boot into a kernel-protected region, and our process
    attaches to it via the `INIT_ATTACH` ioctl (request 4 in upstream
    nomenclature) without seeing the bytes.

**Why upstream freethedsp worked**: older Snapdragon devices used the
"each app loads its own shell" flow, so every app got a userspace
mapping of the shell ELF and could patch it. Samsung/Qualcomm hardened
this for 8 Gen 1 by moving the shell load entirely into the daemon's
kernel-mediated path. Patching the daemon would require root.

**What this means for Shannon-Prime**:

  - The infrastructure built here (LD_PRELOAD shim, build script,
    integration hooks, drift detection) is sound and reusable. If we
    ever run on a dev kit or older device where INIT does fire, all
    the plumbing is ready.
  - But the specific shell-patch trick won't unlock Mode D Stage 1's
    DMA permissions on THIS device.
  - **This does not block Phase 2 (QNN-on-device).** Qualcomm's QNN
    runtime works in unsigned PD by design — that's how AI Hub
    deploys models. Standard graph execution doesn't need test mode.
  - **It does block re-running Mode D Stage 1 / Stage 3** here. Those
    are now permanently gated behind a real testsig or a dev kit.

**Decision**: keep the infrastructure (it's cheap to maintain and
useful elsewhere), pivot lead path back to Phase 2 (commit `f6799bd`),
park Mode D as a "needs hardware" experiment.

## Strategic role

Production-locked Android phones run unsigned PD on the cDSP by default,
which gates a long list of APIs we want: Halide DMA engine allocation
(Mode D Stage 1 wall, see backends/halide/probe/sp_dma_raw/BUILD_NOTES.md),
CVP/ISP MFNR handles (Mode D Stage 3), VTCM sharing in fully-customized
modes, and various direct hardware-coprocessor entitlements. None of
these can be unlocked by `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)`
alone — that flag only relaxes signature checks on our own skel; it
doesn't lift the permission wall enforced by `is_test_enabled()` inside
the per-process FastRPC shell on the cDSP.

geohot's `freethedsp` (https://github.com/geohot/freethedsp) is the known
public technique that closes this gap from pure userspace, no root, no
unlocked bootloader. It LD_PRELOAD-hooks `ioctl()`, watches for
`FASTRPC_IOCTL_INIT`, finds the in-memory shell ELF, and patches one
4-byte instruction so `is_test_enabled()` returns 0xFFFFFFFF. The cDSP
then treats the calling process as test-signed-equivalent, granting full
API permissions.

**Strategy for Shannon-Prime:** bake this shim in from the start, behind
a single `SP_FREETHEDSP=1` env var. Every backend that talks to the cDSP
(QNN-on-device runner in Phase 2, Halide DMA probe in Mode D Stage 1+,
the existing shannon-prime-hexagon FastRPC bridge, future CV-ISP probe)
loads it via LD_PRELOAD. If a given path doesn't *need* it (most don't),
the shim is a no-op overhead — single ioctl pass-through plus one
memcmp on the patched address. If a path *does* need it, no per-call-site
plumbing is required: it just works.

This way we're set for any obstacle, regardless of which API surface we
hit next.

## Adaptations needed for our S22U

geohot's reference patches `fastrpc_shell_0` (ADSP-domain shell) at
offset `0x5200c`, sha-hash `fbadc96848aefad99a95aa4edb560929dcdf78f8`.
Two device-specific adaptations:

1. **Different shell binary** — our cDSP uses domain 3, which loads
   `/vendor/dsp/cdsp/fastrpc_shell_3` (logcat from prior Stage 1 run
   confirmed: `Successfully opened file /vendor/dsp/cdsp/fastrpc_shell_3`).
   That file is permission-denied to shell user, but we don't need to
   read it from disk: when our process calls `FASTRPC_IOCTL_INIT`, the
   kernel has already mmaped the shell into our own address space at
   `init->mem`. Our shim dumps from there, hashes it, and locates
   `is_test_enabled` via Hexagon objdump on the dump (offline).

2. **Cache-flush path** — geohot's flush goes through `/dev/ion` with
   `ION_IOC_CLEAN_INV_CACHES`. Android Q+ phones (including S22U)
   replaced `/dev/ion` with `/dev/dma_heap/qcom,system` (we confirmed
   `/dev/ion` doesn't exist on this device during Mode D Stage 1
   debugging). Two options:
     a. Use the dma_heap-equivalent flush ioctl (DMA_HEAP_IOCTL_*).
     b. Skip the explicit flush and rely on `msync(MS_INVALIDATE)` on
        the userspace mapping — the kernel handles coherence between
        the user-side write and the cDSP-side read on V66+.
   Option (b) is simpler and likely sufficient.

## Layout

```
backends/freethedsp/
├── README.md                  # this file
├── discover.c                 # dump-only mode: captures fastrpc_shell_N
│                              # to disk, no patching. Used to derive the
│                              # PATCH_ADDR for our specific device.
├── freethedsp_s22u.c          # patching mode: our adapted shim, with
│                              # the S22U-specific PATCH_ADDR + hash
│                              # baked in once we've discovered them.
├── tools/
│   ├── find_is_test_enabled.py # Hexagon-objdump wrapper that finds the
│   │                            # `mov r0, #0; jumpr lr` pattern at the
│   │                            # is_test_enabled symbol.
│   └── verify_patch.py         # sanity-checks PATCH_OLD bytes match the
│                                # captured shell at PATCH_ADDR.
├── build.cmd                  # cross-compile via NDK aarch64-clang into
│                              # libfreethedsp.so (and discover_dump.so).
└── INTEGRATION.md             # how every other backend loads the shim
                               # (LD_PRELOAD via build-example.ps1's
                               # -Run flag, the existing build.cmd's
                               # adb shell command, etc).
```

## Phasing

- **D.1** Build `discover.c` + cross-compile + push + LD_PRELOAD against
  any FastRPC-using binary on the device (the S22U test or our existing
  sp_dma_raw probe). Capture the in-process `fastrpc_shell_3` dump.
  (~30 min.)

- **D.2** Run `tools/find_is_test_enabled.py` on the dump. Locate the
  `is_test_enabled` symbol via Hexagon objdump's symbol table; identify
  the instruction at its entry that loads the immediate `0` into r0.
  Compute PATCH_ADDR (offset within the shell binary), PATCH_OLD (the 4
  bytes of `r0=#0; jumpr lr` epilogue), PATCH_NEW (`r0=#-1; jumpr lr`).
  (~1 hr if hexagon-objdump symbols are present; ~3 hr if we need to
  reverse-engineer the function entry from a stripped binary.)

- **D.3** Bake offsets into `freethedsp_s22u.c`, port the cache flush
  to msync/dma_heap, build, smoke-test against the existing Stage 1
  probe. Expected outcome: `halide_hexagon_dma_allocate_engine` returns
  success instead of qurt_exit. (~1-2 hr.)

- **D.4** Wire into `backends/halide/build-example.ps1`'s `-Run` flag
  and `backends/hexagon/scaffold/build.cmd`'s adb invocation. Document
  in `INTEGRATION.md`. (~30 min.)

Total estimate: 4-7 hours, gated only on D.2 (finding the right offset).

## What this gives us

After D.4 lands, Mode D Stage 1+2+3 all become re-runnable on our exact
S22U with no testsig and no dev kit. The original probe (eDmaFmt_RawData
accepts non-camera bytes? UBWC bonus? CVP MFNR handle?) executes
end-to-end. So does the entire Mode C streaming architecture if we
choose to go there.

It also future-proofs us: any new Qualcomm API we want to call that's
gated on signed PD becomes accessible by setting one env var.

## Risk register

- **Patch offset drift across firmware updates.** The user has the 8.0
  OTA staged but not installed on this device (per session memory). If
  that gets installed, the shell binary changes and we re-run D.2.
  Mitigation: discover.c is permanent, the tooling makes re-discovery
  cheap.
- **Hexagon ELF symbol table may be stripped.** If `is_test_enabled`
  isn't in the symbol table, we fall back to disassembly + signature
  matching. Slower, still doable.
- **Kernel-side ioctl interception detection.** Some devices have
  hardened the FastRPC driver to detect modified shell pages. If we
  see a post-patch failure mode like "shell hash mismatch", we need
  to also patch the page-protection or remap the shell page. geohot's
  technique presumes pages are RW from userspace at INIT time, which
  has been true on every Qualcomm device tested publicly so far.
