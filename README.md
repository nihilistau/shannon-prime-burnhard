# Shannon-Prime BurnHard — Distributed Inference Engine

The production mono-repo for Shannon-Prime's distributed inference pipeline.
BurnHard runs across a heterogeneous compute fabric: Beast Canyon desktop 
(AVX-512, CUDA, Level Zero) coupled with an S22 Ultra mobile sidecar via 
ADB/USB-C bridge.

## Architecture

```
burnhard/
├── core/           Shared SP logic (CRT shredder, Garner join, Q4_K dequant, modular math)
├── desktop/        Beast Canyon (AVX-512, CUDA/RTX 2060, L0/UHD 750, Optane reservoir)
├── mobile/         S22 Ultra (Vulkan/OpenCL, Hexagon HTP, speculative draft model)
├── bridge/         ADB/USB-C heartbeat telemetry + task migration protocol
├── tests/          Smoke tests and integration harness
└── docs/           Architecture docs and runbooks
```

## Operating Modes

BurnHard detects hardware at runtime and selects one of four modes:

| Mode | Description |
|------|-------------|
| **FULL_PULSE** | All silicon active — S22 drafts tokens, Beast Canyon crunches residues |
| **DESKTOP_SOLO** | S22 detached — desktop handles both drafting and residue computation |
| **LEGACY** | No Optane — reservoir falls back to NVMe/system RAM |
| **LONE_WOLF** | Mobile standalone — S22 runs its own local GGUF engine |

Mode transitions happen live: unplug the phone and the desktop instantly reclaims 
speculative tasks. Plug it back in and FULL_PULSE re-engages.

## Building

```bash
mkdir build && cd build
cmake .. -G Ninja
ninja
./burnhard_test
```

## See Also

- [shannon-prime-bernhard](https://github.com/nihilistau/shannon-prime-bernhard) — Mathematical foundations and proofs
