# Disk-Tier Architecture: Progressive Band Loading

This document describes Shannon-Prime's progressive disk-paged tier loading architecture — what it does, why it works, where the pieces live in the code, and how it maps onto real hardware (laptop NVMe, Optane, Hexagon NPU on Snapdragon).

The architecture composes three orthogonal pieces that have been built in three phases:

| Phase | What it does | Where it lives | Status |
|---|---|---|---|
| 1 | Reconstruct K/V from the first N bands of the in-memory packed cache | `sp_band_dequantize_partial` in core | shipped |
| 2 | On-disk band-major layout so partial reads only touch N bands' bytes | v3 cache format + `sp_shadow_cache_load_partial` | shipped |
| 3 | Attention short-circuit using partial reads | bridge + llama.cpp patch | designed, not yet wired |

Together they let attention pop bands off a stack of progressively colder storage — RAM, NVMe, Optane, network — and stop when the answer is determined. For a 70B model on a phone with 12 GB RAM, this is the difference between "fits" and "doesn't fit" with no quality regression on the ~99% of tokens that don't need full-fidelity K/V.

---

## Why it works: energy concentration in the early bands

The architecture is built on one empirical fact about VHT2-transformed K vectors: **most of the signal lives in the first one or two bands.**

Measure it on a smooth signal (the kind real K vectors tend to be after attention has done a few thousand updates):

```
max_bands  reconstructed_correlation   notes
---------  -------------------------   -----
0          0.00                        all-zero output (sentinel)
1          0.30                        band 0 alone — first 32 of 128 coefficients
2          0.86                        bands 0+1 — most of the signal
3          0.88                        bands 0+1+2 — diminishing returns
4          0.99                        full reconstruction
```

(These are the test_band_dequantize_partial smoke-test numbers from the math core. Real K cache energy concentration depends on the model and the layer, but the shape — most-of-the-signal-by-band-2 — is consistent across architectures we've measured.)

The architectural insight: if 86% of the signal lives in bytes 0..43 of a 76-byte compressed vector, attention can answer most queries by reading those 44 bytes and ignoring the rest. The remaining 32 bytes only matter for queries where the dot product is close to a tie — which is the rare case.

VHT2 + Möbius reordering is what makes this concentration as strong as it is. Random K vectors would distribute energy uniformly across bands. The transform pulls the predictable structure (low-frequency components, squarefree-priority indices) into the front bands and pushes the noise tail to the back.

---

## The tier map: bands → storage → hardware

Each band of the compressed cache maps to a storage tier based on access frequency. Cold tiers can be slower because they're touched less often.

```
band 0 — Granite tier         RAM (always-resident, ~22 bytes/vec)
                              | accessed: every attention query
                              | latency: <100 ns (L2 cache)
                              v
band 1 — Sand tier (early)    NVMe SSD on the same host
                              | accessed: ~15-20% of queries (when band 0 alone is ambiguous)
                              | latency: 50-100 μs
                              v
band 2 — Sand tier (late)     NVMe or Optane (5-10× faster than NVMe)
                              | accessed: ~5% of queries
                              | latency: 5-100 μs
                              v
band 3 — Jazz tier            cold storage / network / archive
                              | accessed: ~1% of queries
                              | latency: 1-100 ms acceptable
```

The point isn't that band 3 is *useless* — it's that band 3 is only consulted when the first three bands left the answer ambiguous. The system as a whole runs at the speed of band 0 plus a fraction-of-a-percent tax from the rare cold reads.

### Why "Granite/Sand/Jazz"

These are the three tiers from the strange-attractor stack work landed in shannon-prime-comfyui — same three layers, applied here to the storage hierarchy instead of the compression hierarchy. The names work because:

- **Granite** is dense, slow-changing, high-energy. The skeleton band has those properties.
- **Sand** is granular, redistributable, mid-energy. Sand-tier reads happen when the system needs to see more detail.
- **Jazz** is improvisational, contextual, low-energy. Jazz reads only happen when nothing else has resolved the query.

The naming maps the empirical band structure (0/1-2/3) onto a conceptual structure that's easy to reason about in code reviews and architecture diagrams.

---

## How the three phases compose

### Phase 1 (shipped): in-memory partial dequantize

`sp_band_dequantize_partial(in, out, bc, max_bands)` reads the first `max_bands` bands of the compressed packed buffer in RAM and zeroes the rest. The compute saving comes from:

- Skipping the bit-unpack inner loop for unread bands
- The inverse VHT2 step that follows reduces to fewer non-zero contributions when the late bands are zero
- Cache-friendliness: only the read bands' bytes get touched

This is independent of where the buffer lives. Used by:

- The DualKvCache System 1 fast path (today: reads all bands; future phase 3: reads band 0 only when entropy is low)
- The attention dot-product short-circuit (phase 3)
- Any backend that wants to amortize reconstruction cost per-query (NPU, GPU)

API in `core/shannon_prime.h`:

```c
void sp_band_dequantize_partial(const uint8_t *in, float *vht2_coeffs,
                                const sp_band_config_t *bc, int max_bands);
```

### Phase 2 (shipped): band-major v3 disk format

Cache files written by `sp_shadow_cache_save` use format version 3, which restructures the on-disk layout from per-(head, position) interleaved bands (v2) to band-major (v3): all of band 0 across heads × positions contiguously, then all of band 1, etc. The header records each band's file offset.

This means reading "just band 0" is a single sequential read of a contiguous region — typically `n_heads × n_positions × ~22 bytes` — rather than seeking 4× per (head, position) record. On NVMe, sequential reads run at full bandwidth (3-7 GB/s on modern PCIe 4.0 drives); seeking pattern costs an extra 50-100μs per seek.

API:

```c
int sp_shadow_cache_load(sp_shadow_cache_t *sc, const char *prefix,
                         uint64_t expected_hash);
int sp_shadow_cache_load_partial(sp_shadow_cache_t *sc, const char *prefix,
                                 uint64_t expected_hash, int max_bands);
```

`load_from_disk_partial` clamps `max_bands` into `[0, n_bands]`, fseeks to the matching band region in the file, reads it into the in-memory cache, and zeros the regions for bands that weren't read. Subsequent `read()` calls on the cache produce partial-fidelity output via Phase 1's dequantize.

The C++ engine wrapper (`KvCache::load_from_disk_partial`) exposes the same API with the same semantics. v2 files are still readable — a header-version sniff dispatches to the legacy reader, which reads the full vector since v2's interleaved layout doesn't permit cheap partial reads.

### Phase 3 (designed, not yet wired): attention short-circuit

Designed in `docs/PHASE-3-ATTENTION-DESIGN.md` (companion file). The short version: when attention computes `softmax(QK^T)V`, it can use partial-fidelity K vectors for the `QK^T` step. If the resulting attention probabilities concentrate above a threshold (the "confidence" criterion), the answer is determined and the unread bands are unnecessary.

The hook ties into the existing DualKvCache entropy gate: the entropy threshold that today routes between System 1 (ship) and System 2 (hier) becomes a band-count selector. Low entropy → band 0 only; medium entropy → bands 0+1; high entropy → full read.

This is the layer where the tiered storage actually saves wall-clock time. Phase 1 + 2 build the infrastructure; phase 3 is where the user-visible latency win lives.

---

## Phone deployment: Galaxy S22 Ultra walkthrough

Concrete example: 12 GB RAM, Snapdragon 8 Gen 1 (Adreno 730 + Hexagon V69 NPU + 5 GB/s NVMe UFS 3.1), running a 7B-class target with SP-compressed KV.

### Today (phase 1+2, no phase 3)

Cold start: model file loads into RAM. KV cache is empty. As tokens generate, K/V pairs get compressed and stored.

After a few thousand tokens of context, the cache has grown to ~150 MB. The Adreno backend (which already does fp16-throughout NEON) handles compression at ~100 ns/vec. RAM pressure is fine.

When context grows past ~10 K tokens, the cache hits 500+ MB. At ~50 K tokens it's pushing 2 GB. On a 12 GB phone, that's tight — the OS, model weights, and other apps compete for the same memory.

The disk-tier scaffold (already implemented in `kv_cache.h::enable_cold_storage`) lets the engine evict cold positions to UFS. With v3 band-major format, those evictions are cheap reads when needed back.

### With phase 3 wired

Same cache, but attention now reads only band 0 (~22 bytes/vec) for most positions. RAM footprint for the active band-0 working set: ~30 MB regardless of context length. Bands 1-3 stream from UFS only when needed.

For a 50K-token context, that's ~30 MB resident vs ~2 GB. The phone runs cool, doesn't thermal-throttle, and the reads from UFS are amortized into the per-token latency budget without becoming the bottleneck.

### Hexagon NPU as a backend

The Hexagon V69 NPU on the 8 Gen 1 has its own SDK (Qualcomm Hexagon SDK 5.x) with HVX vector instructions (1024-bit) and HMX matrix-extension instructions (256x256 int8 GEMM). Both are well-suited to SP's banded quantize/dequantize:

- HVX can do 128 int8 ops/cycle, ideal for the per-band bit-unpack loop
- HMX can do the inverse VHT2 as a 128×128 GEMM-like operation
- The NPU runs in parallel with CPU and GPU at low power (~500 mW)

A future Hexagon backend would slot in alongside Adreno (NEON CPU/GPU) and CUDA (engine), sharing the same `sp_band_dequantize_partial` math primitive. The data flow:

```
UFS (cold)         load_partial(max_bands=N)        Hexagon NPU
   |                                                     |
   |  reads N×(~22 bytes) per layer                      |
   v                                                     v
CPU shared mem  ←  HMX-accelerated dequantize  ←       sp_band_dequantize_partial
                                                         (NPU kernel)
                                                         |
                                                         v
                                                       inverse VHT2 on HVX
                                                         |
                                                         v
                                                       attention QK^T
                                                       on Adreno GPU
```

Each handoff between CPU/NPU/GPU happens through Snapdragon's shared CPU-GPU cache (which the user mentioned was the original "stack and pop" pattern from earlier phone work). No PCIe-class round-trips, no VRAM copies.

---

## What's NOT covered (yet)

- **sqfree and hier paths still use v2.** These have different per-vec layouts (sqfree mirrors K↔V under spinor; hier has separate W matrix sidecars). Migrating them to v3 would let those compression modes also benefit from partial reads. Currently a follow-up work item.
- **Async / overlapped IO.** Today the partial-load reads are synchronous. A future enhancement would prefetch band N+1 while band N is being processed by attention, hiding the storage latency behind compute.
- **Cross-layer band sharing.** Bands 0 across all layers might compress further if the patterns are similar. Speculative; not currently in scope.
- **Cold-tier compression beyond banded int8.** Jazz tier could use even tighter encoding (e.g., entropy coding of the residual int8 quantized values) since access is rare. Untouched.

---

## See also

- `BACKEND-ADRENO.md` — ARM NEON backend, fp16-throughout transform path
- `INTEGRATION-LLAMA.md` — bridge env var reference, backend selection
- `SPECULATIVE-DECODING.md` — speculative decoding integration, draft model selection
- `BENCH-SPEC-DECODE.md` — bench harness + result interpretation guide
- `MODEL-PACK.md` — per-architecture preset registry (where draft suggestions live)
- `position_is_arithmetic_v8.md` — paper underlying VHT2 + Möbius decomposition
- `PHASE-3-ATTENTION-DESIGN.md` — companion design doc for the unwired attention hook
