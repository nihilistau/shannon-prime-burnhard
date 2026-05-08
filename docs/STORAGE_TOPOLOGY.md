# Beast Canyon Storage Topology

## Hardware

| Device | Capacity | Interface | Latency | Bandwidth | Role |
|--------|----------|-----------|---------|-----------|------|
| Intel P4800X | 375 GB | U.2 NVMe (PCIe 3.0 x4) | ~10 µs | ~2.5 GB/s seq | **Primary model reservoir** |
| Samsung 980 PRO | 32 GB (partition) | M.2 NVMe (PCIe 3.0 x4) | ~50 µs | ~3.5 GB/s seq | **Hot cache / scratch / KV spill** |

## Tiered Layout

### Tier 0: Optane P4800X (375 GB) — `O:\` or mmap'd directly

The Optane's defining feature is low random-read latency (~10 µs for 4 KB)
compared to NAND (~100 µs). At MoE expert granularity (gate=~50 MB, up=~50 MB,
down=~50 MB per expert for Qwen3.6-35B-A3B), random access patterns during
top-k expert selection benefit enormously from Optane's flat latency curve.

Contents (priority order):
1. **Active model GGUF** — the full model file, mmap'd by sp_optane_init().
   Qwen3.6-35B-A3B Q4_K_M = 21.2 GB, Qwen3.6-27B Q4_K_M = 16.5 GB,
   Gemma4-31B Q4_K_M = 18.7 GB. All fit with room to spare.
2. **Second model GGUF** — for dual-model spec-decode or A/B testing.
   35B + 27B = 37.7 GB, still under 375 GB.
3. **Draft model GGUF** — Qwen3-0.6B Q4_K_M = 0.4 GB (phone sidecar copy).
4. **Activation oracle snapshots** — per-layer activation statistics for
   calibration, ~200 MB per model.
5. **Cold storage for alternate quants** — Q6_K, Q8_0 variants for validation.

### Tier 1: NVMe 980 PRO partition (32 GB) — `N:\` or `D:\sp-scratch`

Fast NAND for data that doesn't benefit from Optane's latency advantage
but needs sequential throughput (the 980 PRO is actually faster sequentially).

Contents:
1. **KV cache spill** — when n_ctx > LLC-resident capacity and SP compression
   isn't active, KV cache pages spill here. At 4096 ctx, Qwen3.6-35B uses
   ~1.2 GB uncompressed KV. With SP compression (6× typical), fits in RAM;
   without, NVMe absorbs the overflow.
2. **Build artifacts** — `build_nocuda/`, `build_beast/` (~2 GB combined).
3. **Benchmark corpus cache** — tokenized test_corpus.txt etc.
4. **Shredder staging overflow** — if the ping-pong LLC buffers need
   backing store during sustained multi-expert decode bursts.

### Tier 2: System RAM (32 GB DDR4-3200)

- **LLC (24 MB L3)** — ping-pong staging buffers, hot attention heads,
  router logits. The Shredder's entire design targets LLC residency.
- **DRAM (remaining ~28 GB after OS)** — ggml compute graph allocations,
  intermediate activations, tokenizer tables, GDN recurrent state.
  With mmap, the model itself doesn't consume DRAM — only the pages
  actively touched by the Shredder are paged in via the OS.

## Data Flow

```
Optane (GGUF mmap) ──Shredder (AVX-512)──► LLC Ping-Pong ──► GPU A (CUDA)
                                                            └──► GPU B (L0)
                                                            └──► CPU (fallback)

NVMe (KV spill) ◄──► SP Compressed KV ◄──► DRAM cache ◄──► LLC
```

## Fallback Modes

| Mode | Optane | NVMe | Behavior |
|------|--------|------|----------|
| Full Pulse | ✓ | ✓ | Optimal: model on Optane, scratch on NVMe |
| Non-Optane | ✗ | ✓ | Model mmap'd from NVMe; higher latency for random expert access |
| Minimal | ✗ | ✗ | Model loaded into RAM (requires model < free RAM); no spill |

The `sp_beast_config_t.optane_budget` field caps Optane usage. When 0 (default),
all available Optane capacity is used. The reservoir init probes for Optane
by checking drive latency characteristics (DAX-capable, < 20 µs 4K random read).

## Capacity Planning

| Model | Q4_K_M | Q6_K | Q8_0 | Dual-model (Q4+draft) |
|-------|--------|------|------|----------------------|
| Qwen3.6-35B-A3B | 21.2 GB | — | — | 21.6 GB |
| Qwen3.6-27B | 16.5 GB | 22.1 GB | — | 16.9 GB |
| Gemma4-31B | 18.7 GB | — | — | 19.1 GB |
| All three Q4_K_M | 56.4 GB | — | — | 56.8 GB |

All target models fit on the P4800X individually with >300 GB headroom.
Even all three simultaneously fit (56.4 GB < 375 GB), enabling hot-swap
without unmapping. The 980 PRO partition handles everything else.

## Mount Points (Windows)

```
O:\Models\              ← Optane, GGUF files
O:\Models\oracles\      ← activation snapshots
N:\sp-scratch\build\    ← NVMe, build artifacts
N:\sp-scratch\kv-spill\ ← NVMe, KV overflow
```

## Implementation Notes

- `sp_optane_init()` already handles mmap on both Windows (CreateFileMapping)
  and Linux (mmap). No code changes needed for the topology — it's a
  deployment/mount concern.
- The `--model` CLI flag just takes a path. Point it at `O:\Models\<gguf>`
  for Optane-backed inference, or any other path for NVMe/DRAM fallback.
- Future: `sp_optane_detect_medium()` could probe 4K random latency to
  auto-select the optimal tier at runtime.
