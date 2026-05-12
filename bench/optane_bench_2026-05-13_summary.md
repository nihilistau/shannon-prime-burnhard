# Optane / NVMe disk bench — 2026-05-13

DiskSpd 2.2.0, 1 GB scratch file per drive, 10 s sweep, hardware + software
cache disabled (`-h`). All read-only (`-w0`).

## Drives

| Mount | Label         | Size   | Notes                              |
|-------|---------------|--------|-------------------------------------|
| F:    | OPTANE 32GB   | 29 GB  | gen 4×4 Optane on PCIe (the fast one) |
| E:    | Optane        | 14 GB  | older 16 GB Optane                  |
| C:    | (system NVMe) | 953 GB | Windows install drive               |
| D:    | Local Disk    | 931 GB | second NVMe                         |

## Results

### T1 — 4K random read, QD=1, 1 thread (small-read latency)

| Drive  | MiB/s | IOPS    | **Avg latency** | p50    | p99    |
|--------|-------|---------|-----------------|--------|--------|
| **F:** | 217   | 55,585  | **0.018 ms** ✓  | 0.017  | 0.029  |
| D:     | 180   | 46,083  | 0.022 ms        | 0.020 *  | 0.029 * |
| E:     | 187   | 47,940  | 0.021 ms        | 0.019  | 0.038  |
| C:     | 137   | 35,049  | 0.028 ms        | 0.028  | 0.037  |

**F: Optane wins on small-read latency**: 18 μs vs NVMe 22–28 μs (35% faster than C:).

### T2 — 4K random read, QD=32, 4 threads (concurrent small reads)

| Drive  | MiB/s | IOPS     | Avg latency | p99    |
|--------|-------|----------|-------------|--------|
| **D:** | 1272  | 325,527  | 0.045 ms    | 0.07 * |
| F:     | 1233  | 315,660  | 0.071 ms    | 0.244  |
| C:     | 1193  | 305,407  | 0.065 ms    | 0.07 * |
| **E:** | 874   | 223,757  | 0.554 ms ✗  | 1.051  |

Both NVMe drives + the 32 GB Optane all converge ~1.2 GiB/s under
concurrency. **The 16 GB Optane E: saturates** — latency jumps 26×
(0.021 → 0.554 ms). Old controller, lower channels.

### T3 — 1 MiB sequential, QD=4, 2 threads (bulk bandwidth)

| Drive  | **MiB/s** | IOPS | Avg latency |
|--------|-----------|------|-------------|
| **D:** | **3364**  | 3364 | 2.38 ms     |
| **C:** | **3285**  | 3285 | 2.43 ms     |
| F:     | 1347      | 1347 | 5.94 ms     |
| E:     | 884       | 884  | 9.05 ms     |

NVMe wins bulk bandwidth ~2.5× over the 32 GB Optane. **Optane was
never about peak bandwidth — it's about latency at low QD.**

## Implications for inference

| Workload                                        | Best drive   | Why                                  |
|--------------------------------------------------|--------------|--------------------------------------|
| Initial model load (22 GB GGUF, bulk seq read)   | D: or C: NVMe | 3.3 GB/s seq, model loads in ~7 s    |
| Routed-MoE expert page faults (sparse 4K reads)  | **F: Optane** | 18 μs lat — page faults during decode |
| KV-cache spillover when RAM tight                | **F: Optane** | Small-block latency dominates here    |
| Draft-model storage (200-800 MB, used at decode) | F: Optane     | Tiny model, latency-sensitive         |
| Pagefile under memory pressure                   | F: Optane     | Random 4K reads where Optane wins     |

**Concrete placement plan:**

1. **Keep the 35B-A3B GGUF on its current drive (D:)** for fast initial
   `lms load` (~7 s vs ~17 s on Optane).
2. **Configure Windows pagefile on F:** (32 GB Optane). With 32 GB DDR4
   RAM you shouldn't hit it often, but when you do during context
   growth or KV expansion, Optane's 18 μs random-read latency is
   ~35% lower than NVMe — preserves decode throughput under pressure.
3. **Store dflash draft GGUFs on F: Optane** (they're 278 MB / 491 MB,
   loaded once per session, then memory-resident — but Optane gives
   the fastest first-load latency).
4. **E: (16 GB Optane)** — its T2 saturation makes it a poor fit for
   concurrent reads. Best use: pagefile overflow only, or a write-heavy
   scratch dir (model conversion intermediates, training shards).

## Raw log

Full DiskSpd output in `bench/optane_bench_2026-05-13.log`.
