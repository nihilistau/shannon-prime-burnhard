# Cauchy Reset System

Decode-chain causal stability control for Shannon-Prime compressed caches.

## What it does

Every compressed KV cache accumulates per-step reconstruction error across a
decode chain. On short chunks the effect is invisible; on long chunks it
compounds into measurable PPL regression at the *tail* of the chain. The
Cauchy reset system detects when this regression is about to break and
refreshes the cache by re-prefilling from ground-truth tokens — recovering
the chain's causal coherence without dropping compression.

It is a **decode-chain supervisor**, not a new compression scheme. It runs
alongside the ship / sqfree / hierarchical paths.

## Shipping configuration

The shipped default is **Mertens-only** (a zeta-zero-derived arithmetic
schedule). Originally architected with three layers — a dynamic Ricci drift
sentinel, the Mertens schedule, and a fixed-N ablation mode — the final
measurement showed Mertens alone delivers 100% of the useful PPL recovery,
so Ricci is now opt-in.

### Measured on Qwen3-8B-Q8 (wiki.test.raw, RTX 2060, `perplexity --cache`)

| Config | PPL | Δ vs baseline | Wall | Resets |
|---|---|---|---|---|
| Baseline (cache only, no Cauchy) | 12.23 | — | 3m04s | — |
| **Mertens-only (default)** @ ctx=1024 | **11.92** | **−0.31** | **4m53s** | **4** |
| Mertens-only @ ctx=2048 chunks=2 | 12.98 | −0.11 | 14m00s | 7 |
| Ricci-only warmup=64 cd=64 | 12.02 | −0.21 | 3m47s | 2 |
| Full (Mertens+Ricci) | 11.92 | −0.31 | 4m53s | 4 |

Across ctx=512 / ctx=1024 / ctx=2048 the recovery fraction is broadly flat
at 2.2–2.3% of baseline PPL at ctx=512–1024 and shrinks to ~0.8% at ctx=2048.
All resets land in the late tail of the chain — Mertens' design intent.

## How it works

Three mechanisms exist in the codebase; only the first is on by default.

**1. Mertens oracle.** Uses 50 Riemann zeta zeros to derive a deterministic
arithmetic schedule of "high-risk" positions across the chain. At each
scheduled position the controller flags a reset candidate; combined with
the cooldown gate below, a small subset (~4–7 on Qwen3-8B ctx=1024) fire
as actual resets. The schedule density scales with `ctx` (32 flagged at
ctx=512, 73 at ctx=1024, 127 at ctx=2056).

**2. Ricci sentinel (opt-in).** Tracks the exponential moving average of
`p=3` band energy in the VHT2 spectrum of K vectors. When the EMA exceeds
a calibrated threshold (`metric_criticality`) the sentinel fires. In
measured practice Ricci fires early (pos 529–641) on fresh cache, not at
the tail where cache drift actually accumulates — and its EMA zeroes on
every fire, losing accumulated drift state. It contributes 0 incremental
PPL recovery over Mertens on Qwen3-8B. Kept in the code for future
research and ablation.

**3. Fixed-N ablation.** Deterministic reset every `cauchy_fixed_n` tokens.
Useful for A/B tests against the zeta schedule.

On reset, the supervisor:
1. Rebinds the cache (clears `kv_pos`, preserves calibrated masks).
2. Re-prefills the ground-truth token sequence up to and including the
   current position (`prefill(chunk[0..i+1])` in `perplexity --cache`,
   or `prefill(running)` in `chat`).
3. Records the reset so the cooldown gate suppresses re-firing for the
   next `cauchy_cooldown` positions.

## CLI flags

Shared across `perplexity --cache`, `cache_ppl`, and `chat`:

| Flag | Default | Effect |
|---|---|---|
| `--cauchy-mode N` | 0 | 0 = off, 1 = fixed-N, 2 = dynamic (zeta schedule) |
| `--cauchy-fixed-n N` | 512 | Reset period for mode 1 |
| `--cauchy-cooldown N` | 64 | Minimum positions between consecutive resets |
| `--cauchy-warmup N` | 64 | Suppress resets for first N positions of each chunk |
| `--cauchy-use-ricci` | off | Add reactive Ricci drift sentinel (opt-in) |
| `--cauchy-ricci-only` | off | Ablation: Ricci without Mertens (`perplexity`/`cache_ppl` only) |
| `--cauchy-mertens-only` | on | Explicit flag for the default (kept for script compat) |
| `--params-b F` | 0 | Model size in billions — only consulted when Ricci is enabled |

## When to use

- **Ship defaults (Cauchy off).** Short decode chains and ctx ≤ 512 rarely
  accumulate enough tail drift for Cauchy to make a measurable difference.
  Pay the wall-time cost only when it's earning PPL.
- **`--cauchy-mode 2`.** Long contexts (ctx ≥ 1024) and decode-heavy
  generation benefit most. Expect ~50–80% wall overhead from the re-prefills.
- **`--cauchy-use-ricci`.** Research-only. Measured 0 incremental PPL.
  Kept in-tree so the p=3 band EMA can be re-shaped (sticky EMA, lower
  learning rate) in future work.

## Files

| File | Role |
|---|---|
| `lib/shannon-prime/core/shannon_prime.h` | `sp_cauchy_ctrl_t`, `sp_ricci_sentinel_t`, `sp_mertens_oracle_t` structs |
| `lib/shannon-prime/core/shannon_prime_cauchy.c` | Init, check, reset, cooldown logic |

### Mertens oracle memory contract

`sp_mertens_init` allocates a `float risk_cache[max_ctx+1]` bake of the
schedule (decays around each flagged position to make `sp_mertens_risk`
O(1) instead of the binary-search fallback). Callers MUST pair each
`sp_mertens_init` with `sp_mertens_free(mo)` at teardown — the cache is
otherwise leaked per oracle instance. The engine's `KvCache` destructor
owns this call; downstream integrations that embed a mertens oracle
directly must do the same.
| `shannon-prime-engine/src/kv_cache.{h,cpp}` | `init_cauchy`, `cauchy_check`, `cauchy_set_cooldown`, `cauchy_record_reset` API |
| `shannon-prime-engine/src/cli/main.cpp` | `perplexity --cache`, `cache_ppl`, `chat` flag parsing + decode-loop wire-up |

## Open threads

- **ctx-scaling puzzle.** Recovery drops from 2.5% PPL at ctx=1024 to 0.8%
  at ctx=2048. Unclear whether the Mertens schedule saturates at deep
  context or whether `chunks=4+` restores the recovery fraction by giving
  the schedule more chain depth.
- **Chat verb at longer n_predict.** Shipping wiring handles resets but
  has not been benchmarked against a held-out reference generation.
- **Larger models.** Scaling law predicts Cauchy's relative value shrinks
  on 70B+ backbones (compression-PPL is already essentially free there).
  Expected regime shift from "PPL recovery" to "decode-chain insurance."
