# Benchmarking Speculative Decoding with Shannon-Prime

This guide documents the `scripts/bench-spec-decode.ps1` harness and how to interpret its output. It pairs with `SPECULATIVE-DECODING.md` (architecture overview) — that doc explains *what* Shannon-Prime + speculative decoding do; this one explains *how to measure whether it actually wins on your hardware*.

## What the harness measures

Five configurations, in order:

| # | Name | What it tests | Env vars |
|---|---|---|---|
| 1 | `vanilla` | No SP at all — baseline tok/sec | `SHANNON_PRIME_ENABLED=0` |
| 2 | `shared` | Ship preset applied to both contexts (sp1-style) | `ENABLED=1`, `K_BITS=5,5,4,3`, `V_BITS=3` |
| 3 | `per-model-same` | sp2 per-model context, same defaults on both | + `SPEC=1` |
| 4 | `differential-agg` | sp2 with aggressive draft compression | + `DRAFT_PRESET=aggressive` |
| 5 | `differential-tern` | sp2 with ternary band-3 on draft | + `DRAFT_K_TERNARY_BANDS=3` |

Each configuration is run `-NRuns` times (default 3). The first run is treated as warmup and discarded; subsequent runs are averaged. Three numbers per config:

- **TokSec** — tokens per second from llama.cpp's `eval time` line, mean of the non-warmup runs.
- **Acceptance** — fraction of draft tokens accepted by the target. Speculative decoding's free win comes from high acceptance: at α≈0.7 with a 10× cheaper draft you get ~2.3× wall-clock; at α≈0.5 the win shrinks toward 1.4×.
- **EditDistVsVan** — character-level Levenshtein distance vs the vanilla run's output. Only computed when `-CompareOutputs` is set; tells you whether the draft compression actually changed what the model produced. Zero means byte-identical output (proves SP is invisible at this temperature/seed); non-zero means real divergence.

## What "good" output looks like

A healthy sp2 run on an 8B target + 1B draft (Qwen 2.5 family) should look roughly like:

```
Config             Note                                       TokSec  Acceptance  EditDistVsVan
------             ----                                       ------  ----------  -------------
vanilla            no SP, baseline                             22.40        0.00              0
shared             ship preset, both ctx (sp1-style)           26.92        0.00              7
per-model-same     sp2 per-model, same defaults                26.85        0.71            12
differential-agg   sp2 differential - draft K=2,1 V=1          28.10        0.65            18
differential-tern  sp2 differential - draft band 3 ternary     27.40        0.69             9

Speedup vs vanilla:
  vanilla                  1.00x
  shared                   1.20x
  per-model-same           1.20x   (within noise of shared - sp2 is correctness fix, not perf win)
  differential-agg         1.25x   (draft compression saves draft-side time)
  differential-tern        1.22x   (mild win - ternary keeps acceptance near baseline)
```

Numbers will vary heavily by hardware, model pair, prompt, and `--draft-max`. The relative shape matters more than absolute values:

- `shared` ≈ `per-model-same` confirms sp2's per-model architecture isn't a perf regression vs sp1's shared global. (sp2 fixes correctness; perf parity is the goal.)
- `differential-agg` > `per-model-same` confirms differential compression is a real win — the draft runs faster because its KV cache is smaller and quantisation is cheaper.
- `differential-tern` between the two — modest speed win, smallest acceptance hit.
- `EditDistVsVan` rises slightly with more aggressive draft compression. That's expected: more aggressive draft → more rejected proposals → speculative loop ends up doing more verification, output diverges from the no-spec baseline.

## When the numbers look wrong

**`per-model-same` is much slower than `shared`:** likely you're on an sp1 binary (single-global SP context). Two contexts loading sequentially with `SPEC=1` set in the env increment a counter that doesn't actually do anything in sp1. Check `git rev-parse HEAD` in shannon-prime-llama; you want a commit at or after PR #8 (commits with the per-model context map merged).

**`differential-agg` has acceptance < 50%:** the draft is too aggressively compressed for this model pair. Try `DRAFT_PRESET=ternary` (less aggressive) or back off to per-band overrides (`DRAFT_K_BITS=3,3`). Or — the draft model is too small for the target; try a 1B draft instead of 0.5B.

**`vanilla` is slower than `shared`:** unusual but possible if the SP path on your hardware has a fast quantisation loop and the vanilla path is memory-bound. Not a bug; just means SP's compute overhead is less than the memory-bandwidth saving on this hardware. Lucky you.

**All four SP configs are identical:** the SP env vars aren't reaching the bridge. Check `SHANNON_PRIME_VERBOSE=1` output — you should see `[Shannon-Prime] enabled: ...` lines. If absent, the patch isn't applied or `LLAMA_SHANNON_PRIME` wasn't defined at compile time.

## Running the harness

### Prerequisites

- Windows PowerShell 5.1+ (Linux/Mac users: port the env-var manipulation; the rest is portable).
- `llama-cli` built with `-DLLAMA_SHANNON_PRIME=ON` against shannon-prime-llama v2.14.0-sp2 or later.
- Target + draft GGUFs sharing a tokeniser (see `SPECULATIVE-DECODING.md` for recommended pairs).

### One-shot run

```powershell
cd path\to\shannon-prime-llama
.\scripts\bench-spec-decode.ps1 `
    -Target  "D:\models\qwen2.5-7b-instruct-q4_k_m.gguf" `
    -Draft   "D:\models\qwen2.5-0.5b-instruct-q8_0.gguf" `
    -OutputCsv "qwen25-bench-$(Get-Date -Format yyyyMMdd).csv" `
    -CompareOutputs
```

Total wall-time on a 2060 (12 GB) for the default 5 configs × 3 runs × 256 tokens at `--draft-max 8`: ~6-8 minutes. The two GPU warmups dominate the first 60 seconds.

### Dry-run

```powershell
.\scripts\bench-spec-decode.ps1 -Target fake.gguf -Draft fake.gguf -DryRun
```

Prints the exact `llama-cli` command each config would run, with the env-var prefix. Useful for sanity-checking the matrix or for porting the harness to a different driver.

## Recording results

Append the CSV row to `archive/eval/bench/` (workspace-local; not version-controlled — same convention as the calibration ledger). Keep one CSV per (model_pair × hardware × commit_sha) tuple. The CSV columns are stable so you can later `csvkit cat` them into a single dataframe for cross-machine comparison.

If a configuration produces a result worth highlighting in a release note (e.g., "sp2 + differential-agg gives 2.7× on Qwen 2.5 7B"), copy the relevant row into `CHANGELOG.md`'s release entry.

## See also

- `SPECULATIVE-DECODING.md` — architecture overview, why per-model SP context matters, draft preset semantics.
- `INTEGRATION-LLAMA.md` — env var reference for all `SHANNON_PRIME_*` knobs the harness exercises.
- `../FUTURE-WORK.md` (workspace root) section 8a — open questions about differential compression bench targets across model-size ratios.
