# Phase 3 Design: Attention Short-Circuit Using Partial Band Reads

This document describes the design for wiring Shannon-Prime's progressive band-read primitive (phase 1 + 2) into llama.cpp's attention kernel. It's a design document — no code yet — written so the implementation work can proceed against a clear specification.

The goal: replace today's "always read full K/V" attention with "read as many bands as the answer requires, stop when confident." The latency win is the difference between reading 22 bytes/vec from RAM (band 0) and reading 76 bytes/vec from disk (full record), multiplied by the number of context positions attention scans per token. On a 50K-token context that's 50,000× a per-position savings — non-trivial.

---

## Two scopes for "confidence"

There are two distinct stopping criteria a partial read can target. They have different tradeoffs.

### Scope A: per-query confidence

Given a single query token Q, run attention against the partial K/V. If the resulting attention probabilities are sufficiently concentrated (e.g., top-1 probability > 0.9), the dot-product partial-fidelity has already determined the output. The unread bands wouldn't change which value vector dominates the softmax.

**Math:** softmax saturates fast. If `p_i = softmax(QK_i^T)_i`, then small noise in `QK_i^T` for the dominant `i` is harmless once `p_i` is large enough — the attention output is already pinned.

**Cost of testing:** running the softmax once, looking at top-k probabilities. Cheap.

**Win:** any query where the answer was determinable from band 0 saves 3-band reads. On structured-text data this is the majority of queries.

### Scope B: per-position confidence

For each (Q, K_i) pair across all positions i, decide whether *that pair* needs more bands. Useful when most positions don't matter (low attention weight) but a few do (high attention weight).

**Math:** if `|QK_i^T|` is small (the position contributes little to the softmax), there's no point reading more bands for K_i — the contribution stays small regardless. If `|QK_i^T|` is large, refine.

**Cost of testing:** comparing each `QK_i^T` magnitude against a threshold. O(n_positions) per query. Cheap if vectorized.

**Win:** scales with sequence length. On a 50K-token context, you might only refine 100-200 positions per query.

### Recommended starting point: Scope A (per-query)

Scope B is more powerful but harder to integrate cleanly because it requires per-position partial reads, which don't quite map onto today's flat band-major storage (you'd need finer granularity than per-band). Scope A is a clean wrapper around the existing partial-read API: read all bands at once for whatever max_bands is currently set, run attention, check confidence, increment max_bands and re-run if needed.

We'll spec scope A here. Scope B is a future optimization once the per-query path is proven.

---

## Confidence as a stopping criterion: three candidates

### Candidate 1: top-1 probability mass

After softmax, check if the highest-probability position has `p_top1 > THRESHOLD`. If yes, stop — that position dominates and reading more bands won't change the dominant attention.

```
def confident(attention_probs, threshold=0.85):
    return attention_probs.max() > threshold
```

**Pros:** simple, intuitive, well-defined math. Threshold is tunable per layer / per workload.

**Cons:** doesn't catch the "two close-tied positions" case where neither has p > 0.85 but the answer is still determined (both candidates contribute and the partial read is good enough for the weighted sum).

### Candidate 2: entropy threshold

Use the attention entropy `H = -sum(p * log p)`. Low entropy = concentrated, stop. High entropy = diffuse, refine.

```
def confident(attention_probs, threshold=1.5):
    return entropy(attention_probs) < threshold
```

**Pros:** single number, captures "concentrated mass" robustly even when multiple positions tie. Already computed by the existing DualKvCache routing code (`logit_entropy`), so the integration is mechanical.

**Cons:** threshold is harder to interpret physically — "1.5 nats" doesn't map to "this many positions are competing" without log-tables.

### Candidate 3: KL divergence between consecutive band counts

Compute attention with band=N and band=N+1, then check `KL(p_N || p_{N+1}) < threshold`. If reading one more band changed the distribution by less than threshold, stop.

**Pros:** directly measures "did adding a band matter?" — answers the actual architectural question.

**Cons:** requires computing both attentions (defeating most of the speedup) unless implemented incrementally. Probably too expensive in practice.

### Recommended: candidate 2 (entropy threshold)

Reuses the existing entropy computation in DualKvCache. Composes cleanly with the System 1/2 routing already wired. The threshold becomes a **two-knob** decision:

| Entropy | Action | Why |
|---|---|---|
| < 1.5 | Stop at band 0 | Attention is concentrated; band 0 is enough |
| 1.5–3.0 | Read up to band 1, re-test | Two-band fast path |
| 3.0–4.5 | Read up to band 2 | Mid-fidelity — covers most edge cases |
| > 4.5 | Read all bands (band 0..N-1) | Full reconstruction; the answer needs all the detail |

The exact thresholds come from per-layer calibration runs (separate work item).

---

## Where the hook lives in the patch

The llama.cpp attention path is:

```
in process_ubatch():
  for each layer:
    for each (Q, K, V):
      score = QK^T / sqrt(d)
      attention_probs = softmax(score + mask)
      output = attention_probs @ V
```

In Shannon-Prime's current per-model patch (`patches/llama-cpp-b8861-full-engine.patch`), we have:

```
llama_sp_post_compute(model, mctx, ubatch):  // after graph compute
  walks the KV cache
  for each (layer, head, position):
    gather K/V from the ggml KV tensor
    round-trip through the SP shadow cache (compress+reconstruct)
    scatter back to the KV tensor
```

Today this round-trip happens with full bands. Phase 3 changes it to:

```
llama_sp_post_compute(model, mctx, ubatch):
  for each (layer, head, position):
    gather K/V (full from ggml KV)
    SP-compress
    reconstruct using sp_band_dequantize_partial(max_bands=current)
    scatter back

  if entropy_check_enabled:
    compute attention probabilities on the partial result
    if entropy too high:
      reconstruct again with max_bands=current+1
      scatter back; loop
```

The entropy check happens after the first attention pass. If the answer is already concentrated, the loop exits. If not, additional bands get loaded and reconstruction repeats. The loop is bounded by `n_bands` (4 in the ship config).

### Alternative: pre-decode entropy estimation

Instead of computing entropy on the partial-attention output, predict it from the query Q alone. Specific Q vectors have characteristic entropy profiles (training-time statistic). This avoids the iterative re-reconstruction at the cost of needing a Q-conditioned threshold table.

Cleaner architecturally but requires a calibration table per layer × head. Not the starting point.

---

## Integration with DualKvCache

The existing System 1/2 routing already uses entropy as the gate between fast (ship) and full-fidelity (hier/sqfree) caches. Phase 3 extends that gate to a **3-tier** structure:

```
Today:
  System 1 (band 0, ship):  always
  System 2 (full, hier):    if entropy > threshold

Phase 3:
  System 1.0 (band 0):      always
  System 1.5 (bands 0+1):   if entropy > t1
  System 2.0 (all bands):   if entropy > t2
```

The existing `route_position(pos, entropy)` gets one more branch. The existing `read_merged()` logic that fetches across both caches becomes a band-aware reader that picks max_bands based on the routing decision.

State the implementation needs to track:

- Per-position `bands_loaded`: how many bands of this position are currently in the in-memory cache
- Promotion path: load_partial(max_bands=N) when entropy demands it
- Eviction path: re-zero bands above max_bands when memory pressure forces it

### Composes with the v3 disk format

When promotion fires (`bands_loaded` < max_bands needed), the engine calls `KvCache::load_from_disk_partial(max_bands)`. The v3 format makes this a single contiguous read of the missing band's region. No re-reading of band 0.

---

## Implementation work items

These are sized for follow-up PRs after this design lands.

### Work item 1: per-query partial attention

Modify `llama_sp_post_compute` to accept a `max_bands` parameter and route to `sp_band_dequantize_partial`. Default `max_bands = n_bands` (current behavior). Add `SHANNON_PRIME_DEFAULT_BANDS` env var for testing.

~80 lines in the patch. No quality regression possible at default `max_bands = n_bands`.

### Work item 2: entropy-gated band-count selection

Compute attention entropy after the first reconstruction. If above threshold, re-reconstruct with `max_bands+1`, repeat until threshold met or all bands consumed.

~150 lines in the patch. Requires tuning thresholds per layer.

### Work item 3: per-layer threshold calibration

Add a calibration phase that profiles entropy distributions per layer over a representative input set. Generate a `band_thresholds[]` table that ships alongside the model preset.

Calibration tooling work. ~200 lines + a calibration dataset.

### Work item 4: async / overlapped IO

When promotion fires, kick off the disk read for `band+1` asynchronously. Continue computing attention with `band` in the meantime. Hide storage latency behind compute.

~100 lines + async-IO library dependency. Optional optimization, not blocking for the main feature.

### Work item 5: per-position partial (Scope B)

The harder Scope B variant: per-position max_bands selection based on `|QK_i^T|`. Requires finer-grained partial reads — likely a v4 disk format with per-(head, position, band) seeks instead of per-band whole-region reads.

Larger architectural change; deferred until Scope A proves the win.

---

## Validation and acceptance criteria

For phase 3 to be considered shipped:

1. **No quality regression at `max_bands = n_bands`.** The existing test suite continues to pass.
2. **Wall-clock improvement at default thresholds.** On a representative workload (e.g., 8K-token continuation), entropy-gated band selection should be ≥ 10% faster than full-band attention with PPL drift ≤ 0.5%.
3. **Per-layer calibration ledger.** The `band_thresholds[]` for each shipping preset gets a row in `MODEL-PACK-CALIBRATION.md` with measurement provenance.
4. **Disk-IO confirmed under cold cache.** A bench run with the OS file cache dropped (Windows: `RAMMap` flush) shows the partial-read win on cold pages.

---

## Open questions for implementation

These are decisions the implementer will need to make once they start writing code:

- **Where to insert the entropy check.** Before or after the V-multiplication? Before saves a multiplication; after gives more accurate entropy.
- **Should layers calibrate independently or share a threshold?** Per-layer is more accurate but harder to maintain. Single global threshold is simpler.
- **What's the fallback if disk reads fail mid-promotion?** Today's `load_from_disk_partial` just returns the partial state. The attention step should detect this and proceed at the lower fidelity rather than crashing.
- **Interaction with Cauchy reset.** Cauchy mode 2 forces full re-compression of recent positions. Does that imply forcing `max_bands = n_bands` for the reset window? Probably yes; needs an explicit pass-through.

---

## See also

- `DISK-TIER-ARCHITECTURE.md` — overarching architecture, where this phase fits
- `BACKEND-ADRENO.md` — phone backend that's already fp16-throughout
- `MODEL-PACK-CALIBRATION.md` — calibration ledger format (where work item 3's results would land)
- `core/shannon_prime.h::sp_band_dequantize_partial` — the math primitive being called
- `kv_cache.h::DualKvCache` — the existing entropy router this phase extends
