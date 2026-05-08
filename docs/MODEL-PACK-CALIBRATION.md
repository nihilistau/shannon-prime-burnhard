# Shannon-Prime Model Pack — Calibration Ledger

Append-only log of every `PROVISIONAL → CALIBRATED` promotion attempt and the
measurements that backed it. One row per attempt — including the ones that
fail budget and stay `PROVISIONAL`, because "we tried and it didn't pass" is
itself a valuable record. This file is the source of truth for the status
column in [MODEL-PACK.md](MODEL-PACK.md#current-calibration-status-2026-04-21);
the table over there summarises the latest row per preset, this file carries
the history.

Reviewer field is the git-author handle that signed off on the promotion
decision. Every row should list the exact git SHAs for both
`shannon-prime` and `shannon-prime-engine` so the measurement is reproducible.

## Conventions

- **Budget** is stated as a fraction of baseline PPL, consistent with the
  [MODEL-PACK.md layering note](MODEL-PACK.md#promotion-recipe-provisional--calibrated):
    - Ship path: drift / baseline ≤ 0.05
    - Sqfree: drift / baseline ≤ 0.10
    - Sqfree + spinor: drift / baseline ≤ 0.15
    - Hierarchical: drift / baseline ≤ 0.15
- **Corpus** is `archive/eval/wiki.raw` — a 488 KB concat of workspace markdown
  (see archive/eval/CALIBRATION_FINDINGS.md for the recipe). Real wikitext
  runs will be appended alongside; the corpus column makes the distinction
  explicit.
- **Drift** is `PPL(candidate) − PPL(baseline)` where both numbers come from
  `sp-engine perplexity` at matched `--ctx` and `--chunks` — baseline with
  `--model-preset off`, candidate with `--model-preset auto` plus the path
  flags (sqfree, spinor, etc.). Never mix `cache_ppl` PPL with `perplexity`
  baseline — see the calibration findings doc for why.
- **Status** tracks the resulting registry state:
    - `PASS` → flipped to `SP_PRESET_CALIBRATED` in the registry
    - `FAIL` → kept at `SP_PRESET_PROVISIONAL`, row is the negative evidence
    - `PARTIAL` → run didn't complete (aborted / crashed / time-capped)

## Ledger

| Date       | Preset    | Path        | Model                              | GGUF SHA256 (first 16) | SP core SHA | Engine SHA | ctx / chunks | Baseline PPL | Candidate PPL | Δ      | Δ / baseline | Budget  | Result   | Reviewer |
|------------|-----------|-------------|------------------------------------|------------------------|-------------|------------|--------------|-------------:|--------------:|-------:|-------------:|---------|----------|----------|
| 2026-04-20 | llama-3   | ship        | Dolphin3.0-Llama3.2-1B.Q8_0.gguf   | (not recorded)         | `aaa3374`   | `a105b99`  | 512 / 4      | 38.4383      | 44.7319       | +6.29  | 16.4 %       | ≤5 %    | **FAIL** | KnackAU  |
| 2026-04-20 | llama-3   | sqfree      | Dolphin3.0-Llama3.2-1B.Q8_0.gguf   | (not recorded)         | `aaa3374`   | `a105b99`  | 512 / 4 (ch1)| 38.4383      | 308.0419      | +269.6 | 701 %        | ≤10 %   | **FAIL** | KnackAU  |
| 2026-04-20 | qwen3     | ship        | Qwen3-8B-Q8_0.gguf                 | `60F232FBBDB88A36…`    | `aaa3374`   | `a105b99`  | 2048 / 8     |  9.8035      | 10.3076       | +0.504 |  5.14 %      | ≤5 %    | **FAIL** (edge, +0.14 pp over budget) | KnackAU |
| 2026-04-20 | qwen3     | sqfree      | Qwen3-8B-Q8_0.gguf                 | `60F232FBBDB88A36…`    | `aaa3374`   | `a105b99`  | 2048 / 8 (ch3) |  9.8035    | 57.82         | +48.0  | 490 %        | ≤10 %   | **FAIL** | KnackAU  |
| 2026-04-20 | qwen3     | sqfree+spinor+cauchy2 | Qwen3-8B-Q8_0.gguf       | `60F232FBBDB88A36…`    | `aaa3374`   | `a105b99`  | 2048 / 8 (aborted pre-chunk 1) | 9.8035 | —      | —      | —            | ≤15 %   | **PARTIAL** (run didn't complete) | KnackAU |
| 2026-04-21 | llama-3   | ship        | Dolphin3.0-Llama3.2-1B.Q8_0.gguf   | (not recorded)         | `bbac74f`   | `9446574`  | 2048 / 8     | 11.6150      | 12.5629       | +0.948 |  8.16 %      | ≤5 %    | **FAIL** (1B regime — scaling law dominates) | KnackAU |
| 2026-04-21 | phi3      | ship        | Phi-3.1-mini-4k-instruct-Q8_0.gguf | (not recorded)         | `b1db3e9`   | `acf9d7e`  | 2048 / 8     |  5.0297      |  5.1523       | +0.123 |  2.44 %      | ≤5 %    | **PASS** — flipped to `SP_PRESET_CALIBRATED`. Engine SHA `acf9d7e` = OI1 fused-QKV + packed-SwiGLU loader+forward + `build_block_decode` packed-FFN crash fix + dbrx tokenizer allow. | KnackAU |
| 2026-04-21 | gemma3    | ship        | gemma-3-12b-it-Q3_K_L.gguf         | `A8E666D198D3C7D1…`    | `f7c5334`   | `021e297`  | 2048 / 8 (ch2 aborted) | 6.7922 | 3.918e+25    | +3.9e25 | 5.8e26 %     | ≤5 %    | **FAIL** (catastrophic chained-decode drift — `cache_ppl` forward-full K_corr=0.990, V_corr=0.960, PPL=6.79 matches baseline, so compression round-trip is healthy; drift is in the decode() accumulator over 1023 chained steps without Cauchy reset). Engine SHA `021e297` = OI2c K-quant getrows patch (f487fe0) + cgraph-size fix (this commit) — both required just to get the run past model-load and decode()-first-call. Same pattern as the qwen3 sqfree / llama-3 sqfree catastrophic FAILs: the preset's bit-budget is too aggressive for a model this deep without a reset mechanism. | KnackAU |

Supporting logs for every row above live under
`archive/eval/logs/` on the maintainer's machine (the `archive/` tree is
untracked and local-only — see the top-level workspace `.gitignore`).

## Status of every preset (2026-04-21)

| Preset      | Decision after ledger | Notes |
|-------------|------------------------|-------|
| qwen3-next  | `PROVISIONAL`          | No calibration run yet. 35B model needs heavy CPU-offload or Colab. |
| qwen3-moe   | `PROVISIONAL`          | No reference checkpoint on disk. Will calibrate when one lands. |
| qwen3       | `PROVISIONAL` (kept)   | Ship path misses the 5 % budget by 0.14 pp; sqfree catastrophic. Preset is still the best shipping default for Qwen3 dense — flipping to `CALIBRATED` would require either a bit-budget revision (doc work, not math) or a tighter preset that trades compression for fidelity. Until one of those lands the preset stays `PROVISIONAL`. |
| gemma4      | `PROVISIONAL`          | Engine doesn't register `gemma4` arch yet (same story as gemma3 below); no forward graph, nothing to calibrate. |
| gemma3      | `PROVISIONAL` (kept, ship ledger row landed 2026-04-21) | Ship path ran end-to-end on `lmstudio-community/gemma-3-12b-it-Q3_K_L.gguf` at ctx=2048 / chunks=8 and failed catastrophically (PPL blew past 1e+25 by chunk 2, aborted). Encouragingly, the forward-full `cache_ppl` path matches baseline (PPL=6.79, K_corr=0.990, V_corr=0.960), so the compression kernel on gemma3 is healthy — the failure mode is chained-decode drift over 48 layers × 1023 steps without a reset mechanism. Enabling Cauchy reset (`--cauchy-mode 2`) or reducing the K bit-budget are the two obvious next sweeps; neither is on the critical path for v1.x ship. Running the calibration also required two vendor patches now committed in engine: `ggml-cuda-getrows-kquant.patch` (K-quant dispatch, SHA `f487fe0`) to load a Q3_K_L model at all, and the `forward.cpp` cgraph-size fix (SHA `021e297`) to stop gemma3's 48-layer decode from tripping `ggml.c:6886:GGML_ASSERT(cgraph->n_nodes < cgraph->size)`. Both fixes are general — same set unblocks `phi-4-Q4_K_M`, `Phi-3.1-mini-128k-Q4_K_M` (K-quant embed tables), and any deep-stack model. |
| phi3        | `CALIBRATED` (2026-04-21) | Ship +2.44 % drift on Phi-3.1-mini-4k-Q8_0 at ctx=2048, chunks=8. OI1 fused-QKV + packed-SwiGLU loader+forward landed in engine + `build_block_decode` FFN fix. `phi-4-Q4_K_M` / `Phi-3.1-mini-128k-Q4_K_M` follow-up runs previously blocked on K-quant getrows; unblocked by `ggml-cuda-getrows-kquant.patch` pending rebuild. |
| llama-3     | `PROVISIONAL`          | 1B ship-path measurement is dominated by the model-scale regime (see CALIBRATION_FINDINGS §2). Re-measure at ≥ 7B before making a call — task #27. |

## When a preset passes

Flip the preset's `.status` to `SP_PRESET_CALIBRATED` in
`core/shannon_prime_modelpack.c`, append the promotion row to the ledger
above, update the summary table in
[MODEL-PACK.md](MODEL-PACK.md#current-calibration-status-2026-04-21), and
include the ledger row hash in the commit message so the audit trail is
one `git log --grep` away.

## When a preset fails

Append the FAIL row anyway. Then decide:
- **Small miss (within 2× of budget):** try a tighter bit allocation on the
  same arch before giving up. Document the sweep in
  `archive/eval/` and append the winning config as a new row.
- **Catastrophic miss (like the sqfree cases above):** the problem isn't
  the preset — it's either (a) the compression path itself on this model
  family, or (b) a code-level regression in the compression kernel. File
  a separate issue and do not chase the preset number in isolation.

The `PROVISIONAL` tag is not a bug — it is the honest label for a preset
whose numbers are reasoned guesses. This file is how we stop pretending
otherwise.
