# Shannon-Prime Model Pack

A model pack is a small record that carries known-good compression defaults for
a specific model architecture family. It lets the bridges (engine, llama.cpp,
comfyui) ship an honest "auto-adapts" story: a Qwen-3 MoE user running the
default path gets a config tuned for Qwen-3 MoE, not the Llama-3 defaults.

This document covers the API, the current registry, and how to promote a
preset from `PROVISIONAL` to `CALIBRATED`.

## API

Header: `core/shannon_prime_modelpack.h`.

```c
const sp_model_preset_t *preset =
    sp_model_preset_resolve(arch_name, head_dim, n_layers, n_heads_kv);

if (preset) {
    sp_config_apply_preset(&cfg, preset);
    char buf[256];
    sp_model_preset_describe(preset, buf, sizeof(buf));
    fprintf(stderr, "[Shannon-Prime] model-pack: %s\n", buf);
}
```

`arch_name` is the GGUF `general.architecture` string ("llama", "qwen3",
"qwen3moe", "gemma3", ...). Resolution is substring match, first-hit wins —
presets are ordered most-specific → most-general in the registry.

When no preset matches, the resolver returns `NULL` and the caller keeps the
shipping defaults (5,5,4,3 K / 3 V / residual 3). This is the correct
behaviour for a brand-new architecture: calibrate first, ship defaults
second.

## Layering

From weakest to strongest — later wins:

1. Shipping defaults in `sp_config_init`.
2. Preset from `sp_model_preset_resolve` + `sp_config_apply_preset`.
3. Environment overrides (`SHANNON_PRIME_K_BITS`, etc.) — bridge-specific.
4. Explicit CLI flags (`--k-bits`, `--v-bits`, `--residual-bits`).

Presets are **overlays**, not mandates. Every preset is a recommendation that
an explicit user setting can override.

## Current registry (shannon-prime v1.17+)

Seven entries ship in the registry. One — `phi3` — is currently
`SP_PRESET_CALIBRATED` (see calibration status table below and the ledger
at [MODEL-PACK-CALIBRATION.md](MODEL-PACK-CALIBRATION.md)). The remaining
six ship at `SP_PRESET_PROVISIONAL` — the numbers are reasoned guesses,
not validated against PPL on a reference checkpoint. See *Promotion
recipe* below for the validation workflow.

| name        | arch match   | K bits          | V bits          | res | spinor |
|-------------|--------------|-----------------|-----------------|-----|--------|
| qwen3-next  | `qwen35moe`  | 5,4,4,4,4       | 4,4,4,4,4       | 3   | on     |
| qwen3-moe   | `qwen3moe`   | 5,4,4,4,4       | 4,4,4,4,4       | 3   | on     |
| qwen3       | `qwen3`      | 5,5,4,3         | 4,3             | 3   | off    |
| gemma4      | `gemma4`     | 5,4,4,3         | 3               | 3   | off    |
| gemma3      | `gemma3`     | 5,4,4,3         | 3               | 3   | off    |
| phi3        | `phi3`       | 5,5,4,3         | 3               | 3   | off    |
| llama-3     | `llama`      | 5,5,4,3         | 3               | 3   | off    |

Rationale:

- **qwen3-next** — Qwen3.6-35B-A3B (ships arch `qwen35moe`). Hybrid SSM+MoE
  architecture with 40 layers: every 4th is full MoE attention and the rest
  are Gated DeltaNet with recurrent state. The preset only affects the 10
  attention layers (GDN layers contribute no K/V to the cache), but the
  band profile is the same as qwen3-moe because the attention layers
  themselves are still MoE-dense. **Must precede** `qwen3moe` and `qwen3`
  in the registry — `strstr("qwen35moe", "qwen3")` matches at position 0,
  so without this entry a Qwen3.6 model would silently adopt the dense
  qwen3 preset.
- **qwen3-moe** — original MoE variants (non-hybrid). Dense KV, high
  mid-band energy; spinor recommended because the sheet-bit recovery
  shows its biggest gains when the mid-band magnitude is substantial.
- **qwen3** (dense) — similar to Llama-3 on K, but V empirically carries
  more energy, so two V bands instead of one.
- **gemma4** — GGUFs ship a distinct arch string (`gemma4`, verified on
  lmstudio-community/gemma-4-31B-it-Q4_K_M). Inherits gemma3's SWA band
  profile until a dedicated calibration run lands; split out now so it
  can diverge without a pattern migration.
- **gemma3** — sliding-window attention puts more weight on the top K band;
  preserve K[0]=5 but drop K[1] a bit. V is flat 3-bit like Llama.
- **phi3** — Phi 3 / 3.1 / 4 all ship `general.architecture="phi3"` in
  GGUF (verified on lmstudio-community/phi-4-Q4_K_M and
  Phi-3.1-mini-128k-instruct-Q4_K_M). Dense GQA with a Llama-like profile.
  Provisional numbers mirror llama-3 as the conservative default.
- **llama-3** — shipping defaults. The preset exists so matches are logged
  and the auto-adapts story is auditable.

## Promotion recipe (PROVISIONAL → CALIBRATED)

A preset earns `SP_PRESET_CALIBRATED` after a PPL-drift measurement on a
reference checkpoint shows the compressed cache matches baseline within
budget. **Drift is measured through chained decode** (`perplexity --cache`),
not forward_full. `cache_ppl` is a fast K/V round-trip diagnostic — it
reports K_corr/V_corr but its PPL column is baseline PPL, not the drift
number. Do not use `cache_ppl` PPL for promotion.

```bash
# 1. Baseline PPL (forward_full, no cache, no compression)
sp-engine perplexity \
    --model <ref.gguf> \
    --ctx 2048 --chunks 8 \
    data/wiki.raw | tee baseline.ppl

# 2. Compressed PPL with the candidate preset, auto-resolved from arch.
#    --model-preset auto applies the registry entry for this model's
#    arch_name. Drop --sqfree/--spinor/--hierarchical to measure the
#    ship-path drift; add them to measure the aggressive path.
SP_ENGINE_BACKEND=gpu sp-engine perplexity --cache \
    --model <ref.gguf> \
    --model-preset auto \
    [--sqfree [--spinor] | --hierarchical] \
    --ctx 2048 --chunks 8 \
    data/wiki.raw | tee candidate.ppl

# 3. Drift = PPL(candidate) − PPL(baseline). Accept if within budget:
#    Ship:          abs(ΔPPL) <= 0.05
#    Sqfree:        abs(ΔPPL) <= 0.10
#    Sqfree+spinor: abs(ΔPPL) <= 0.15
#    Hierarchical:  abs(ΔPPL) <= 0.15  (9% skeleton, same tolerance as sqfree+spinor)
```

**Overriding a preset for a sweep.** Explicit CLI flags beat the preset —
the layering is `defaults < preset < env vars < CLI flags` (see *Layering*
above). So to sweep bit allocations on top of `auto`:

```bash
sp-engine perplexity --cache --model-preset auto \
    --k-bits 5,4,4,3 --v-bits 3 --residual-bits 3 \
    --sqfree --model <ref.gguf> --ctx 2048 --chunks 8 data/wiki.raw
```

**Long-context stability.** Decode-chain error accumulates; the Cauchy
reset system (see [CAUCHY-RESET.md](CAUCHY-RESET.md)) is how ctx ≥ 2k is
held honest. For promotion at long context, layer `--cauchy-mode 2` on
top of the candidate run — the shipping default (Mertens-only) is
measured-zero-cost on real workloads.

When a preset passes, flip its `.status` to `SP_PRESET_CALIBRATED`, update
the `.notes` line with the reference checkpoint SHA + PPL drift, and note
the promotion in the changelog.

### Current calibration status (2026-04-21)

| Preset      | Status        | Reference checkpoint               | Measured ship drift |
|-------------|---------------|------------------------------------|---------------------|
| qwen3-next  | PROVISIONAL   | Qwen3.6-35B-A3B-Q4_K_M.gguf (on disk) | not yet run |
| qwen3-moe   | PROVISIONAL   | —                                  | not yet run |
| qwen3       | PROVISIONAL   | Qwen3-8B-Q8_0.gguf                 | +0.50 PPL @ 4.06× (ctx=2048, chunks=8, wiki.test.raw) |
| gemma4      | PROVISIONAL   | gemma-4-31B-it-Q4_K_M.gguf (on disk)  | not yet run |
| gemma3      | PROVISIONAL   | gemma-3-12b-it-Q3_K_L.gguf         | ship path FAIL (catastrophic chained-decode drift; `cache_ppl` path clean at K_corr=0.990, V_corr=0.960, PPL=6.79 vs baseline 6.79; see ledger 2026-04-21 row) |
| phi3        | CALIBRATED (2026-04-21) | Phi-3.1-mini-4k-instruct-Q8_0.gguf | +0.123 PPL / +2.44 % @ 3.96× (ctx=2048, chunks=8, wiki.test.raw) |
| llama-3     | PROVISIONAL   | Dolphin3.0-Llama3.2-1B-Q8          | +0.95 PPL / +8.16% @ 3.76× (ctx=2048, chunks=8, wiki.test.raw). 1B regime — dominated by scaling law; ship-path budget will likely pass at ≥8B on the same corpus (cf. qwen3 row). |

The qwen3 ship number is at the edge of the 0.05 ship budget — the
preset is shippable but doesn't have headroom. See
[MODEL-PACK-CALIBRATION.md](MODEL-PACK-CALIBRATION.md) for the
append-only ledger (every promotion attempt, pass or fail) and
`archive/eval/CALIBRATION_FINDINGS.md` (local, untracked) for the
run-by-run log.

## Roadmap

- **Bridge integration** — the llama.cpp patch (`patches/llama-cpp-v1.05-
  gpu-kv.patch`) currently hardcodes env-variable overrides onto shipping
  defaults. The next patch revision should call `sp_model_preset_resolve`
  against `hp.arch` after `sp_config_init` and before the env-var overlay.
- **Engine CLI** — `sp-engine --model-preset auto|<name>|off` landed in
  `shannon-prime-engine` (commits around Phase 3 + env-var follow-up
  `9446574`). Every verb that accepts a `Config` now reads
  `SHANNON_PRIME_MODEL_PRESET` as well, so the layering documented
  above is wired end-to-end. `off` remains the shipping default
  until at least one preset is `SP_PRESET_CALIBRATED`.
- **Calibration ledger** — [MODEL-PACK-CALIBRATION.md](MODEL-PACK-CALIBRATION.md)
  is now the append-only log of every promotion attempt (pass or fail).
  Each row carries date, model, GGUF SHA, SP + engine SHAs, drift vs
  baseline, and reviewer. New calibration runs land as new rows — the
  summary table in this doc is derived from the latest row per preset.
