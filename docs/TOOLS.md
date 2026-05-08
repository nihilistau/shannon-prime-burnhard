# Shannon-Prime: Tools Reference

## Overview

Command-line tools for working with Shannon-Prime outside of the runtime integration.

**Production / deployment:**

| Tool | Purpose | Requires Model? |
|------|---------|-----------------|
| `sp_benchmark.py` | Validate VHT2 math, measure compression quality | No |
| `sp_inject_freqs.py` | Inject lattice RoPE frequencies into GGUF models | Yes (GGUF) |
| `sp_compress_model.py` | Analyze/compress model weights via VHT2 | Yes (safetensors) |
| `sp_scaling_law.py` | K_corr → PPL design rule (pre-bench filter) | No |
| `sp_auto_bands.py` | Fit `SHANNON_PRIME_K_BITS` from a live K-dump | Needs a dump file |

**Research / diagnostic** (behind v1.14/v1.15 findings; used to produce the adaptive-calibration and hierarchical-predictor results):

| Tool | Purpose | Requires Model? |
|------|---------|-----------------|
| `extract_kv.py` | Capture post-RoPE K vectors from a live model run → `.npz` | Yes (GGUF or HF) |
| `sp_chord_diagnostic.py` | Prime-chord analysis per head and per layer on a K dump | Needs `.npz` dump |
| `sp_regime_analysis.py` | Two-regime (early/late) reconstruction quality per skeleton strategy | Needs `.npz` dump |
| `sp_hierarchical_skeleton.py` | Validate the hierarchical Vilenkin predictor (`sp_hier_cache_t`) | Yes (GGUF) |
| `sp_compress.py` | End-to-end KV compression round-trip: extract, compress, inject, compare | Yes (HF transformers) |
| `benchmark_ab.sh` | A/B compare ship vs sqfree/hierarchical via `sp-engine` binary | Optional (synthetic + real) |

---

## sp_benchmark.py — Compression Benchmark

Demonstrates and measures VHT2 KV cache compression on synthetic vectors. This is the first thing to run — it validates that the compression math works and shows the K/V spectral asymmetry that makes Shannon-Prime possible.

No model required. No GPU required. Runs in seconds.

### Usage

```bash
# Full benchmark across all standard configs
python tools/sp_benchmark.py

# Spectral energy analysis — see VHT2 concentration
python tools/sp_benchmark.py --spectral

# Mobile head dimension
python tools/sp_benchmark.py --head-dim 64

# Test a specific bit allocation
python tools/sp_benchmark.py --config 5,5,4,3

# More vectors for stable statistics
python tools/sp_benchmark.py --n-vectors 2000
```

### What the Output Shows

**Spectral Analysis** (`--spectral`):

```
  RoPE-like K:
    Band    Energy% Bar
       0       3.6%  █
       1      76.7%  ██████████████████████████████████████
       2       1.9%
       3      17.8%  ████████
  First half: 80.3% (concentrated → banded quant helps)

  Random V:
    Band    Energy% Bar
       0      25.6%  ████████████
       1      38.0%  ███████████████████
       2      19.8%  █████████
       3      16.5%  ████████
  First half: 63.6% (uniform → flat quant OK)
```

This is the foundational observation. K vectors (with RoPE periodicity) concentrate 80%+ of their VHT2 energy in the first two bands. V vectors (random content) spread energy uniformly. This asymmetry is why K gets banded quantization (more bits where energy is) and V gets flat quantization (all bands equally important).

**Compression Table**:

Shows correlation, compression ratio, bytes per vector, and microseconds per vector for each bit allocation config. Look for:

- Correlation >0.99 on ship config (5/5/4/3)
- Möbius improvement is positive (typically +0.003)
- Flat beats banded for V vectors

**Memory Projection**:

Shows actual GB savings for a real model configuration (32K context, 32 layers, 8 KV heads).

### Dependencies

PyTorch only (for the torch backend). No model files needed.

---

## sp_inject_freqs.py — GGUF Frequency Injection

Blends prime-lattice-aligned RoPE frequencies into an existing GGUF model. This corrects positional frequency mismatch — a structural error in standard geometric RoPE that is independent of weight quantization precision.

**Zero retraining.** The modified model loads and runs normally in llama.cpp. The only difference is which RoPE frequencies are used during inference.

### Paper Results

| Model | Quant | Baseline PPL | Injected PPL | α | Improvement |
|-------|-------|-------------|-------------|---|-------------|
| Dolphin 3.0 Llama 3.2 1B | Q8_0 | 11.6413 | 11.5462 | 0.22 | −0.82% |
| Dolphin 3.0 Llama 3.2 1B | Q6_K | 11.7615 | 11.6843 | 0.17 | −0.66% |
| Dolphin 3.0 Llama 3.2 1B | Q4_K_M | 12.2380 | 12.1630 | 0.17 | −0.61% |
| Qwen2.5-3B | — | — | — | 0.12 | −0.20% |
| Phi-3.1-3.8B | — | — | — | 0.05 | −0.02% |

Optimal α range: 0.15–0.22 (flat optimum, deployment-robust). Higher `rope_freq_base` → wider harmonic gaps → more room for injection.

### Usage

```bash
# Show model's RoPE configuration
python tools/sp_inject_freqs.py model.gguf --info

# Analyze what injection would do (no file modification)
python tools/sp_inject_freqs.py model.gguf --analyze --alpha 0.17

# Inject and produce output
python tools/sp_inject_freqs.py model.gguf model_sp.gguf --alpha 0.17

# Use prime frequencies instead of composite
python tools/sp_inject_freqs.py model.gguf model_sp.gguf --alpha 0.22 --tier-mode prime
```

**v1.03 change:** the tool now writes (or replaces) a real shared `rope_freqs.weight` tensor in the output GGUF — llama.cpp's `LLM_TENSOR_ROPE_FREQS` resolves to the unqualified root name even though the category is LAYER_REPEATING, so every layer reads the same tensor. No runtime env var or sidecar consumer is required; loading `model_sp.gguf` is enough for the injected frequencies to take effect.

A companion `.sp_freq_factors.bin` sidecar is still written next to the output for debugging and for the alternate `SHANNON_PRIME_SIDECAR=<path>` loader (see `patches/llama-cpp-v1.03-sidecar-kdump.patch`), which patches existing rope-factor tensors in-place at context init via `ggml_backend_tensor_set`.

### Alpha Sweep

To find the optimal α for your specific model:

```bash
# PowerShell
for ($a=0.05; $a -le 0.30; $a+=0.05) {
    python tools/sp_inject_freqs.py model.gguf "model_a$a.gguf" --alpha $a
}
# Then benchmark each with llama.cpp perplexity

# Bash
for a in 0.05 0.10 0.15 0.17 0.20 0.22 0.25 0.30; do
    python tools/sp_inject_freqs.py model.gguf model_a${a}.gguf --alpha $a
done
```

### How It Works

Standard RoPE uses geometric frequencies: θ_j = base^(−2j/d). This creates redundant coverage at some scales and gaps at others. Shannon-Prime replaces a fraction (α) of these frequencies with lattice-aligned values distributed across three tiers:

| Tier | Head Range | Frequency Range | Scale |
|------|-----------|-----------------|-------|
| Local (25%) | First quarter | 2–101 | Word/syntax |
| Mid (33%) | Middle third | 101–1009 | Clause/paragraph |
| Long (42%) | Last 42% | 1009–8209 | Section/document |

The tiered allocation preserves per-head scale diversity (which is load-bearing — uniform replacement caused +370% degradation in the paper).

### Output Files

The tool produces two files:

1. **`output.gguf`** — Copy of input model (currently metadata-identical; full GGUF tensor modification for embedding freq_factors directly is the next integration step)
2. **`output.sp_freq_factors.bin`** — Binary file containing the freq_factors array (n_freqs × float32). This companion file is loaded by the Shannon-Prime llama.cpp integration layer.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--alpha` | 0.17 | Blending ratio. 0 = pure geometric, 1 = pure lattice. |
| `--tier-mode` | composite | `prime` (lattice generators) or `composite` (lattice coordinates). Paper showed identical results. |
| `--info` | — | Print model RoPE info and exit |
| `--analyze` | — | Show per-frequency analysis without modifying |
| `--quiet` | — | Suppress verbose output |

### Dependencies

`gguf` Python package (`pip install gguf`), NumPy.

---

## sp_compress_model.py — Weight Compression Analysis

Analyzes and optionally compresses transformer model weights using VHT2 spectral quantization. This is the research tool that answers the question: do model weights have the same spectral structure as KV cache vectors?

### The Hypothesis

W_K and W_Q weight matrices are trained with RoPE applied. The learned weights implicitly encode the frequency patterns they were optimized for. If the rows of W_K show VHT2 spectral concentration similar to K vectors at inference time, VHT2 weight quantization will outperform generic quantization on those specific tensors.

W_V, W_O, and FFN weights were NOT trained with RoPE — they shouldn't show this structure, and they shouldn't benefit from VHT2.

### Usage

```bash
# Analyze a HuggingFace model directory (safetensors format)
python tools/sp_compress_model.py --analyze path/to/model/

# Specify head dimension explicitly
python tools/sp_compress_model.py --analyze path/to/model/ --head-dim 64

# Analyze with custom bit allocation
python tools/sp_compress_model.py --analyze path/to/model/ --k-bits 4,4,4,3
```

### What the Analysis Shows

```
Tensor                                        Shape                Conc%  VHT2   Flat      Δ
───────────────────────────────────────────── ──────────────────── ────── ────── ────── ───────
  model.layers.0.self_attn.q_proj.weight      [4096, 4096]         72.3% 0.994  0.991 +0.0030 ◀
  model.layers.0.self_attn.k_proj.weight      [1024, 4096]         74.1% 0.995  0.990 +0.0050 ◀
  model.layers.0.self_attn.v_proj.weight      [1024, 4096]         51.2% 0.991  0.991 +0.0000
  model.layers.0.self_attn.o_proj.weight      [4096, 4096]         53.8% 0.990  0.991 -0.0010
```

For each attention weight tensor, the tool reports:

- **Conc%**: VHT2 spectral concentration (first half energy). >60% indicates structure VHT2 can exploit.
- **VHT2**: Correlation after VHT2 compression (VHT2 → Möbius → 5/5/4/3 banded quant → reconstruct).
- **Flat**: Correlation after flat quantization (same total bits, uniform 4/4/4/4 allocation).
- **Δ**: VHT2 advantage. Positive = VHT2 is better. Marked with ◀ when significant.

**What to look for:**

- Q_proj and K_proj should show higher concentration (>65%) and positive Δ
- V_proj and O_proj should show ~50% concentration and near-zero Δ
- If Q/K don't show structure, VHT2 weight compression won't help for this model

### Compression Mode (Research)

Full weight compression (producing a modified model with VHT2-compressed Q/K weights) is implemented in code but requires a custom dequantization path at inference time. The analysis mode is the validated, ship-ready tool. The compression path is the next integration piece.

### Dependencies

PyTorch, safetensors (`pip install torch safetensors`).

---

## Tool Workflow

The tools form a progression:

```
1. sp_benchmark.py          Does VHT2 work at all?
   (no model needed)        → Confirms spectral asymmetry and compression quality

2. sp_inject_freqs.py       Can we improve this model's RoPE?
   (needs GGUF model)       → Produces freq-injected model for PPL testing

3. sp_compress_model.py     Do this model's weights have spectral structure?
   (needs HF safetensors)   → Identifies which tensors benefit from VHT2

4. sp_scaling_law.py        Is this K_corr / bits / params combo worth running?
   (no model)               → Predicts ΔPPL so you can skip doomed configs

5. sp_auto_bands.py         What bit allocation does THIS model actually want?
   (needs K dump)           → Fits K_BITS to measured per-band variance
```

For production deployment, `sp_inject_freqs.py` is the immediate tool — it produces a GGUF file that runs in unmodified llama.cpp with measurable PPL improvement. The VHT2 KV cache compression (the core Shannon-Prime system) operates at inference time via the shannon-prime library integration, not via model file modification.

---

## sp_scaling_law.py — K_corr → PPL design rule

Empirically-fitted formula across 9 measured configurations (±20%):
```
log(PPL / PPL_base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)
```

Use as a pre-bench filter. Before you burn an hour running perplexity, call `predicted_ppl_ratio(k_corr, params_b, bits)` — if the predicted ΔPPL is >5%, don't bother; pick a wider bit allocation or a bigger model. See `test_sqfree.py` for example usage.

---

## sp_auto_bands.py — Fit K_BITS from a measured K dump

Picks a `SHANNON_PRIME_K_BITS` allocation by measuring per-band VHT2 variance on a warmup K dump. Closes the loop between "you have a real model" and "what allocation does its K actually want".

### Workflow

**Step 1 — Produce a K dump from a warmup run.** The llama.cpp integration patch (`patches/llama-cpp-v1.03-sidecar-kdump.patch`) adds a `SHANNON_PRIME_DUMP_K=<path>` env that stream-writes post-RoPE K vectors to disk inside the post-decode hook:

```bash
SHANNON_PRIME_ENABLED=1 \
SHANNON_PRIME_SQFREE=1 SHANNON_PRIME_SPINOR=1 \
SHANNON_PRIME_DUMP_K=/tmp/k_dump.bin \
SHANNON_PRIME_DUMP_K_LIMIT=8192 \
llama-perplexity -m model.gguf -f wiki.test.raw -c 512 --chunks 1
```

The dump is raw fp32, `head_dim` elements per vector (pre-pad — the hook captures K as it arrives at the cache, not post-sqfree-pad).

**Step 2 — Fit the allocation.**

```bash
python tools/sp_auto_bands.py \
    --dump /tmp/k_dump.bin \
    --head-dim 128 \
    --n-bands 10 \
    --total-bits 35 \
    --min-bits 3 --max-bits 6
# → "4,4,4,4,4,3,3,3,3,3"
```

Defaults enforce the **3-bit floor** from CLAUDE.md invariant #3 (2-bit bands are catastrophic — measured PPL 428.78 on Qwen3-8B 10×2, see `logs/cuda_qwen3_sqfree_10band_2.json`). Override `--min-bits` at your own risk.

**Step 3 — Use it.**

```bash
export SHANNON_PRIME_K_BITS=$(python tools/sp_auto_bands.py \
    --dump /tmp/k_dump.bin --head-dim 128 --n-bands 10 --total-bits 35)
./llama-perplexity ...
```

### How the allocator works

Per-band VHT2 energy → log-weight allocation, clipped to `[min_bits, max_bits]`, rebalanced so the bits sum to `total_bits` exactly. Output can also be emitted as JSON with `--json` (band energies + chosen bits + metadata).

### Known limitation

For the sqfree+spinor aggressive path, the runtime operates on a **sqfree-padded** Vilenkin basis (pad_dim=154 for hd=128) while the dump captures the **unpadded** raw K. The analyser's Walsh-at-hd transform is a monotonic-decay proxy; the ordering of the allocation holds but the absolute magnitudes aren't comparable. Dumping the post-pad Vilenkin coefficients is a deferred v1.05 item — see `logs/vulkan_diagnostic_v1.03.json` for context.

---

# Research / diagnostic tools

These produced the findings behind v1.14 (adaptive calibration +
L/2 skeleton default) and v1.15 (hierarchical Vilenkin predictor).
Not on the hot path for deployment, but load-bearing for the paper
and for validating any change to the C-core compression code.

The common data format is an `.npz` file with a `k_vectors` array
of shape `(n_layers, n_heads, n_positions, head_dim)` — produced
by `extract_kv.py`, consumed by every analyser below.

---

## extract_kv.py — Capture K vectors from a model

Runs a prompt through a model, captures the post-RoPE K projections
at every (layer, head, position), writes them to `.npz` for the
analyses below. Supports GGUF models (via llama-cpp-python) and
HuggingFace models (via transformers + PyTorch).

```bash
# GGUF (auto-detected by .gguf extension)
python tools/extract_kv.py --model path/to/model.gguf --output kv.npz

# GGUF with GPU offload
python tools/extract_kv.py --model path/to/model.gguf --n-gpu-layers 99 \
    --output kv.npz

# HuggingFace (local or hub id)
python tools/extract_kv.py --model microsoft/Phi-3-mini-4k-instruct \
    --output kv_phi3.npz --prompt-file long_prompt.txt

# Longer prompts → more positions → tighter per-layer / per-head statistics
python tools/extract_kv.py --model model.gguf --prompt-file wiki.test.raw \
    --max-positions 4096 --output kv_wiki4k.npz
```

Output is consumed verbatim by `sp_chord_diagnostic.py`,
`sp_regime_analysis.py`, and `sp_compress.py`. The `.npz` carries
metadata (`model_name`, `n_layers`, `n_heads`, `head_dim`) so
downstream tools don't need to re-infer them.

---

## sp_chord_diagnostic.py — Prime chord analysis

Tests the Prime Chord hypothesis: attention heads in a trained
transformer activate small subsets of primes (Z/pZ harmonic
components) from the sqfree-aligned Vilenkin basis, and those
subsets form "chords" that evolve with layer depth.

Produces:

* **Per-head prime histograms** — which primes each head uses.
* **Chord entropy curve** across layers — mid-layers diversify,
  late layers collapse toward a small stable set.
* **Jaccard adjacency** per layer — heads that share primes.
* **Cross-layer persistence** — which chords survive depth.
* **Ghost Head classification** — heads that live off the
  arithmetic manifold, probed at extended primes (p=13, 17, 19).

```bash
python tools/sp_chord_diagnostic.py --input kv.npz --output chord.json
python tools/sp_chord_diagnostic.py --input kv.npz --plot chord.png
```

This tool produced the observation that late-layer chord collapse
is the signal the hierarchical predictor exploits — a small
low-prime sub-projection determines the high-prime refinement.

---

## sp_regime_analysis.py — Two-regime reconstruction

Empirical finding (published in `docs/Shannon-Prime.md` and the
paper): transformers have two spectral regimes.

* **Early layers (0 → ~L/2):** clean T² manifold. p=2 dominates
  ~30%, p=3 dominates ~22%, rest is tail. Compressible by
  algebraic (divisibility) skeletons.
* **Late layers (~L/2 → L):** diffuse spectrum. p=3 drops toward
  p=5 / p=7, approaching uniform. Algebraic skeletons fail,
  adaptive (variance-ranked) skeletons win.

This tool runs three competing skeleton selection strategies at
matched compression and reports relative L2 reconstruction error
per layer:

1. **T² Algebraic** — keep indices divisible by 2 or 3.
2. **Adaptive Top-K** — keep top-K indices by measured variance.
3. **T³ Algebraic** — keep indices divisible by 2, 3, or a
   secondary prime.

```bash
python tools/sp_regime_analysis.py --input kv_phi3.npz --sqfree
python tools/sp_regime_analysis.py --input kv_qwen3.npz --sqfree \
    --output regime.json
```

Produced the "T² algebraic skeleton never beats adaptive
(0/476 wins)" finding that motivated the v1.14 default-skeleton
change (75% → L/2 variance-ranked). Also produced the paper-side
"context-length compressibility is architecture-dependent"
finding (Phi-3 late layers concentrate with long context;
Qwen3 slightly disperses), which informs per-model compression
budget selection.

---

## sp_hierarchical_skeleton.py — Hierarchical predictor research

The validation harness behind `sp_hier_cache_t` (v1.15). Tests
whether the low-prime sub-projections of the Vilenkin basis can
linearly predict the high-prime refinement, using calibration
data from real K vectors.

The Vilenkin basis on sqfree-padded dimensions has Kronecker
structure:

```
hd=128 → pad=154 = 2 · 7 · 11 → Z/2Z × Z/7Z × Z/11Z
```

The script partitions the pad_dim coefficients into a **skeleton**
(the Z/2Z × Z/7Z subgroup, 14 coefficients, ~9%) and a **refinement**
(the other 140 coefficients, ~91%). It then fits a ridge-regression
linear map from skeleton → refinement per (layer, head) slot.

If the predictor is good, you can compress to:

* 14 skeleton coefficients, banded-quantised (~70 bits)
* a 14×140 matrix per slot, calibrated once (~31 KB, amortised)
* a tiny residual correction on the predicted 140

That's 9% skeleton instead of 50% — the architecture
shannon-prime v1.15's `sp_hier_cache_t` ships.

```bash
python tools/sp_hierarchical_skeleton.py model.gguf
python tools/sp_hierarchical_skeleton.py model.gguf --n-tokens 256
```

Prints per-slot predictor quality (R² against measured refinement)
plus the aggregated reconstruction quality under quantisation
budgets.

---

## sp_compress.py — End-to-end KV compression round-trip

The most integrated research tool: loads a GGUF, runs a prompt,
extracts the full KV state, applies VHT2 + adaptive top-K
skeleton compression, reconstructs via inverse VHT2, **injects
the compressed cache back into the model**, and generates
continuation tokens with both the original and compressed caches.
Compares perplexity and token agreement.

```bash
# Basic round-trip at 30% skeleton
python tools/sp_compress.py --model model.gguf --skeleton-frac 0.30

# Sweep compression ratios
python tools/sp_compress.py --model model.gguf \
    --skeleton-fracs 0.10,0.25,0.50,0.75 \
    --prompt "The quick brown fox"
```

Answers the only question that matters for this whole stack:
"does spectral KV compression actually preserve generation
quality on a real model?" — which is the question
`shannon-prime-engine`'s `perplexity --cache` verb now also
answers end-to-end for GGUF-quantised models, without the
HuggingFace round-trip.

---

## benchmark_ab.sh — A/B bench via sp-engine

Wrapper around the `sp-engine` binary that runs three
compression configs head-to-head and summarises the results:

* **Config A** — shadow + Möbius reorder + default bands (ship path)
* **Config B** — sqfree + variance-ranked L/2 skeleton
* **Config C** — hierarchical Vilenkin predictor (~9% skeleton)

Two phases:

1. `kv_smoke` — synthetic Gaussian data (no model required), fast.
2. `cache_ppl` — real model perplexity + K/V correlation + the
   scaling-law numerator, real data (needs `--model` + `--textfile`).

```bash
SP_ENGINE=/path/to/sp-engine.exe ./benchmark_ab.sh \
    --head-dim 128 --n-tokens 64 --n-head-kv 8 --n-layer 4

# Full three-way comparison with a model:
SP_ENGINE=/path/to/sp-engine.exe ./benchmark_ab.sh \
    --model dolphin_1b.gguf --textfile wiki.test.raw \
    --ctx 512 --chunks 2
```

This is the script that produced the **0/476 wins for T² algebraic
skeleton vs adaptive** cross-model cross-compression comparison
cited in the v1.14 commit message.
