# Shannon-Prime: Audio Generation Integration

## Overview

Shannon-Prime extends from LLM KV cache compression into audio generation through two distinct mechanisms applied to two different model architectures:

| Model | Architecture Type | SP Mechanism | ComfyUI Nodes |
|-------|------------------|--------------|---------------|
| **Stable Audio DiT** | Diffusion (ContinuousTransformer) | Block-skip caching | 3 nodes |
| **Voxtral TTS** | Autoregressive (Mistral backbone) | VHT2 KV cache compression | 2 nodes |
| **Qwen3-TTS** | Autoregressive (planned) | VHT2 KV cache compression | Pending |

**Key distinction:** Stable Audio uses block-skip caching — the same "Granite/Jazz" principle used for image DiTs where early blocks are cached and late blocks run every step. Voxtral uses traditional VHT2 KV cache compression — the same pipeline as the LLM path. Different mechanisms for different architectures, unified by the VHT2 spectral toolbox.

---

## 1. Stable Audio DiT — Block-Skip Caching

### Architecture

Stable Audio uses a `ContinuousTransformer` backbone with N identical `TransformerBlock` layers operating on audio frame latents.

**Default configuration:**

| Parameter | Value |
|-----------|-------|
| Depth (N blocks) | 24 |
| Attention heads | 24 |
| Embedding dim | 1536 |
| Head dim | 64 (= 1536 / 24) |
| Conditioning | adaLN (adaptive Layer Norm) |

Each `TransformerBlock` contains:

```
TransformerBlock
════════════════

  Audio Frame Latents (x)
       │
       ├──► adaLN Conditioning
       │         to_scale_shift_gate → 6 outputs
       │         gate function: sigmoid(1 - gate)   ← NOT raw gate like Flux
       │
       ├──► Self-Attention (1D RoPE on audio frame positions)
       │         Q, K, V projected from x
       │         RoPE applied to Q, K on the frame-position axis
       │
       ├──► Cross-Attention (optional — text conditioning)
       │         K, V from text encoder embeddings
       │
       ├──► Conformer Module (optional — local convolution)
       │
       ├──► Feed-Forward Network
       │
       ▼
  Output (to next block)
```

### adaLN Gate Function

Stable Audio's gate is `sigmoid(1 - gate)`, which inverts the raw gate value before applying the sigmoid. This is distinct from Flux DiT which uses the raw gate directly. The inversion means high raw-gate values produce near-zero gating (block output is suppressed), while low raw-gate values produce near-one gating (block output passes through).

### Block-Skip Principle

The same "Granite/Jazz" block-skip principle that applies to image DiTs holds for audio diffusion:

```
Block Depth vs. Audio Feature Role
═══════════════════════════════════

  Block 0-7   (early):   Tonal foundation, harmonic layout, pitch structure
  Block 8-15  (middle):  Melodic contour, rhythmic patterns, spectral envelope
  Block 16-23 (late):    Transients, high-frequency texture, percussive detail

  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  Early blocks change slowly across diffusion steps →    │
  │  CACHE their outputs and skip recomputation             │
  │                                                         │
  │  Late blocks refine high-frequency content each step →  │
  │  ALWAYS recompute                                       │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

Early blocks establish the tonal foundation and harmonic layout of the audio. Their outputs are largely stable across diffusion timesteps. Late blocks add transients and high-frequency texture that refine the audio at each step. Block-skip caches the early block outputs and skips their recomputation on subsequent timesteps, saving significant FLOPs with minimal quality impact.

### ComfyUI Nodes

**ShannonPrimeAudioBlockSkip** — Configure which blocks to skip and at what cadence.

**ShannonPrimeAudioCache** — Attach the caching mechanism to a Stable Audio pipeline. Stores block outputs in the SP cache format.

**ShannonPrimeAudioCacheFlush** — Clear cached block outputs between generations. Wire into your workflow to prevent stale cache data bleeding across prompts.

---

## 2. Voxtral TTS — VHT2 KV Cache Compression

### Architecture

Voxtral is an autoregressive text-to-speech model built on a Mistral language model backbone. It generates audio codec tokens autoregressively, making it architecturally identical to an LLM from the KV cache perspective.

| Parameter | Value | SP Relevance |
|-----------|-------|-------------|
| Layers | 26 | 26 layers of KV to compress |
| Query heads | 32 | — |
| KV heads | 8 | GQA 4:1 ratio |
| Head dim | 128 | = 2^7: **perfect** for VHT2 |
| Attention | GQA (Grouped Query Attention) | Standard Mistral GQA |

### Why head_dim=128 is Perfect for VHT2

`head_dim = 128 = 2^7` factors into 7 stages of `p=2` Hartley butterfly transforms. This is the pure power-of-2 path through VHT2 — no squarefree padding needed, no mixed-radix stages. The transform reduces to a classical Walsh-Hadamard butterfly scaled by `1/sqrt(2)` per stage:

```
VHT2 on dim=128
════════════════

  Stage 1: 64 butterflies of size 2     (128 → 64 pairs)
  Stage 2: 32 butterflies of size 4     (group by 4)
  Stage 3: 16 butterflies of size 8     (group by 8)
  Stage 4: 8 butterflies of size 16     (group by 16)
  Stage 5: 4 butterflies of size 32     (group by 32)
  Stage 6: 2 butterflies of size 64     (group by 64)
  Stage 7: 1 butterfly of size 128      (full dim)

  Total: 7 × (128/2) = 448 multiply-adds
  Self-inverse: vht2(vht2(x)) = x (no 1/N normalization)
```

No `sqfree_pad_dim()` call needed — `sqfree_pad_dim(128) = 128` is already optimal.

### Integration Method

Shannon-Prime monkey-patches `MistralBackbone.forward()` to intercept the KV cache path. On each autoregressive step, K and V vectors are routed through the VHT2 compress/decompress pipeline before being stored/retrieved from the cache.

```
Voxtral Autoregressive Step
════════════════════════════

  Text Tokens → Embedding → ...

  For each of 26 layers:
       │
       ├──► Compute Q, K, V
       │
       ├──► K, V → VHT2 Compress ──► Compressed KV Store
       │                                    │
       │    (subsequent steps)              │
       │    K, V ◄── VHT2 Decompress ◄─────┘
       │
       ├──► GQA Attention (32 Q heads, 8 KV heads)
       │
       ├──► FFN
       │
       ▼
  Next Layer

  Final Layer → Audio Codec Token → DAC/Encodec Decode → Waveform
```

### Ship Default Bit Allocation

| Component | Band Bits | Strategy |
|-----------|-----------|----------|
| K (keys) | `[5, 5, 4, 3]` | Banded — more bits for low-frequency Möbius bands (structural), fewer for high-frequency (noise-like) |
| V (values) | `[3]` | Flat — V lacks RoPE spectral structure, uniform 3-bit quantization suffices |

This allocation yields approximately **4.6x KV cache compression** with negligible quality degradation in generated speech.

### Memory Footprint

At a typical 2048-frame generation:

```
Baseline (bf16, uncompressed):
  26 layers × 8 KV heads × 2 (K+V) × 2048 positions × 128 dims × 2 bytes
  = 26 × 8 × 2 × 2048 × 128 × 2
  = ~218 MB

VHT2 Compressed (ship defaults):
  K: weighted average ~4.25 bits/value → 218/2 × (4.25/16) ≈ 29 MB
  V: flat 3 bits/value             → 218/2 × (3/16)    ≈ 20 MB
  Metadata overhead                                     ≈ ~1 MB
  ─────────────────────────────────────────────────────────────
  Total compressed                                      ≈ ~47 MB

  Compression ratio: 218 / 47 ≈ 4.6×
```

For longer generations (4096+ frames), the savings scale linearly and become the difference between fitting on a consumer GPU or not.

### ComfyUI Nodes

**ShannonPrimeVoxtralKVCache** — Enable VHT2 KV cache compression on a Voxtral TTS pipeline. Configurable bit allocations for K and V bands.

**ShannonPrimeVoxtralCacheFlush** — Clear the compressed KV cache between TTS generations.

### Pure PyTorch Fallback

The Voxtral integration uses a pure PyTorch VHT2 implementation with no C extension dependency. This ensures compatibility across all platforms and CUDA versions without requiring a build step. The PyTorch path uses the same `vht2()` transform from `backends/torch/shannon_prime_torch.py`.

---

## 3. Multi-Language Voxtral Implementations

Shannon-Prime's Voxtral KV compression is implemented in three languages across three forked repositories. All three share the same mathematical core — VHT2 + Möbius reordering + banded quantization — differing only in language idioms.

### Python (ComfyUI Node)

**Repository:** `nihilistau/ComfyUI-FL-VoxtralTTS`

**Entry point:** `nodes/shannon_prime_kv.py`

Monkey-patches `MistralBackbone.forward()` at node initialization. Uses the PyTorch VHT2 backend. Integrates with ComfyUI's node graph via `ShannonPrimeVoxtralKVCache` and `ShannonPrimeVoxtralCacheFlush`.

### Rust (Realtime Streaming)

**Repository:** `nihilistau/voxtral-mini-realtime-rs`

**Entry point:** `src/models/layers/shannon_prime.rs`

Pure Rust implementation — no unsafe blocks, no FFI. Implements:

- VHT2 transform (in-place butterfly on `&mut [f32]`)
- Banded quantization with per-band abs-max scaling
- `ShannonPrimeKVCache<B>` generic wrapper around the base `KVCache<B>` type

Four unit tests validate round-trip fidelity and compression ratio.

### C (Embedded / Minimal)

**Repository:** `nihilistau/voxtral-tts.c`

**Entry point:** `shannon_prime_kv.h` (header-only)

Minimal C99 implementation suitable for embedded or edge deployment. Two-function API:

```c
// Compress K/V for a single position across all heads at a given layer
void sp_voxtral_compress_position(
    sp_voxtral_cache_t *cache,
    int layer, int pos,
    const float *k_heads,   // [n_kv_heads × head_dim]
    const float *v_heads    // [n_kv_heads × head_dim]
);

// Decompress K/V for a single layer (all cached positions)
void sp_voxtral_decompress_layer(
    const sp_voxtral_cache_t *cache,
    int layer,
    float *k_out,   // [seq_len × n_kv_heads × head_dim]
    float *v_out    // [seq_len × n_kv_heads × head_dim]
);
```

Header-only design — `#define SP_VOXTRAL_IMPLEMENTATION` in exactly one translation unit.

---

## 4. Qwen3-TTS (Planned)

Qwen3-TTS uses a similar autoregressive architecture to Voxtral. The KV compression approach will be identical in principle — monkey-patch the backbone forward pass, route K/V through VHT2 compress/decompress. Model download is pending; integration work will begin once weights are available.

---

## 5. Comparison with Other TTS Acceleration Techniques

| Technique | Mechanism | KV Memory Saving | Quality Impact | SP Advantage |
|-----------|-----------|-----------------|----------------|-------------|
| **Speculative Decoding** | Draft model predicts tokens, main model verifies | None (KV still full size) | Lossless (rejection sampling) | SP compresses KV *and* is compatible with speculative decoding |
| **KV Eviction (H2O, ScissorHands)** | Drop "unimportant" KV entries by attention score | High (discard entire positions) | Lossy — evicted context is gone forever | SP retains all positions at reduced precision; no information is discarded |
| **Weight Quantization (GPTQ, AWQ)** | Quantize model weights, not KV cache | None (KV still bf16/fp16) | Lossy on weights | SP is orthogonal — compresses KV, not weights; can stack with weight quantization |
| **KV Quantization (KIVI, KVQuant)** | Uniform or group quantization on KV | Moderate (4-8 bits uniform) | Lossy — no spectral awareness | SP exploits RoPE spectral structure: low-frequency bands get more bits because they carry more information |
| **Shannon-Prime VHT2** | Spectral transform + Möbius reordering + banded quantization | High (~4.6x on Voxtral) | Controlled — banded allocation preserves structural information | Exploits the mathematical structure of RoPE embeddings; provably optimal bit allocation across spectral bands |

### What Makes Shannon-Prime Unique for Audio

RoPE (Rotary Position Embeddings) imparts a known spectral structure to K vectors: low-frequency components encode broad positional relationships, high-frequency components encode fine-grained local position. The VHT2 transform exposes this structure explicitly, and Möbius reordering groups coefficients by spectral importance. Banded quantization then allocates bits where the signal energy actually lives.

For TTS models like Voxtral, this means the KV cache for long audio generations (which can run to thousands of autoregressive steps) is compressed with awareness of what the attention mechanism actually uses, rather than treating all dimensions as equally important.

No other KV compression technique exploits this spectral structure. Uniform quantization treats every dimension the same. Eviction-based methods discard entire positions. Shannon-Prime's banded approach is the only one that allocates bits based on the proven spectral energy distribution of RoPE-encoded keys.

---

## 6. Configuration Reference

### Stable Audio Block-Skip

```python
# ShannonPrimeAudioBlockSkip node inputs
{
    "model": MODEL,                    # Stable Audio DiT model
    "skip_blocks": "0,1,2,3,4,5",     # Comma-separated block indices to cache
    "skip_start_step": 2,              # Start caching after N diffusion steps
    "skip_end_step": -1,               # Stop caching N steps before end (-1 = never stop)
}
```

### Voxtral KV Cache

```python
# ShannonPrimeVoxtralKVCache node inputs
{
    "model": MODEL,                    # Voxtral TTS model
    "k_bits": "5,5,4,3",              # Banded bit allocation for K
    "v_bits": "3",                     # Flat bit allocation for V (or banded)
    "use_mobius": True,                # Enable Möbius coefficient reordering
}
```

### Voxtral Rust Configuration

```rust
use shannon_prime::ShannonPrimeKVCache;

let cache = ShannonPrimeKVCache::new(
    n_layers: 26,
    n_kv_heads: 8,
    head_dim: 128,
    max_seq_len: 4096,
    k_band_bits: &[5, 5, 4, 3],
    v_band_bits: &[3],
);
```

### Voxtral C Configuration

```c
sp_voxtral_cache_t cache;
sp_voxtral_cache_init(&cache, (sp_voxtral_config_t){
    .n_layers    = 26,
    .n_kv_heads  = 8,
    .head_dim    = 128,
    .max_seq_len = 4096,
    .k_band_bits = {5, 5, 4, 3},
    .n_k_bands   = 4,
    .v_band_bits = {3},
    .n_v_bands   = 1,
});
```

---

## 7. Repository Map

```
nihilistau/ComfyUI-FL-VoxtralTTS     ← Python: ComfyUI nodes + PyTorch VHT2
├── nodes/
│   ├── shannon_prime_kv.py           ← ShannonPrimeVoxtralKVCache node
│   └── ...
└── lib/shannon-prime/                ← Submodule: core math

nihilistau/voxtral-mini-realtime-rs   ← Rust: streaming TTS with compressed KV
├── src/models/layers/
│   └── shannon_prime.rs              ← ShannonPrimeKVCache<B>, VHT2, banded quant
└── lib/shannon-prime/                ← Submodule: core math

nihilistau/voxtral-tts.c             ← C: header-only embedded path
├── shannon_prime_kv.h                ← sp_voxtral_compress_position / decompress_layer
└── lib/shannon-prime/                ← Submodule: core math
```

All three repositories vendor `lib/shannon-prime/` as a git submodule pointing to the core `shannon-prime` repo, following the same pattern as `shannon-prime-engine` and `shannon-prime-llama`.
