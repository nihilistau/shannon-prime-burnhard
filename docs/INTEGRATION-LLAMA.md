# Shannon-Prime: llama.cpp Integration Guide

## What This Does

Shannon-Prime replaces the standard fp16 KV cache in llama.cpp with a spectrally compressed shadow cache. The inference engine runs normally — the only difference is that KV vectors are compressed on write and reconstructed on read, using 3.4× less memory.

For a 70B model at 128K context, the KV cache alone can exceed 40 GB. Shannon-Prime reduces that to under 12 GB with less than 1.25% perplexity impact.

## Where It Hooks In

```
llama.cpp Inference Pipeline
═══════════════════════════════════════════════════════

  Token Input
       │
       ▼
  ┌─────────────┐
  │  Embedding   │
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────────────────────────────┐
  │            Transformer Layer (×N)            │
  │                                              │
  │  Q = W_q · hidden                            │
  │  K = W_k · hidden                            │
  │  V = W_v · hidden                            │
  │       │                                      │
  │  K = RoPE(K, position)                       │
  │       │                                      │
  │  ┌────▼────────────────────────────────┐     │
  │  │     Shannon-Prime Hook Point        │     │
  │  │                                     │     │
  │  │  WRITE: K,V → shadow_cache          │     │
  │  │  (VHT2 → Möbius → quant → store)    │     │
  │  │                                     │     │
  │  │  READ: shadow_cache → K,V           │     │
  │  │  (load → dequant → unreorder → VHT2)│     │
  │  │  (VHT2 is self-inverse — no 1/N)    │     │
  │  └────┬────────────────────────────────┘     │
  │       │                                      │
  │  Attention = softmax(Q · K^T / √d) · V       │
  │       │                                      │
  │  Output = W_o · attention + residual          │
  │       │                                      │
  │  FFN → next layer                             │
  └──────────────────────────────────────────────┘
         │
         ▼
     LM Head → Token Output
```

The hook point is **after RoPE is applied to K** and **before KV vectors enter the cache**. This is critical — the spectral structure Shannon-Prime exploits is created by RoPE. Hooking before RoPE would see unstructured K vectors with no spectral concentration.

## Integration Methods

### Method 1: Environment Variables (Zero Rebuild)

If using the Shannon-Prime patched llama.cpp build:

```bash
export SHANNON_PRIME_ENABLED=1
export SHANNON_PRIME_K_BITS=5,5,4,3
export SHANNON_PRIME_V_BITS=3
export SHANNON_PRIME_MOBIUS=1
export SHANNON_PRIME_VERBOSE=1

./llama-server -m model.gguf -c 32768
```

### Method 2: Programmatic API

```c
#include "tools/shannon_prime_llama.h"

// During model init — extract params from llama_model
sp_llama_params_t params = {
    .head_dim    = llama_n_embd(model) / llama_n_head(model),
    .n_layers    = llama_n_layer(model),
    .n_heads_kv  = llama_n_head_kv(model),
    .max_seq_len = ctx_params.n_ctx,
    .backend     = SP_BACKEND_CPU,  // or SP_BACKEND_CUDA, SP_BACKEND_VULKAN
};

sp_llama_ctx_t *sp = sp_llama_init(&params);

// In the KV write path (after RoPE):
sp_llama_write_kv(sp, layer, head, pos, k_vec, v_vec);

// In the attention read path:
sp_llama_read_k(sp, layer, head, pos, k_out);
sp_llama_read_v(sp, layer, head, pos, v_out);

// Batch operations for prefill:
sp_llama_write_k_batch(sp, layer, head, start_pos, n_tokens, k_matrix);
sp_llama_read_k_batch(sp, layer, head, start_pos, n_tokens, k_out_matrix);

// Memory reporting:
sp_llama_memory_t mem = sp_llama_memory(sp);
printf("KV cache: %.1f MB (%.1f× compression)\n",
       mem.compressed_bytes / 1048576.0, mem.compression_ratio);

// Cleanup:
sp_llama_free(sp);
```

### Method 3: Compile-Time Integration

Link against the core library and the llama integration layer:

```bash
gcc -O2 -o llama-server \
    llama-server.c \
    shannon-prime/tools/shannon_prime_llama.c \
    shannon-prime/core/shannon_prime.c \
    -lm
```

For CUDA builds, add the CUDA backend:

```bash
nvcc -O2 -o llama-server \
    llama-server.c \
    shannon-prime/backends/cuda/shannon_prime_cuda.cu \
    shannon-prime/core/shannon_prime.c \
    -lm -lcudart
```

## Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SHANNON_PRIME_ENABLED` | 0 | Enable VHT2 shadow cache |
| `SHANNON_PRIME_K_BITS` | 5,5,4,3 | K band bit allocation (4 comma-separated values) |
| `SHANNON_PRIME_V_BITS` | 3 | V bit allocation (flat, single value) |
| `SHANNON_PRIME_MOBIUS` | 1 | Enable Möbius squarefree-first reordering |
| `SHANNON_PRIME_VERBOSE` | 0 | Print configuration and memory stats at init |

### Bit Allocation Presets

| Preset | K Bits | V Bits | Compression | Quality | Use Case |
|--------|--------|--------|-------------|---------|----------|
| Ship default | 5,5,4,3 | 3 | 3.4× | Beats fp16 | Production |
| Conservative | 5,5,5,4 | 4 | 2.8× | <0.1% PPL | Quality-critical |
| Balanced | 5,4,4,3 | 3 | 3.6× | +0.36% PPL | Memory-constrained |
| Aggressive | 4,4,4,3 | 3 | 3.8× | +0.39% PPL | Very memory-constrained |
| Mobile | 4,3,3,3 | 3 | 4.3× | +1.98% PPL | Mobile devices |
| Floor | 3,3,3,3 | 3 | 4.6× | +3.90% PPL | Extreme compression |
| Spec-draft (target) | 5,5,4,3 | 3 | 3.4× | Beats fp16 | Speculative target — same as Ship default |
| Spec-draft (draft, future) | 2,1 | 1 | ~10× | Bounded by acceptance | Speculative draft only — see `SPECULATIVE-DECODING.md` |

The `Spec-draft (draft, future)` row reflects the differential-compression roadmap. Today both target and draft share the same SP compression because env vars are process-wide; the aggressive draft preset becomes selectable per-context once the role-aware init API lands (FUTURE-WORK section 8a).

### Speculative Decoding

llama.cpp's `-md` (draft model) flag works with shannon-prime-llama out of the box — both target and draft are routed through the same SP shadow cache. See [`SPECULATIVE-DECODING.md`](SPECULATIVE-DECODING.md) for recommended draft/target pairs, observed acceptance rates, and the differential-compression roadmap.

**Never go below 3-bit on any band.** 2-bit is catastrophic.

## GQA (Grouped Query Attention) Support

Shannon-Prime is GQA-aware. Set `n_heads_kv` to the number of KV heads, not the total attention heads. For Llama 3.x models with GQA:

- Llama 3.2 1B: n_heads=32, n_heads_kv=8 → use `n_heads_kv=8`
- Llama 3.1 8B: n_heads=32, n_heads_kv=8 → use `n_heads_kv=8`
- Llama 3.1 70B: n_heads=64, n_heads_kv=8 → use `n_heads_kv=8`

This means compression applies to 8 KV heads, not 32/64 attention heads — the memory savings are on the KV cache, which is already efficient from GQA.

## Validation

After integration, verify correctness:

```c
// Spot-check a K vector
float corr = sp_llama_validate_k(sp, k_vec, head_dim);
printf("K correlation: %.4f (expect >0.990)\n", corr);
```

Run the integration test suite:

```bash
cd shannon-prime
make test-integration  # 7 tests covering single, batch, multi-layer, multi-head
```

## Memory Savings Example

Qwen3-8B at 32K context, 8 KV heads, hd=128:

| | fp16 Baseline | Shannon-Prime (5/5/4/3) |
|---|---|---|
| K cache | 2.0 GB | 0.59 GB |
| V cache | 2.0 GB | 0.44 GB |
| **Total KV** | **4.0 GB** | **1.03 GB** |
| **Savings** | — | **2.97 GB (74%)** |
