# Models, Hardware, and the Road Ahead

What Shannon-Prime supports today, what hardware it targets, and what this means for the future of local inference.

---

## 1. Supported Model Architectures

Shannon-Prime works with any transformer model that uses rotary position embeddings (RoPE) and stores KV cache as dense vectors. This covers the vast majority of modern LLMs and generative models.

### Validated Architectures

| Architecture | Models | head_dim | Status | Notes |
|---|---|---|---|---|
| **Qwen3** | Qwen3-8B, Qwen2.5-Coder-3B/0.5B | 128 | Production | Primary validation target |
| **Qwen3-MoE** | Qwen3.6-35B-A3B | 128 | Production | 26.92 tok/sec in LM Studio |
| **Llama** | Llama 3.2 1B/3B | 64, 128 | Production | PrimePE validated across quant levels |
| **Dolphin** | Dolphin 1B | 64 | Production | Used in scaling law derivation |
| **Mistral** | Mistral 7B | 128 | Provisional | Default config, needs calibration |
| **Gemma** | Gemma 2B/7B | 256 | Provisional | hd=256 → pad=330 for sqfree |
| **Command-R** | Command-R 35B | 128 | Provisional | |
| **DeepSeek** | DeepSeek-V3/V4 MoE | 128 | Provisional | V4 independently validates our architecture |
| **Phi** | Phi-3 Mini/Small | 96 | Calibrated | +2.44% PPL, within quality budget |

### Head Dimension Support

| head_dim | VHT2 Stages | Sqfree Pad | Sqfree Factorization | Status |
|---|---|---|---|---|
| 64 | 6 (2^6) | 66 | 2·3·11 | Full support |
| 96 | N/A (not power-of-2) | 110 | 2·5·11 | Sqfree path only |
| 128 | 7 (2^7) | 154 | 2·7·11 | Full support (primary target) |
| 256 | 8 (2^8) | 330 | 2·3·5·11 | Full support |

**GQA-aware:** Shannon-Prime compresses per KV head, not per attention head. Models with grouped-query attention (GQA) where n_head_kv < n_head benefit fully — fewer KV heads means less cache to compress per token.

### MoE (Mixture of Experts)

Shannon-Prime compresses the KV cache per-expert. In MoE models, the KV cache is shared across all tokens (not per-expert), so the compression ratio is the same as for dense models. However, Beast Canyon adds per-expert weight staging:

- Expert weights live on Optane (zero-copy via DAX mmap)
- The shredder dequantizes selected experts into LLC on demand
- Next-expert prefetch overlaps with current-expert GPU compute
- Per-expert routing is round-robin across available GPUs

This means a 35B-A3B MoE model (8 experts, top-2) can run on hardware that couldn't hold all expert weights in GPU VRAM — Optane serves as the weight reservoir, and the AVX-512 shredder streams experts to GPUs as needed.

### Quantization Compatibility

Shannon-Prime's KV compression operates independently of weight quantization. It works with any GGUF quantization format:

| Weight Quant | SP KV Compression | Quality Impact | Notes |
|---|---|---|---|
| Q8_0 | Ship (3.4×) | < 1.25% PPL | Best quality; spinor sheet bit most effective here |
| Q6_K | Ship (3.4×) | < 1.5% PPL | Sweet spot for desktop |
| Q5_K_M | Ship (3.4×) | ~ 2% PPL | Good for memory-constrained |
| Q4_K_M | Ship (3.4×) | ~ 3% PPL | Weight quant noise starts to dominate |
| Q3_K | Ship (3.4×) | ~ 5% PPL | Approaching quality floor; spinor not recommended |
| IQ2 | Ship (3.4×) | Variable | Works for spec-decode draft models; not for correctness |

The scaling law quantifies this: `bits^1.5` in the denominator means Q4 amplifies K error ~2.8× vs Q8. At Q3 and below, weight quantization errors dominate and the marginal benefit of SP compression is smaller (though still positive).

---

## 2. Hardware Requirements

### Minimum (CPU-Only, Any Model)

- x86-64 or ARM64 CPU
- Enough RAM to hold the model weights + KV cache (SP reduces the KV portion by 3.4×+)
- No special instructions required (AVX-512, NEON, etc. improve throughput but aren't mandatory)

### Recommended Desktop

| Component | Specification | Purpose |
|---|---|---|
| CPU | Intel 12th Gen+ (AVX-512) or AMD Zen 4 | VHT2 transform, shredder |
| GPU | NVIDIA RTX 3060+ or AMD RX 6800+ | GPU-resident cache, matmul |
| RAM | 32 GB+ | Model weights + working memory |
| Storage | NVMe SSD (500 GB+) | Disk-tier band storage |
| Optional | Intel Optane M10/M15 (16-64 GB) | Zero-copy weight reservoir (Beast Canyon) |

### Beast Canyon (Heterogeneous Desktop)

| Component | Specification | Purpose |
|---|---|---|
| CPU | Intel i9-12900 (16C/24T, 24MB LLC, AVX-512) | Shredder, orchestration |
| GPU 0 | NVIDIA RTX 2060+ (CUDA) | Expert matmul |
| GPU 1 | Intel Xe iGPU or second discrete (Vulkan) | Expert matmul (round-robin) |
| Optane | Intel Optane M10 16GB / M15 32-64GB | DAX-mapped model reservoir |
| Optional | Samsung Galaxy S22 Ultra via USB-C | Sidecar DSP (PrimePE, draft inference) |

Beast Canyon degrades gracefully: dual-GPU → single GPU → CPU-only, Optane → filesystem mmap, sidecar → local CPU.

### Phone (Snapdragon 8 Gen 1+)

| Component | Use |
|---|---|
| Cortex-A78 (big cores) | Inference, VHT2 transform |
| Cortex-A510 (little cores) | Background compression |
| Adreno 730 GPU | Attention matmul via Vulkan/OpenCL |
| Hexagon V69 DSP | VHT2 + banded quant (dedicated thermal envelope) |
| HTP (Tensor Processor) | QNN .bin execution for prefill |
| UFS 3.1 Flash | Disk-tier band storage (~2 GB/s) |

**Primary validation device:** Samsung Galaxy S22 Ultra (Snapdragon 8 Gen 1).

---

## 3. What This Means for Models Going Forward

### The KV Cache Problem Is Solved

The KV cache has been the memory bottleneck for long-context inference since transformers were introduced. For a 70B model at 32K context, fp16 KV cache alone requires ~40 GB. This is why consumer hardware couldn't do long-context inference on large models.

Shannon-Prime eliminates this bottleneck:

| Model | Context | fp16 KV Cache | SP Ship (3.4×) | SP Hier (7×) |
|---|---|---|---|---|
| 7B (hd=128, 32 heads) | 8K | 4 GB | 1.2 GB | 0.6 GB |
| 7B | 32K | 16 GB | 4.7 GB | 2.3 GB |
| 35B MoE (hd=128, 8 KV heads) | 32K | 4 GB | 1.2 GB | 0.6 GB |
| 70B (hd=128, 8 KV heads GQA) | 32K | 5 GB | 1.5 GB | 0.7 GB |
| 70B | 100K | 15 GB | 4.4 GB | 2.1 GB |

A 70B model at 100K context goes from "impossible on consumer hardware" (15 GB KV alone, before model weights) to "fits in 2 GB" with hierarchical compression.

### MoE Models Are Now Practical Everywhere

MoE models like Qwen3.6-35B-A3B define many experts but activate only a few per token. The problem has always been that all expert weights need to be accessible, even though most aren't used at any given time. Beast Canyon solves this:

1. All expert weights live on Optane (16-64 GB, byte-addressable)
2. The shredder dequantizes only the selected experts into LLC
3. GPU VRAM holds only the active expert's fp16 weights + the compressed KV cache
4. Total GPU VRAM requirement: expert_size × top_k + compressed_kv

For a 35B-A3B model with 8 experts (top-2 activation): the model needs ~70 GB total weight storage (all experts), but only ~17 GB of active weight data per token (2 experts). With Optane as the reservoir, a 24 GB GPU can run this model at 26+ tok/sec.

### Hardware Is Becoming the Programmable Part

Shannon-Prime doesn't pick one compute target. It runs the same math on every compute unit your hardware has. The engine's job is to route work to the right unit:

- **CPU:** VHT2 transform, Möbius reorder, calibration, control flow
- **GPU:** Attention matmul, expert matmul, GPU-resident cache ops
- **DSP:** Compression (phone), parallel VHT2 (dedicated thermal envelope)
- **NPU:** Pre-compiled inference graphs (QNN .bin execution)
- **Optane:** Weight reservoir, disk-tier band storage

Future hardware (dedicated KV-cache compression units, sparsity-aware matmul engines, CXL-attached memory pools) will slot into this architecture as new backends. The math doesn't change.

### Speculative Decoding Changes the Equation

Shannon-Prime + speculative decoding is a multiplicative win:

1. SP compresses the KV cache → both draft and target models fit in less memory
2. Compressed KV is cheaper to read → per-token cost drops
3. Draft model runs at aggressive compression (errors are recoverable)
4. The SP-fused cost curve crosses vanilla at n_q ≈ 3-4

Net result: 3.58× on phone CPU (43.72 t/s), with projections to 4-10× at optimal draft depth.

### Convergence with Industry

DeepSeek-V4 (1.6T FP8 MoE, April 2025) independently validates the Shannon-Prime architectural choices: KV-cache compression + sliding-window attention + prefetch-oracle are the core ideas behind their serving efficiency. Shannon-Prime got there first, from first principles, on a phone.

---

## 4. Scaling Projections

The K-corr scaling law predicts how Shannon-Prime compression quality scales with model size:

```
log(PPL/base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)
```

At constant K_corr (same compression config), **bigger models tolerate compression better.** The sub-linear params exponent (1.1) means a 70B model absorbs the same absolute K-error with ~8× less relative PPL degradation compared to a 1B model.

| Model Size | Ship (K_corr ~0.99) | Predicted PPL Impact |
|---|---|---|
| 1B | 3.4× | ~ 2.5% |
| 7B | 3.4× | ~ 1.2% |
| 35B | 3.4× | ~ 0.6% |
| 70B | 3.4× | ~ 0.3% |

At 70B, ship compression adds virtually nothing to the PPL. The model's internal redundancy (head-averaging) absorbs the reconstruction error completely.

This means Shannon-Prime compression becomes **more valuable** as models get larger — exactly the regime where memory is the binding constraint.
