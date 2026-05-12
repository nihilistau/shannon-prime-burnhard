# desktop/policy/

Hardware-aware orchestration policy — the rules that decide "20 layers
to RTX 2060, top-K=4 with 2/2 split, CRT dispatch armed when both GPUs
are alive". Promoted from `beast_canyon/sp_layer_split_policy.cpp` in
the 2026-05 reorg.

## What lives here

| File | Purpose |
|---|---|
| `sp_layer_split_policy.cpp` | The hardcoded Qwen3.6-35B-A3B / RTX 2060 12 GB / i9-11900KB + UHD 750 split policy. Sets `gpu_0_layers=20`, `host_layers=20`, `enable_crt_dispatch=true`, `expert_split_mode=VARIABLE`. Also flips `sidecar_mode` based on S22U heartbeat presence. |

## Status

`sp_layer_split_policy.cpp` is **orphaned** — it was never wired into
the current build (`SP_BEAST_SOURCES` in beast_canyon's CMakeLists
doesn't list it) and `#include "sp_moe_curriculum.h"` references a
header that doesn't exist in the tree. It compiled at some past point.
Kept verbatim because the policy values it bakes in are the
correct-for-this-box defaults documented elsewhere
(`CLAUDE.md` lines about the 20/20 split + top-K=4).

## Where this is heading

The re-imaging proposal positions this folder as the host of a future
**recommender service** the lmstudio-server MCP can call to pre-compute
optimal LM Studio load configs (`num_experts`, `n_gpu_layers`,
`context_length`, `eval_batch_size`, `flash_attention`,
`offload_kv_cache_to_gpu`) given the current free VRAM + RAM at request
time. That would become a new MCP tool:
`inference_recommend_load_config(model_id) → { ...config... }`.

The 20-layer-on-RTX + top-K=4 numbers in `sp_layer_split_policy.cpp`
are exactly what the user's existing LM Studio load already shows
(`lmstudio_v1_active_config` reports `num_experts: 4`,
`flash_attention: true`, `parallel: 2`). Not currently driving
anything — informational baseline.

## Why not delete it

Three reasons:
1. The policy values are the right defaults; we may need them when we
   move from "static config baked into LM Studio's UI" to "dynamic
   recommender that returns configs over the MCP".
2. It documents intent (the comments name the hardware targets).
3. The `extern "C"` symbol it exports
   (`sp_beast_canyon_apply_split_policy`) is the right shape for the
   future recommender to keep.
