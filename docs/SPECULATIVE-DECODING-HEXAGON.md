# Speculative Decoding with Hexagon — Phone Draft, PC Target

**Status:** working host plumbing as of the `feat/sp-backend-hexagon` /
`feat/hexagon-scaffold-from-s22u` branches; integration on the LM Studio
side requires the per-model `std::unordered_map<llama_model*, sp_per_model>`
patch shipped in v2.14.0-sp2.

## What this enables

Differential KV-cache compression in a `-md` (speculative-decoding) run:
the **target** runs on the PC's CPU/GPU with normal Shannon-Prime defaults,
the **draft** runs on a connected phone with its KV cache compressed
through the Snapdragon cDSP (Hexagon V69+ via FastRPC + HVX). Each
draft-side write/read is one FastRPC dispatch to `sp_hex_compress_f32` /
`sp_hex_decompress_f32` over rpcmem-backed packed-byte storage; the
DSP and host share the same physical pages via the SMMU, so there's no
marshal copy on the per-position hot path.

## Configuration

```bash
# Target (PC): default SP defaults, CPU shadow cache.
export SHANNON_PRIME_ENABLED=1
export SHANNON_PRIME_BACKEND=cpu

# Draft (phone): aggressive compression, run on Hexagon cDSP.
export SHANNON_PRIME_DRAFT_BACKEND=hexagon
export SHANNON_PRIME_DRAFT_PRESET=aggressive   # K=2,1  V=1

# Run llama-cli with -md as usual.
./llama-cli -md /path/to/draft.gguf -m /path/to/target.gguf ...
```

The role-aware initialiser (`sp_llama_init_with_role(params, ROLE_DRAFT)`)
looks up `SHANNON_PRIME_DRAFT_<NAME>` first and falls back to
`SHANNON_PRIME_<NAME>`, so you can mix and match — e.g. set
`SHANNON_PRIME_K_BITS=5,5,4,3` globally and override only the draft
backend, or override only the draft K bits with the global backend.

## What runs where

```
PC (target)                          Phone (draft)
-----------                          -------------
target llama.cpp ctx                 draft llama.cpp ctx
  └─ sp_llama_ctx (target)             └─ sp_llama_ctx (draft, ROLE_DRAFT)
       └─ cpu_cache (sp_shadow_*)          └─ hexagon_ctx + hexagon_cache
                                                ├─ FastRPC handle (cDSP)
                                                ├─ VTCM 64 KB region
                                                ├─ rpcmem ping-pong scratch
                                                └─ rpcmem packed slots
                                                     [n_layers × n_heads_kv]

                                          cDSP (Hexagon V69 HVX-qf32)
                                            ├─ sp_hex_compress_f32
                                            │    (VHT2 + band_quantize)
                                            └─ sp_hex_decompress_f32
                                                 (band_dequantize + VHT2 inverse)
```

The target context is unaffected — its `sp_llama_init_with_role(...,
ROLE_TARGET)` resolves the global `SHANNON_PRIME_BACKEND`. Only the draft
context picks up the `_DRAFT_BACKEND=hexagon` path.

## Why this composition wins

- **Speculative speedup × compressed draft KV.** Standard speculative
  decoding gives 1.5–3× target tok/sec (well documented in vLLM, TGI,
  llama.cpp). Compressing the draft KV further reduces draft compute on
  the phone — the draft's quality matters less than its acceptance rate,
  so aggressive K=2,1 V=1 compression is acceptable when paired with a
  reasonable draft model size.
- **Phone-as-coprocessor topology.** The draft model is fully resident
  in phone RAM; the cDSP does the KV compress/decompress at HVX speed.
  Per the 2026-04-29 V69 validation, VHT2 forward at head_dim=1024 hits
  1.73× scalar via the qf32 intermediate path, with bit-equivalent-to-
  scalar correctness within fp32 epsilon.
- **No marshal copy on the hot path.** rpcmem-backed packed storage
  with `RPCMEM_HEAP_ID_SYSTEM` + `RPCMEM_TRY_MAP_STATIC` lets every
  per-position write/read go through one FastRPC dispatch with the
  same physical pages visible to both host and cDSP via the SMMU.

## Validation status

Lifecycle, backend selection, per-vector dispatch, and FastRPC fallback
are all syntax-clean under both `-DSP_HAVE_HEXAGON` and the default
desktop build. End-to-end on real silicon requires:

1. Building the scaffold ARM target on Android (cross-compile path
   already wired in `backends/hexagon/scaffold/build.cmd`).
2. Pushing the resulting `libsp_hex_skel.so` and stub libraries to
   `/data/local/tmp/sp22u/` per `hexagon_s22u_working_setup.md`.
3. Running a `-md` pair with the env vars above and confirming the
   draft tok/sec is dominated by FastRPC ping cost, not compute.

The speculative wire-up is "compose three already-working pieces"
rather than new research — speculative decoding (llama.cpp `-md`),
draft-target pairs, and the validated Hexagon backend. Every piece
has its own validation; this doc just records how they're meant to
fit together for the phone deployment milestone (FUTURE-WORK §8e).

## See also

- `docs/SPECULATIVE-DECODING.md` — backend-agnostic speculative-decoding
  setup and recommended draft/target pairs.
- `docs/BACKEND-HEXAGON.md` — the Hexagon backend itself: build flags,
  V69 HVX gotchas, qf32 intermediate pattern, rpcmem zero-copy.
- `hexagon_s22u_2026-04-29_session.md` — the S22 Ultra validation
  writeup with V69 numerics, rpcmem zero-copy, and the disk-tier proof.
