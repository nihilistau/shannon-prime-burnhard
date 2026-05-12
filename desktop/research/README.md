# desktop/research/

Code paths that were active research lines during burnhard's "standalone
inference engine" era and are now **deprecated** in favour of the
LM Studio + shannon-prime-llama (modded llama.dll) path that hits the
26-30 tok/s production target on the RTX 2060 reference setup.

Files in this folder are **kept** rather than deleted because:

1. Each represents real measurement work — the bench numbers, the
   regression diagnoses, the kernels themselves are valuable as
   research baselines.
2. Some of these will resurface on different hardware (CRT becomes
   real with two NVIDIA GPUs; Shredder becomes the reference CPU-only
   bench when porting to a node without CUDA; the Q4_K kernel pattern
   informs any future custom CUDA work).
3. The CMake build still includes them — they compile and link into
   `sp_beast_canyon` exactly as before. Functional behaviour unchanged.

## What's here

| File | Why retired |
|---|---|
| `sp_avx512_shredder.{c,cpp,h}` | CPU-only AVX-512 Q4_K/Q6_K dequant. Redundant on GPU path — ggml-cuda's MMQ does inline GPU dequant in the matmul. Useful as the CPU baseline. |
| `sp_shredder_crt.{c,h}` | Chinese Remainder Theorem dual-ring matmul. Only relevant with two NVIDIA GPUs on this box. Currently theoretical. |
| `sp_shredder_svm.cpp` | Intel Level Zero shared-virtual-memory variant of the Shredder. Same theoretical-with-no-current-hardware status. |
| `sp_beast_gpu.{cpp,h}` | cuBLAS-based gemv host wrapper. Regressed 12× on this hardware tier — Windows WDDM sync floor dominates per-matvec calls. Documented in commit `5f63f35`. |
| `sp_beast_gpu_q4k.cu` | Hand-rolled Q4_K-resident CUDA matvec kernel. Math correct (token IDs match CPU), still regresses on this hardware due to WDDM sync. Kept as foundation for any future Linux/TCC or batched-decode work. Commit `7e6eb1b`, `7897ca8`. |

## Why these aren't in the current chat path

burnhard's chat verb defaults to running through `ForwardContext` +
ggml-cuda (or, with `SP_BEAST_OWN_CHAT=1`, through `sp_beast_generate`).
On Qwen3.6-A3B specifically the ForwardContext path has a separate
correctness issue with hybrid SSM layers (`build_block_gdn`) so the
short-circuit is the only burnhard standalone path that produces
coherent text today — and that path is also where the regressions live
when you flip on the GPU hot-staging from this folder.

The honest engineering story is in
`bench/optane_bench_2026-05-13_summary.md` (Optane vs NVMe disk tier
analysis informing future use) and the burnhard QuickBorn-branch
commit log (lookup commits between `40064a5` "OMP+AVX-512 matvec" and
`7897ca8` "CUDA Graphs no speedup" for the optimization arc).

## Status

Active build target: **YES** — `desktop/beast_canyon/CMakeLists.txt`
still pulls these into `sp_beast_canyon` via `../research/...` source
paths post the 2026-05 reorg. No behavioural change vs. pre-reorg.

Future direction: when the QuickBorn SP-Flash diffusion-draft work
solidifies in `Shannon-Prime-QuickBorn`, the most valuable bits here
(the Q4_K kernel pattern + the AVX-512 dequant) may get ported into
that or into a future shannon-prime-llama patch. Until then, frozen.
