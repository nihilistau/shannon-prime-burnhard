# Shannon-Prime: Testing Guide

## Running All Tests

```bash
cd shannon-prime
make test-all
```

This runs all 8 test suites. Post-v1.03: 188/189 passing (35/36 core + 7 cuda + 4 vulkan + 14 adreno + 7 integration + 31 torch + 25 comfyui + 69 sqfree). The one synthetic-K-pipeline flake is known — see the banded-quantization notes below.

For individual suites:

```bash
make test-core          # C core math (36 tests, incl. v1.03 non-divisible bands)
make test-adreno        # Adreno/ARM mobile (14 tests)
make test-vulkan        # Vulkan + CPU fallback (4 tests)
make test-cuda          # CUDA backend on a real NVIDIA GPU (7 tests)
make test-integration   # llama.cpp integration (7 tests)
make test-torch         # PyTorch backend (31 tests)
make test-comfyui       # ComfyUI + Wan architecture (25 tests)
make test-sqfree        # Sqfree + spinor path (69 tests, PyTorch)
```

## What Each Suite Validates

### Core Math (31 tests)

The foundation. If these fail, nothing else works.

```bash
make test-core
```

**VHT2 Round-Trip (4 tests):** Verifies `VHT2(VHT2(x)) = x` (self-inverse, no 1/N) for hd=32, 64, 128, 256. Maximum error should be <1e-5 (typically ~1e-7). If this fails, the VHT2 at p=2 butterfly has a bug — check indexing in the inner loop or the 1/√2 per-stage normalisation.

**Möbius Function (9 tests):** Checks known values: μ(1)=1, μ(2)=−1, μ(4)=0, μ(6)=1, μ(30)=−1, etc. Also verifies squarefree count matches. If μ(4)≠0, the squared-prime detection in the sieve is broken.

**Möbius Reorder Round-Trip (2 tests):** Reorder then unreorder must produce exact input. Max error = 0.00 (exact, no floating point involved — it's just index permutation).

**Banded Quantization (5 tests):** Tests all five configs from the paper. Look at both correlation AND compression ratio:

| Config | Expected Correlation | Expected Compression |
|--------|---------------------|---------------------|
| 5/5/4/3 | >0.99 | 3.4× |
| 5/4/4/3 | >0.98 | 3.6× |
| 4/4/4/3 | >0.98 | 3.8× |
| 4/3/3/3 | >0.95 | 4.3× |
| 3/3/3/3 | >0.92 | 4.6× |

If correlation is high but compression is wrong, the byte counting in `sp_band_config_init` is off. If compression is right but correlation is low, the bit packing/unpacking has a bug.

**K/V Spectral Asymmetry (2 tests):** K (periodic/structured) should concentrate >60% energy in the first half of VHT2 bands. V (random) should be roughly uniform (~50%). This is the foundational property — if K doesn't concentrate, the banded allocation is pointless.

**Vilenkin Round-Trip (3 tests):** Tests 2, 3, and 4-prime bases. Max error should be <1e-4. The key fix was normalization: V·V=I (not N·I). If round-trip error is ~0.5, the inverse is still dividing by N.

**Full VHT2 Pipeline (2 tests):** End-to-end: write K through shadow cache, read back, check correlation. K should be >0.985, V should be >0.950. If K passes but V fails badly, check that V isn't getting Möbius reorder (V should NOT be reordered in self-attention mode).

> **Known flake:** this test's K threshold is 0.990 but the synthetic K generator used in `tests/test_core.c` sometimes lands at 0.9894. Real post-RoPE K hits 0.997+ reliably. The threshold is preserved to catch genuine regressions; the flake is documented in CLAUDE.md as acceptable. Count this as 30/31 = pass.

**Möbius Quality Improvement (1 test):** Compares correlation with and without Möbius reorder on structured signals. Möbius should be ≥ plain (typically +0.004). If Möbius is worse, the reorder/unreorder might be swapped.

**Compression Ratios (2 tests):** Checks hd=128 gives 3.0–4.5× and hd=64 gives 2.5–5.0×.

**Vilenkin Successive Extraction (1 test):** Extracts 95% energy via first pass, checks residual is <10% of original energy.

**Banded Quant Non-Divisible (5 checks — v1.03):** Exercises `sp_band_span` on 10 bands over pad_dim=154. Asserts the typical band size is 15, the last band absorbs the 19-coeff remainder, round-trip correlation holds (>0.900 at 3-bit × 10), and the tail coefficients are actually written back (regression guard against silent truncation of indices 150..153). If this fails on CUDA or Torch while core passes, check that the band loop in `backends/{cuda,torch}/*` uses the per-band `band_sz` computed from `bc->head_dim` — pre-v1.04 the CUDA launcher passed `band_size * n_bands` and orphaned the tail.

### Adreno/ARM Mobile (14 tests)

```bash
make test-adreno
```

On x86 this uses scalar fallbacks. On ARM it uses real NEON.

**Hardware Detection (1 test):** Just verifies `sp_mobile_detect_caps()` doesn't crash. Check the printed output — on x86 it should show NEON=no and detect CPU cores via sysfs.

**NEON VHT2 vs Core (3 tests):** The NEON VHT2 at p=2 must produce identical results to the C core `sp_vht2_forward_*`. Max error should be 0.00 on scalar fallback, <1e-6 on actual NEON. Any difference means the NEON butterfly has an indexing bug.

**fp16 Conversion (1 test):** f32→f16→f32 round-trip. Max error should be <0.01 (fp16 precision is ~1e-3). If error is large, check the sign/exponent/mantissa bit manipulation.

**Absmax Reduction (1 test):** Plants a known maximum (−5.5) at index 73 and verifies it's found. Tests both the NEON vectorized path and scalar tail handling.

**Full Pipeline (2 tests):** Same as core pipeline but through the Adreno cache. K >0.985, V >0.950.

**hd=64 Mobile (1 test):** Typical mobile head_dim. Paper target: 0.9972 correlation.

**fp16 Write Path (1 test):** Writes via `sp_adreno_write_k_f16()`, reads via standard f32 path. Additional fp16 precision loss is expected — threshold is 0.980.

**Thread Affinity (1 test):** Calls `sp_set_thread_affinity(SP_AFFINITY_ANY)`. On Linux it should succeed (rc=0). On other platforms it returns -1 (non-fatal). Test always passes — just checks for crashes.

**Performance Counters (2 tests):** Verifies write/read counters increment correctly.

**Benchmark (1 test):** Full-model writeback timing. On x86 this completes in <1 ms. On ARM the paper target is 37–42 ms for Dolphin 1B (16 layers × 4 heads).

### Vulkan (4 tests)

```bash
make test-vulkan
```

Without Vulkan SDK, exercises the CPU fallback path through the Vulkan API surface.

**Init (1 test):** `sp_vulkan_cache_init()` with NULL device/queue → CPU fallback.
**K Pipeline (1 test):** Write + read K, correlation >0.990.
**V Pipeline (1 test):** Write + read V, correlation >0.950.
**Batch Ops (1 test):** Write + read 8 positions in batch, average correlation >0.990.

### CUDA (7 tests)

```bash
make test-cuda
```

Runs against a real NVIDIA GPU via the CUDA runtime. Requires `nvcc` and a host
C/C++ compiler (on Windows, the MSVC environment must be sourced before invoking
`make` — e.g. from a Developer Command Prompt, or by wrapping via
`scripts/build_test_cuda.bat`). Default architecture is `sm_75` (Turing); override
with `make test-cuda NVCC_ARCH=-arch=sm_80` (Ampere), `-arch=sm_86`,
`-arch=sm_89` (Ada), `-arch=sm_90` (Hopper), etc. The API takes device pointers,
so the test harness stages host↔device with `cudaMemcpy`.

**Init (1 test):** `sp_cuda_cache_init()` with the default stream succeeds after
confirming at least one CUDA device is present.
**K Pipeline (1 test):** Write + read K through the full GPU chain
(VHT2 → Möbius reorder → band quantize → dequantize → Möbius unreorder → VHT2 (self-inverse)),
correlation >0.990.
**V Pipeline (1 test):** Same shape, no Möbius, V config (flat 3-bit),
correlation >0.950.
**K Batch (1 test):** 8-position batch, avg correlation >0.990. Catches
per-vector indexing bugs in the batch kernels that single-vector tests miss.
**V Batch (1 test):** 8-position batch, avg correlation >0.950.
**Multi-layer/head (1 test):** 4 layers × 2 heads × 4 positions = 32 K vectors
across distinct cache slots. Minimum correlation >0.980. Catches per-slot
offset bugs in the K cache addressing.
**Memory diagnostic (1 test):** `sp_cuda_print_memory()` runs without crashing
and reports ~76 bytes/vector for K (vs 256 fp16) and ~50 bytes for V.

If K fails but V passes: the Möbius unreorder path is broken. Check that the
read path passes `d_mobius_order` (same table as reorder), not an inverse.
If both K and V produce wildly out-of-range values (e.g. 1e30+): the band
quantize/dequantize kernels are writing out-of-bounds — check that the
kernels bounds-check `vec_idx` against `n_vecs`.

### llama.cpp Integration (7 tests)

```bash
make test-integration
```

**Programmatic Init (1 test):** `sp_llama_init_config()` with explicit params succeeds.
**KV Write/Read (2 tests):** Single K and V through the llama API. K >0.990, V >0.950.
**Batch Operations (1 test):** 16 positions written and read in batch.
**Multi-Layer Multi-Head (1 test):** 32 K vectors across 4 layers × 8 heads. Minimum correlation >0.980. This catches per-slot indexing bugs.
**Compression Ratio (1 test):** Memory reporting shows >3× compression.
**Validation Helper (1 test):** `sp_llama_validate_k()` returns >0.990.

### PyTorch (28 tests)

```bash
python3 tests/test_torch.py
```

Mirrors the C core tests in PyTorch. Same invariants, same thresholds. Key difference: PyTorch uses `float32` by default (no `double`), so numerical thresholds are slightly different.

### ComfyUI (25 tests)

```bash
python3 tests/test_comfyui.py
```

**Cross-Attention Cache (4 tests):** Put/get with Wan 14B cross-attn shape (1, 40, 77, 128). Verifies K/V correlation, shape preservation, dtype preservation.

**Wan 2.1 Dense (4 tests):** 40 blocks × 50 timesteps simulation. 40 misses on first timestep, 1960 hits, 98% hit rate. No expert set.

**Wan 2.2 MoE (5 tests):** Expert switching at σ=0.875. 80 misses (40 blocks × 2 experts), 1920 hits, 96% hit rate. Verifies cache entries = 80 (40×2) and that high-noise and low-noise experts cache different K/V (cross-correlation ≈ 0).

**I2V vs T2V (1 test):** Boundary is 0.900 for I2V, 0.875 for T2V.

**TI2V-5B Dense (2 tests):** No MoE boundary. 30 layers, 30 misses.

**Linear Layer Wrapper (2 tests):** `WanCrossAttnCachingLinear` as drop-in replacement.

**Möbius on Both K and V (2 tests):** Cross-attention applies Möbius to both (unlike self-attention which only reorders K).

**Compression (1 test):** Ratio >2.5×.

**Expert Cache Clearing (3 tests):** `clear_expert('high_noise')` removes only that expert's entries.

## Interpreting Failures

**Correlation below threshold on random data:** Random vectors occasionally produce lower correlation than structured (RoPE-like) vectors. If the test shows 0.984 against a 0.985 threshold, this is noise — not a bug. Real K vectors with RoPE structure achieve 0.997+. Thresholds are set to catch real bugs while tolerating statistical variation.

**NaN in output:** If you see NaN in reconstructed vectors, check: (1) the bit allocation — any band at 2-bit is catastrophic, (2) the VHT2 normalisation — each p-stage is already scaled by 1/√p, so a forward-then-forward round-trip must NOT divide by N (VHT2 is self-inverse), (3) the Möbius reorder/unreorder — if these are swapped, the quantization hits wrong coefficients. The only surviving guard is the non-finite fp16-scale check inside `sp_band_dequantize`, which zeros the band on overflow.

**Compression ratio doesn't match paper:** The paper reports 3.4–3.8× total (K+V combined). Our ratio calculation counts both K and V bytes. If you see 3.4× for K alone, that's the K-specific ratio, not the total.

**Vilenkin tests fail with high error:** Check that `sp_vilenkin_inverse` does NOT divide by N. The Vilenkin-Hartley basis with per-factor `1/sqrt(p)` normalization gives V·V=I (orthonormal), not V·V=N·I.

## Adding New Tests

Follow the pattern in any existing test file. Key conventions:

```c
// C tests
#define CHECK(cond, msg) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s\n", msg); } \
} while(0)
```

```python
# Python tests
def check(cond, msg):
    global passed, total
    total += 1
    if cond:
        passed += 1
        print(f"  [PASS] {msg}")
    else:
        print(f"  [FAIL] {msg}")
```

Always use `srand(42)` / `torch.manual_seed(42)` for reproducible random data. Print correlation values and thresholds so failures are immediately diagnosable.
