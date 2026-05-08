# Mathematical Foundations

The theoretical framework behind Shannon-Prime. This document covers the mathematics in sufficient detail to understand why the system works, what guarantees it provides, and where the empirical laws come from. For implementation details, see [COMPRESSION-FEATURES.md](COMPRESSION-FEATURES.md). For the full proofs and experimental validation, see the papers.

---

## 1. The Observation: RoPE Creates Spectral Structure

Rotary position embeddings encode sequence position m into key vector k by applying a block-diagonal rotation:

```
RoPE(k, m) = [k_0 cos(mθ_0) − k_1 sin(mθ_0),
              k_0 sin(mθ_0) + k_1 cos(mθ_0),
              k_2 cos(mθ_1) − k_3 sin(mθ_1),
              ...]
```

where θ_i = base^(−2i/d) are geometric frequencies. This rotation imprints a multi-frequency sinusoidal signature on every K vector. The signature is determined by the position m and the frequency schedule θ, not by the content of the key.

The key insight is that this signature creates **predictable spectral structure**. When K vectors are transformed into a frequency-domain basis, the energy concentrates in specific coefficients that correspond to the RoPE frequency pairs. The high-energy coefficients are at frequencies that align with the RoPE schedule; the low-energy coefficients are at frequencies that don't.

---

## 2. The Transform: Vilenkin-Hartley (VHT2)

### 2.1 Definition

The Vilenkin-Hartley Transform over the group Z/p₁Z × Z/p₂Z × ... × Z/pₖZ is the Kronecker product of individual Hartley stages:

```
V = H_{p₁} ⊗ H_{p₂} ⊗ ... ⊗ H_{pₖ}
```

where each p-point Hartley matrix has entries:

```
H_p[i,j] = cas(2πij/p) / √p     (cas(x) = cos(x) + sin(x))
```

### 2.2 Self-Inverse Property

The Hartley matrix satisfies H_p · H_p = I for all primes p (with the 1/√p normalization). Since the Kronecker product of self-inverse matrices is self-inverse:

```
V · V = (H_{p₁} · H_{p₁}) ⊗ (H_{p₂} · H_{p₂}) ⊗ ... = I ⊗ I ⊗ ... = I
```

This means VHT2(VHT2(x)) = x exactly. The same function compresses and decompresses.

### 2.3 Power-of-2 Specialization

When the dimension is n = 2^k, all prime factors are p=2, and VHT2 reduces to the Walsh-Hadamard butterfly with Hartley normalization:

```
For each of k stages:
    a' = (a + b) / √2
    b' = (a − b) / √2
```

This is the ship path — O(n log n) operations, in-place, no additional memory.

### 2.4 Why Not DFT/DCT?

The DFT is not self-inverse (it requires a separate IDFT with conjugation). The DCT family includes self-inverse variants (DCT-I), but they don't decompose naturally over the multiplicative lattice of the integers. The Hartley transform is self-inverse, real-valued, and its multi-prime generalization (Vilenkin) exposes the Kronecker product hierarchy that enables the hierarchical predictor.

---

## 3. The Multiplicative Lattice

### 3.1 Squarefree Numbers

An integer n is squarefree if no prime squared divides it: μ(n) ≠ 0 where μ is the Möbius function.

```
μ(n) = { (-1)^k  if n is a product of k distinct primes
        { 0       if n has a squared prime factor
```

At N=210 (= 2·3·5·7, the dimension relevant for hd=128 padded to 154 → embedded in 210):
- 61.4% of indices 1..210 are squarefree
- These carry the majority of VHT2 spectral energy for RoPE'd K vectors

### 3.2 Why Squarefree Indices Matter

The multiplicative structure of the integers creates a natural partition:

- **Squarefree indices** correspond to "fundamental frequencies" in the VHT2 spectrum. They are the irreducible building blocks from which all other coefficients can be predicted.
- **Non-squarefree indices** correspond to "harmonic frequencies" that are linear combinations of the fundamental ones (modulo the Möbius inversion formula).

This is not an analogy — it's the literal structure of the Möbius inversion:

```
f(n) = Σ_{d|n} μ(d) · g(n/d)
```

If g is the VHT2 spectrum and the squarefree coefficients (where μ(d) ≠ 0) are known, the non-squarefree coefficients can be predicted via the divisor sum.

### 3.3 The Möbius Predictor

For a residual coefficient at index r+1, the prediction from skeleton coefficients is:

```
pred[r] = Σ_{d|(r+1), μ(d)≠0} μ(d) · skel_vals[slot((r+1)/d)]
```

This is stored as a Compressed Sparse Row (CSR) matrix for O(1) per-residual prediction.

**Basis dependence:** On the power-of-2 WHT basis, the Möbius predictor has r ≈ 0 (it functions as a partition heuristic, not a predictor). On the sqfree-padded Vilenkin basis, it has r = 0.40–0.58 (a genuine predictor). This basis-dependence is the reason the sqfree path exists: it unlocks the predictor.

---

## 4. The Knight Skeleton

The Knight mask partitions pad_dim indices into:

- **Skeleton:** The top-K squarefree indices by empirical variance. These are stored directly (banded quantization).
- **Residual:** Everything else. Predicted from the skeleton via the Möbius CSR predictor, then the prediction error is quantized at N bits.

The default skeleton size is L/2 (half the padded dimension). With variance-ranked selection (after calibration), the skeleton captures the highest-information coefficients.

The name "Knight" comes from the chess analogy: the skeleton coefficients are at positions that have an indirect multiplicative relationship to the residual positions, connected through the Möbius divisor lattice.

---

## 5. The Spinor Sheet Bit

### 5.1 Background: SU(2) Double Cover

The rotation group SO(3) has a double cover: SU(2). Every rotation in 3D corresponds to two elements of SU(2) — they differ by a sign (a "sheet" of the covering space).

RoPE applies rotations in 2D subspaces of the head_dim-dimensional space. The VHT2 spectrum has an analogous phase ambiguity: the Möbius predictor can estimate the magnitude of a residual coefficient but not always its sign. The sign depends on which "sheet" of the spectral double cover the original value occupies.

### 5.2 The Correction

The spinor bit records this sheet assignment: 1 bit per residual position. On dequantization, the bit is used to resolve the sign ambiguity.

### 5.3 Impact

On Qwen3-8B Q8 hd=128:
- Without spinor: K+μ+3bit at 3/3/3/3/3 gives PPL 7.35 @ 3.1×
- With spinor: K+μ+3bit+spinor at 3/3/3/3/3 gives PPL 7.32 @ 3.3×

The spinor bit achieves MOBIUS-default quality (PPL 7.31) at +27% more compression. It is the first feature that shifts the Pareto frontier rather than moving along it.

### 5.4 Precision Sensitivity

The spinor bit is most effective on Q8+ backbones. On Q3, the same K-corr improvement costs ~7× more PPL (from the scaling law's bits^1.5 exponent). At Q3 precision, weight quantization noise dominates and the 1-bit correction is washed out.

---

## 6. The Hierarchical Vilenkin Predictor

### 6.1 Kronecker Sub-Projection

The Vilenkin basis V = H_{p₁} ⊗ H_{p₂} ⊗ ... ⊗ H_{pₖ} has natural sub-projections: restrict to the first j < k prime factors.

For pad_dim=154 = 2·7·11:
- Level 1: H_2 → 2 coefficients (1.3% of 154)
- Level 2: H_2 ⊗ H_7 → 14 coefficients (9.1%)
- Level 3: H_2 ⊗ H_7 ⊗ H_11 → 154 coefficients (100%)

The level-2 sub-projection is a natural "core skeleton" — it captures the coarsest spectral features.

### 6.2 The Linear Predictor

Given the 14 skeleton coefficients, the remaining 140 target coefficients are predicted by a calibrated linear map:

```
target_predicted = W · skeleton
```

W is a 140 × 14 matrix (fp16), calibrated per-(layer, head) slot via ridge regression:

```
W = (X^T X + λI)^{-1} X^T Y
```

where X is the matrix of skeleton vectors from calibration, Y is the corresponding target vectors, and λ is the ridge regularization parameter.

### 6.3 Storage Analysis

Per position: 14 skeleton coefficients at 5-bit banded + 140 residual coefficients at 2-bit = 350 bits = 43.75 bytes.

From 308 bytes fp16: **7.0× compression.**

Aggressive: 14 × 4 bits + 140 × 1 bit = 196 bits = 24.5 bytes = **12.6× compression.**

Predictor overhead: 14 × 140 × 2 bytes per (layer, head) slot. For a 32-layer model with 8 KV heads: 32 × 8 × 3920 = ~1 MB. Amortized across all sequence positions.

### 6.4 Sticky-EMA Online Adaptation

`sp_hier_calibrate_end_blend(hp, W_prev, keep_frac)` blends the fresh ridge-regression solution with a previous W using exponential moving average. This enables online adaptation without catastrophic forgetting of prior calibration data.

---

## 7. The K-Corr Scaling Law

### 7.1 The Law

```
log(PPL / PPL_base) ≈ 4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)
```

### 7.2 Derivation Sketch

Attention computes `softmax(Q · K^T / √d) · V`. The KV compression error ε_K enters through the dot product Q · (K + ε_K)^T = Q · K^T + Q · ε_K^T.

- The error term Q · ε_K^T is bilinear in K-error, hence **quadratic** in (1 − K_corr) (since K_corr is the correlation between K and K_reconstructed).
- Larger models average over more attention heads, reducing the per-head error contribution by a factor related to params. The averaging is not perfect (heads are not independent), giving a **sub-linear** exponent of 1.1.
- Lower weight precision (fewer bits) means the model's internal representations are already noisy. KV compression error compounds with weight quantization error. The compounding is **super-linear** in bits with exponent 1.5.

### 7.3 Validation

The law fits 9 configurations spanning:
- Models: Dolphin 1B Q8, Qwen3-8B Q8, Qwen3-8B Q3
- Compression: Ship, sqfree, sqfree+spinor, various band allocations
- PPL ratios: from 1.01 (barely noticeable) to 100+ (catastrophic)

Fit quality: ±20% across 4 orders of magnitude.

### 7.4 Practical Use

```c
float ratio = sp_predicted_ppl_ratio(k_corr, params_b, bits);
if (ratio > 1.05) {
    // Skip this config — predicted PPL impact exceeds 5%
}
```

---

## 8. The Cauchy Reset Theory

### 8.1 The Problem

Over long decode chains, compression errors at each token contribute a small perturbation to the K/V cache. After N tokens, these perturbations accumulate. If the accumulated error becomes "timelike" (aligned with the model's information flow direction), the compressed past no longer determines the correct future — a Cauchy horizon forms.

### 8.2 The p=3 Sentinel

The p=3 spectral band energy is a scalar proxy for this alignment. The p=3 frequency is the lowest non-trivial frequency in the Vilenkin decomposition, making it sensitive to systematic drift while being robust to random noise.

The Ricci sentinel tracks:
```
p3_ratio = EMA(current_p3_energy / calibrated_p3_energy)
```

When |1 − p3_ratio| exceeds the threshold (model-size-dependent: 0.05 for 1B, 0.15 for 8B+), a reset is recommended.

### 8.3 The Mertens Oracle

The Mertens function M(n) = Σ_{k=1}^{n} μ(k) has oscillatory behaviour governed by the non-trivial zeros of the Riemann zeta function:

```
M(n) ≈ Σ_ρ c_ρ · n^ρ / ρ
```

where ρ ranges over the zeta zeros on the critical strip. The half-period of these oscillations at context scales n ~ 256–2048 is 200–500 tokens.

This matches empirically-observed optimal reset windows. The oracle pre-computes the risk schedule from the first 50 zeta zeros and provides O(1) lookup.

---

## 9. PrimePE: Lattice-Aligned Frequencies

### 9.1 The Frequency Mismatch

Standard RoPE uses geometric frequencies: θ_i = base^(−2i/d). These are irrational numbers that don't align with the multiplicative lattice of the integers. The mismatch means the RoPE frequency schedule doesn't maximally exploit the spectral structure that VHT2 captures.

### 9.2 The Correction

PrimePE blends in frequencies drawn from the lattice:

```
freq_factor[i] = (1 − α) · 1.0 + α · lattice_factor[i]
```

where lattice_factor[i] are computed from composite-tiered frequencies (coordinates in the multiplicative lattice). Alpha controls the blend — 0.17 is the validated default.

### 9.3 Validation

From the Position Is Arithmetic paper:

- Prime-tiered (129.2 PPL) and composite-tiered (129.4 PPL) perform identically — because composites are coordinates in the same lattice.
- Both dramatically outperform random frequencies (+5.0 PPL), scrambled tier assignment (+6.3 PPL), and pure ALiBi (+7.3 PPL).
- Three ZetaZeroPredictor transformers show geometric RoPE diverging (r=0.57) while lattice-aligned PE locks into a stable attractor (r=0.86, −80.7% MSE).

Production: −0.6% to −0.8% PPL improvement on Llama 3.2 1B at Q8_0/Q6_K/Q4_K_M, zero retraining.

---

## 10. Energy Concentration Theorem

The foundation of progressive band reads and the disk-tier architecture:

**Empirical fact:** After VHT2 + Möbius reorder, the spectral energy of RoPE'd K vectors concentrates in the early bands and decays toward the tail.

Measured on real K vectors (hd=128, ship path):
- Band 0 (coefficients 0–31): ~30% of total energy
- Bands 0–1 (coefficients 0–63): ~86% of total energy
- Bands 0–2 (coefficients 0–95): ~88% of total energy
- Full (all 128 coefficients): 99%+ reconstructable

This concentration is a consequence of:
1. RoPE imprinting smooth low-frequency structure
2. VHT2 capturing that structure in its early basis functions
3. Möbius reorder pushing the highest-energy (squarefree) indices to the front

The concentration is stronger for K vectors than for V vectors (V vectors have smoother distributions without the strong frequency signature from RoPE), which is why banded quantization helps K more than V.

---

## References

1. **Position Is Arithmetic: The Multiplicative Lattice as the Natural Basis for Positional Encoding** (Knack, 2026, v8). The foundational paper. PrimePE, the scaling law, spinor sheet bit validation.

2. **The KV Cache Is a View: Spectral Compression, the K-corr Scaling Law, and Basis-Dependent Möbius Prediction** (Knack, 2026, v2). The compression paper. Scaling law derivation, Möbius predictor characterization, spinor Pareto shift.

3. **The Multiplicative Lattice** (combined framework). Vilenkin-Hartley transform, squarefree factorization, Knight masks, CSR predictor.

4. **DeepSeek-V4** (DeepSeek AI, 2026). 1.6T FP8 MoE. Independent validation of KV-compression + sliding-window + prefetch-oracle architecture.
