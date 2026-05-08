// Shannon-Prime: Theoretical Mathematics Module
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// ============================================================================
// Research-grade implementations of advanced mathematical structures for
// transformer inference. These are desktop-only — phone hardware constraints
// make them impractical on mobile (see PHONE-ENGINE.md Part IV).
//
// Contents:
//   1. Quaternion Weight Matrices (Hamilton product rotation)
//   2. Poincare Disk / Hyperbolic Attention (gyrovector operations)
//   3. Ricci Flow Regularization (weight manifold smoothing)
//   4. Free Energy Minimization Objective (variational band selection)
//   5. Stereographic Projection (sphere <-> plane mapping)
//   6. Heat Equation Diffusion Smoothing
//   7. Fano Plane Multiplication Table (octonion-inspired mixing)
//
// Integration points:
//   - Quaternion: alternative to standard linear projection in attention
//   - Poincare: System 2 attention scoring for hierarchical text
//   - Ricci flow: weight regularization loss term during calibration
//   - Free energy: automatic band-bit allocation at model-pack time
//   - Stereographic: normalisation layer for hyperbolic embeddings
//   - Heat equation: KV cache smoothing between System 1/2 transitions
//   - Fano plane: MoE expert mixing pattern (7-element combinatorial)
// ============================================================================

#ifndef SP_THEORETICAL_H
#define SP_THEORETICAL_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 1. QUATERNION WEIGHT MATRICES
// ============================================================================
//
// Hamilton product: q = (a, b, c, d) where q = a + bi + cj + dk
//   q1 * q2 = (a1a2 - b1b2 - c1c2 - d1d2,
//              a1b2 + b1a2 + c1d2 - d1c2,
//              a1c2 - b1d2 + c1a2 + d1b2,
//              a1d2 + b1c2 - c1b2 + d1a2)
//
// Cost: 4x FLOPs vs real matmul. Benefit: tighter spectral concentration
// in VHT2 domain due to multi-dimensional rotation. Desktop only.

typedef struct {
    float a, b, c, d;  // scalar, i, j, k components
} sp_quaternion_t;

static inline sp_quaternion_t sp_quat_mul(sp_quaternion_t p, sp_quaternion_t q) {
    sp_quaternion_t r;
    r.a = p.a*q.a - p.b*q.b - p.c*q.c - p.d*q.d;
    r.b = p.a*q.b + p.b*q.a + p.c*q.d - p.d*q.c;
    r.c = p.a*q.c - p.b*q.d + p.c*q.a + p.d*q.b;
    r.d = p.a*q.d + p.b*q.c - p.c*q.b + p.d*q.a;
    return r;
}

static inline sp_quaternion_t sp_quat_conj(sp_quaternion_t q) {
    return (sp_quaternion_t){ q.a, -q.b, -q.c, -q.d };
}

static inline float sp_quat_norm(sp_quaternion_t q) {
    return sqrtf(q.a*q.a + q.b*q.b + q.c*q.c + q.d*q.d);
}

static inline sp_quaternion_t sp_quat_normalize(sp_quaternion_t q) {
    float n = sp_quat_norm(q);
    if (n < 1e-12f) return (sp_quaternion_t){1,0,0,0};
    float inv = 1.0f / n;
    return (sp_quaternion_t){ q.a*inv, q.b*inv, q.c*inv, q.d*inv };
}

// Quaternion rotation of a 3D vector v = (x,y,z) by unit quaternion q:
//   v' = q * (0,v) * q^{-1}
static inline void sp_quat_rotate_vec3(sp_quaternion_t q,
                                       float x, float y, float z,
                                       float *ox, float *oy, float *oz) {
    sp_quaternion_t v = {0, x, y, z};
    sp_quaternion_t qc = sp_quat_conj(q);
    sp_quaternion_t r = sp_quat_mul(sp_quat_mul(q, v), qc);
    *ox = r.b; *oy = r.c; *oz = r.d;
}

// Quaternion linear layer: W_q * x where W_q is a quaternion weight matrix.
// Input x is treated as pairs of complex numbers packed into quaternions.
// dim must be divisible by 4.
void sp_quat_linear(const sp_quaternion_t *W, const float *x, float *y,
                    int in_dim4, int out_dim4);

// ============================================================================
// 2. POINCARE DISK / HYPERBOLIC ATTENTION
// ============================================================================
//
// The Poincare disk model maps hierarchical structures into hyperbolic space
// where distances grow exponentially near the boundary. This is natural for
// tree-structured text (syntax trees, document outlines, code ASTs).
//
// Gyrovector operations replace Euclidean vector addition with Mobius
// addition, which respects the hyperbolic metric.
//
// WARNING: The (1-||x||^2) denominator is numerically catastrophic on int16.
// This requires fp32 minimum — NEON only, not HVX. Desktop/System 2 only.

// Poincare disk conformal factor: lambda_x = 2 / (1 - ||x||^2)
static inline float sp_poincare_lambda(const float *x, int dim) {
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) norm_sq += x[i] * x[i];
    float denom = 1.0f - norm_sq;
    if (denom < 1e-7f) denom = 1e-7f;  // clamp to avoid singularity at boundary
    return 2.0f / denom;
}

// Mobius addition: x (+) y in the Poincare disk.
// Formula: (x (+) y) = ((1+2<x,y>+||y||^2)*x + (1-||x||^2)*y) /
//                       (1+2<x,y>+||x||^2*||y||^2)
void sp_poincare_mobius_add(const float *x, const float *y, float *out, int dim);

// Poincare distance: d(x,y) = acosh(1 + 2*||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
float sp_poincare_distance(const float *x, const float *y, int dim);

// Hyperbolic attention score: uses Poincare distance instead of dot product.
// score(Q,K) = -beta * d_poincare(Q, K)^2
// where beta is a learnable temperature parameter.
void sp_poincare_attention_scores(const float *Q, const float *K,
                                  float *scores,
                                  int n_queries, int n_keys, int dim,
                                  float beta);

// Exponential map: project from tangent space at origin to Poincare disk.
// exp_0(v) = tanh(||v||/2) * v/||v||
void sp_poincare_exp_map(const float *v, float *out, int dim);

// Logarithmic map: project from Poincare disk to tangent space at origin.
// log_0(x) = 2*atanh(||x||) * x/||x||
void sp_poincare_log_map(const float *x, float *out, int dim);

// ============================================================================
// 3. RICCI FLOW REGULARIZATION
// ============================================================================
//
// Discrete Ricci flow on the weight manifold: smooth the curvature of the
// loss landscape by penalising deviations from mean curvature.
//
// L_ricci = lambda * ||nabla^2 W - mean(nabla^2 W)||^2
//
// This is a regularisation loss term added during calibration, not a
// runtime operation. It encourages the weight manifold to have uniform
// curvature, which improves the conditioning of VHT2 compression.

typedef struct {
    float lambda;       // regularisation strength (default 0.01)
    int   dim;          // weight matrix dimension
    float *laplacian;   // scratch buffer for nabla^2 W  (dim elements)
    float mean_curv;    // running mean curvature
} sp_ricci_flow_t;

int   sp_ricci_flow_init(sp_ricci_flow_t *rf, int dim, float lambda);
void  sp_ricci_flow_free(sp_ricci_flow_t *rf);

// Compute the Ricci flow regularisation loss for a weight vector.
// Approximates nabla^2 via finite differences on the 1D weight sequence.
float sp_ricci_flow_loss(sp_ricci_flow_t *rf, const float *weights, int n);

// Compute the gradient of the Ricci flow loss w.r.t. weights.
// grad_out must be pre-allocated with n elements.
void  sp_ricci_flow_grad(sp_ricci_flow_t *rf, const float *weights,
                         float *grad_out, int n);

// ============================================================================
// 4. FREE ENERGY MINIMIZATION OBJECTIVE
// ============================================================================
//
// Variational free energy for automatic band-bit allocation:
//   F = Complexity - Accuracy
//     = KL(q(z|x) || p(z)) - E_q[log p(x|z)]
//
// Where:
//   q(z|x) = the VHT2 compressed representation
//   p(z)   = prior from the Zeta manifold (spectral structure of natural data)
//   p(x|z) = reconstruction likelihood (how well we can recover from compressed)
//
// Minimising F at calibration time automatically selects the optimal band-bit
// allocation for each architecture. This replaces manual grid search.

typedef struct {
    int     n_bands;
    int    *candidate_bits;    // array of candidate bit widths per band
    int     n_candidates;      // number of candidate allocations
    float  *F_scores;          // free energy score per candidate
    float   best_F;
    int     best_idx;
} sp_free_energy_t;

int  sp_free_energy_init(sp_free_energy_t *fe, int n_bands, int n_candidates);
void sp_free_energy_free(sp_free_energy_t *fe);

// Evaluate free energy for one candidate allocation.
// kl_divergence: KL(compressed || prior) for this allocation
// reconstruction_loss: -E[log p(x|z)] (negative log-likelihood)
float sp_free_energy_eval(sp_free_energy_t *fe, int candidate_idx,
                          float kl_divergence, float reconstruction_loss);

// After evaluating all candidates, return the optimal band-bit allocation.
// Returns pointer to the best allocation's bit array (n_bands elements).
const int* sp_free_energy_best(const sp_free_energy_t *fe);

// ============================================================================
// 5. STEREOGRAPHIC PROJECTION
// ============================================================================
//
// Maps between the n-sphere S^n and R^n (tangent plane). Used as a
// normalisation layer for hyperbolic embeddings — project to the plane
// for linear operations, then back to the sphere for geometric ops.
//
// Forward (sphere -> plane): pi(x) = x[1..n] / (1 + x[0])
// Inverse (plane -> sphere): pi^{-1}(y) = (||y||^2-1, 2*y) / (||y||^2+1)

// Project from S^{dim} (dim+1 components) to R^{dim} (dim components).
void sp_stereo_project(const float *sphere, float *plane, int dim);

// Inverse: from R^{dim} to S^{dim}.
void sp_stereo_inverse(const float *plane, float *sphere, int dim);

// ============================================================================
// 6. HEAT EQUATION DIFFUSION SMOOTHING
// ============================================================================
//
// Discrete heat equation on the KV cache coefficient sequence:
//   u(t+dt) = u(t) + dt * nabla^2 u(t)
//
// This smooths the coefficient spectrum, reducing high-frequency noise
// that degrades compression quality at band boundaries. Applied as a
// post-processing step after VHT2 forward transform.
//
// The diffusion coefficient dt controls smoothing intensity:
//   dt too small: no effect
//   dt too large: destroys signal (CFL condition: dt < dx^2 / 2)

// In-place 1D heat equation diffusion step.
// Applies n_steps iterations of the discrete Laplacian.
void sp_heat_diffuse_1d(float *u, int len, float dt, int n_steps);

// 2D diffusion on a (rows x cols) coefficient matrix.
// Used for per-head coefficient smoothing across the position dimension.
void sp_heat_diffuse_2d(float *u, int rows, int cols, float dt, int n_steps);

// ============================================================================
// 7. FANO PLANE MULTIPLICATION TABLE
// ============================================================================
//
// The Fano plane is the smallest finite projective plane: 7 points,
// 7 lines, 3 points per line, 3 lines per point. It encodes the
// multiplication table of the octonions (Cayley numbers).
//
// In MoE context: the Fano plane provides a combinatorial pattern for
// mixing 7 experts in a way that every pair of experts shares exactly
// one "interaction line". This creates a balanced mixing structure
// where no expert pair is over- or under-represented.
//
// For models with n_experts != 7, we tile/fold the Fano structure.

// Fano plane lines (each line has 3 point indices, 0-indexed):
//   L0: {0,1,3}  L1: {1,2,4}  L2: {2,3,5}  L3: {3,4,6}
//   L4: {4,5,0}  L5: {5,6,1}  L6: {6,0,2}
extern const int sp_fano_lines[7][3];

// Fano plane multiplication: e_i * e_j = sign * e_k
// Returns the product index k and sets *sign to +1 or -1.
// For i==j, returns 0 (scalar, sign==-1 since e_i^2 = -1 for octonions).
int sp_fano_multiply(int i, int j, int *sign);

// Check if two experts share a Fano line (should interact).
// Returns the line index (0-6) or -1 if they don't share a line.
int sp_fano_shared_line(int expert_a, int expert_b);

// Generate a Fano-balanced mixing matrix for n_experts (folds if n_experts != 7).
// mixing_matrix is [n_experts x n_experts], output weights in [0,1].
void sp_fano_mixing_matrix(float *mixing_matrix, int n_experts);

#ifdef __cplusplus
}
#endif

#endif // SP_THEORETICAL_H
