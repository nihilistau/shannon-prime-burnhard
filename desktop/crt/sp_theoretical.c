// Shannon-Prime: Theoretical Mathematics Module — Implementation
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "sp_theoretical.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <stdbool.h>

// ============================================================================
// 1. QUATERNION WEIGHT MATRICES
// ============================================================================

void sp_quat_linear(const sp_quaternion_t *W, const float *x, float *y,
                    int in_dim4, int out_dim4) {
    // Quaternion linear layer: each output quaternion is the Hamilton product
    // of a weight quaternion row with the input quaternion column.
    // x is treated as in_dim4 quaternions (4*in_dim4 floats).
    // y is out_dim4 quaternions (4*out_dim4 floats).

    for (int o = 0; o < out_dim4; o++) {
        sp_quaternion_t acc = {0, 0, 0, 0};
        for (int i = 0; i < in_dim4; i++) {
            sp_quaternion_t w = W[o * in_dim4 + i];
            sp_quaternion_t xi = {
                x[i*4+0], x[i*4+1], x[i*4+2], x[i*4+3]
            };
            sp_quaternion_t prod = sp_quat_mul(w, xi);
            acc.a += prod.a;
            acc.b += prod.b;
            acc.c += prod.c;
            acc.d += prod.d;
        }
        y[o*4+0] = acc.a;
        y[o*4+1] = acc.b;
        y[o*4+2] = acc.c;
        y[o*4+3] = acc.d;
    }
}

// ============================================================================
// 2. POINCARE DISK / HYPERBOLIC ATTENTION
// ============================================================================

void sp_poincare_mobius_add(const float *x, const float *y, float *out, int dim) {
    float x_sq = 0.0f, y_sq = 0.0f, xy = 0.0f;
    for (int i = 0; i < dim; i++) {
        x_sq += x[i] * x[i];
        y_sq += y[i] * y[i];
        xy   += x[i] * y[i];
    }

    float num_coeff_x = 1.0f + 2.0f * xy + y_sq;
    float num_coeff_y = 1.0f - x_sq;
    float denom = 1.0f + 2.0f * xy + x_sq * y_sq;

    if (fabsf(denom) < 1e-10f) denom = 1e-10f;
    float inv_denom = 1.0f / denom;

    for (int i = 0; i < dim; i++) {
        out[i] = (num_coeff_x * x[i] + num_coeff_y * y[i]) * inv_denom;
    }

    // Clamp to disk interior (||out|| < 1 - epsilon).
    float out_sq = 0.0f;
    for (int i = 0; i < dim; i++) out_sq += out[i] * out[i];
    if (out_sq > 0.999f * 0.999f) {
        float scale = 0.999f / sqrtf(out_sq);
        for (int i = 0; i < dim; i++) out[i] *= scale;
    }
}

float sp_poincare_distance(const float *x, const float *y, int dim) {
    float diff_sq = 0.0f, x_sq = 0.0f, y_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - y[i];
        diff_sq += d * d;
        x_sq += x[i] * x[i];
        y_sq += y[i] * y[i];
    }

    float denom = (1.0f - x_sq) * (1.0f - y_sq);
    if (denom < 1e-10f) denom = 1e-10f;

    float arg = 1.0f + 2.0f * diff_sq / denom;
    if (arg < 1.0f) arg = 1.0f;  // numerical guard

    return acoshf(arg);
}

void sp_poincare_attention_scores(const float *Q, const float *K,
                                  float *scores,
                                  int n_queries, int n_keys, int dim,
                                  float beta) {
    for (int q = 0; q < n_queries; q++) {
        for (int k = 0; k < n_keys; k++) {
            float d = sp_poincare_distance(Q + q * dim, K + k * dim, dim);
            scores[q * n_keys + k] = -beta * d * d;
        }
    }
}

void sp_poincare_exp_map(const float *v, float *out, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) norm += v[i] * v[i];
    norm = sqrtf(norm);

    if (norm < 1e-10f) {
        memcpy(out, v, dim * sizeof(float));
        return;
    }

    float scale = tanhf(norm / 2.0f) / norm;
    for (int i = 0; i < dim; i++) out[i] = v[i] * scale;
}

void sp_poincare_log_map(const float *x, float *out, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) norm += x[i] * x[i];
    norm = sqrtf(norm);

    if (norm < 1e-10f) {
        memcpy(out, x, dim * sizeof(float));
        return;
    }

    float scale = 2.0f * atanhf(norm) / norm;
    for (int i = 0; i < dim; i++) out[i] = x[i] * scale;
}

// ============================================================================
// 3. RICCI FLOW REGULARIZATION
// ============================================================================

int sp_ricci_flow_init(sp_ricci_flow_t *rf, int dim, float lambda) {
    if (!rf || dim <= 0) return -1;
    memset(rf, 0, sizeof(*rf));
    rf->dim = dim;
    rf->lambda = lambda;
    rf->laplacian = (float *)calloc(dim, sizeof(float));
    if (!rf->laplacian) return -1;
    return 0;
}

void sp_ricci_flow_free(sp_ricci_flow_t *rf) {
    if (!rf) return;
    free(rf->laplacian);
    memset(rf, 0, sizeof(*rf));
}

float sp_ricci_flow_loss(sp_ricci_flow_t *rf, const float *weights, int n) {
    if (!rf || !weights || n < 3) return 0.0f;

    // Compute discrete Laplacian: nabla^2 w_i = w_{i-1} - 2*w_i + w_{i+1}
    float mean_curv = 0.0f;
    for (int i = 1; i < n - 1; i++) {
        rf->laplacian[i] = weights[i-1] - 2.0f * weights[i] + weights[i+1];
        mean_curv += rf->laplacian[i];
    }
    // Boundary: Neumann (zero-flux)
    rf->laplacian[0] = rf->laplacian[1];
    rf->laplacian[n-1] = rf->laplacian[n-2];
    mean_curv += rf->laplacian[0] + rf->laplacian[n-1];
    mean_curv /= (float)n;
    rf->mean_curv = mean_curv;

    // L_ricci = lambda * sum_i (laplacian[i] - mean_curv)^2
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = rf->laplacian[i] - mean_curv;
        loss += diff * diff;
    }

    return rf->lambda * loss;
}

void sp_ricci_flow_grad(sp_ricci_flow_t *rf, const float *weights,
                        float *grad_out, int n) {
    if (!rf || !weights || !grad_out || n < 3) return;

    // Recompute Laplacian + mean (in case loss wasn't called first)
    sp_ricci_flow_loss(rf, weights, n);

    // Gradient of L_ricci w.r.t. w_i via chain rule through the Laplacian.
    // d(laplacian[j])/d(w_i) is the discrete Laplacian stencil [1,-2,1]
    // applied at the right offset. For simplicity, use finite differences.
    float eps = 1e-5f;
    float *w_perturbed = (float *)malloc(n * sizeof(float));
    if (!w_perturbed) return;

    for (int i = 0; i < n; i++) {
        memcpy(w_perturbed, weights, n * sizeof(float));
        w_perturbed[i] += eps;
        float loss_plus = sp_ricci_flow_loss(rf, w_perturbed, n);

        w_perturbed[i] = weights[i] - eps;
        float loss_minus = sp_ricci_flow_loss(rf, w_perturbed, n);

        grad_out[i] = (loss_plus - loss_minus) / (2.0f * eps);
    }

    free(w_perturbed);

    // Restore internal state
    sp_ricci_flow_loss(rf, weights, n);
}

// ============================================================================
// 4. FREE ENERGY MINIMIZATION
// ============================================================================

int sp_free_energy_init(sp_free_energy_t *fe, int n_bands, int n_candidates) {
    if (!fe || n_bands <= 0 || n_candidates <= 0) return -1;
    memset(fe, 0, sizeof(*fe));

    fe->n_bands = n_bands;
    fe->n_candidates = n_candidates;
    fe->candidate_bits = (int *)calloc(n_candidates * n_bands, sizeof(int));
    fe->F_scores = (float *)calloc(n_candidates, sizeof(float));

    if (!fe->candidate_bits || !fe->F_scores) {
        sp_free_energy_free(fe);
        return -1;
    }

    fe->best_F = FLT_MAX;
    fe->best_idx = -1;

    return 0;
}

void sp_free_energy_free(sp_free_energy_t *fe) {
    if (!fe) return;
    free(fe->candidate_bits);
    free(fe->F_scores);
    memset(fe, 0, sizeof(*fe));
}

float sp_free_energy_eval(sp_free_energy_t *fe, int candidate_idx,
                          float kl_divergence, float reconstruction_loss) {
    if (!fe || candidate_idx < 0 || candidate_idx >= fe->n_candidates) return FLT_MAX;

    // F = Complexity - Accuracy = KL - E[log p(x|z)]
    // reconstruction_loss is already -E[log p(x|z)] (positive = worse)
    float F = kl_divergence + reconstruction_loss;

    fe->F_scores[candidate_idx] = F;

    if (F < fe->best_F) {
        fe->best_F = F;
        fe->best_idx = candidate_idx;
    }

    return F;
}

const int* sp_free_energy_best(const sp_free_energy_t *fe) {
    if (!fe || fe->best_idx < 0) return NULL;
    return fe->candidate_bits + fe->best_idx * fe->n_bands;
}

// ============================================================================
// 5. STEREOGRAPHIC PROJECTION
// ============================================================================

void sp_stereo_project(const float *sphere, float *plane, int dim) {
    // Forward: pi(x) = x[1..dim] / (1 + x[0])
    float denom = 1.0f + sphere[0];
    if (fabsf(denom) < 1e-10f) denom = 1e-10f;
    float inv = 1.0f / denom;
    for (int i = 0; i < dim; i++) {
        plane[i] = sphere[i + 1] * inv;
    }
}

void sp_stereo_inverse(const float *plane, float *sphere, int dim) {
    // Inverse: pi^{-1}(y) = (||y||^2 - 1, 2*y) / (||y||^2 + 1)
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) norm_sq += plane[i] * plane[i];
    float inv = 1.0f / (norm_sq + 1.0f);
    sphere[0] = (norm_sq - 1.0f) * inv;
    for (int i = 0; i < dim; i++) {
        sphere[i + 1] = 2.0f * plane[i] * inv;
    }
}

// ============================================================================
// 6. HEAT EQUATION DIFFUSION
// ============================================================================

void sp_heat_diffuse_1d(float *u, int len, float dt, int n_steps) {
    if (!u || len < 3 || n_steps <= 0) return;

    // CFL stability condition: dt < 0.5 * dx^2 (dx = 1 for discrete case)
    if (dt > 0.5f) dt = 0.5f;

    float *tmp = (float *)malloc(len * sizeof(float));
    if (!tmp) return;

    for (int step = 0; step < n_steps; step++) {
        // Neumann boundary: du/dx = 0 at boundaries
        tmp[0] = u[0] + dt * (u[1] - u[0]);
        for (int i = 1; i < len - 1; i++) {
            tmp[i] = u[i] + dt * (u[i-1] - 2.0f * u[i] + u[i+1]);
        }
        tmp[len-1] = u[len-1] + dt * (u[len-2] - u[len-1]);

        memcpy(u, tmp, len * sizeof(float));
    }

    free(tmp);
}

void sp_heat_diffuse_2d(float *u, int rows, int cols, float dt, int n_steps) {
    if (!u || rows < 3 || cols < 3 || n_steps <= 0) return;
    if (dt > 0.25f) dt = 0.25f;  // CFL for 2D: dt < dx^2 / 4

    float *tmp = (float *)malloc(rows * cols * sizeof(float));
    if (!tmp) return;

    for (int step = 0; step < n_steps; step++) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int idx = r * cols + c;

                // 5-point stencil Laplacian with Neumann boundaries
                float center = u[idx];
                float left   = (c > 0)        ? u[idx - 1]    : center;
                float right  = (c < cols - 1)  ? u[idx + 1]    : center;
                float up     = (r > 0)        ? u[idx - cols]  : center;
                float down   = (r < rows - 1)  ? u[idx + cols]  : center;

                tmp[idx] = center + dt * (left + right + up + down - 4.0f * center);
            }
        }
        memcpy(u, tmp, rows * cols * sizeof(float));
    }

    free(tmp);
}

// ============================================================================
// 7. FANO PLANE
// ============================================================================

// The 7 lines of the Fano plane (each is a set of 3 points, 0-indexed).
const int sp_fano_lines[7][3] = {
    {0, 1, 3},  // L0
    {1, 2, 4},  // L1
    {2, 3, 5},  // L2
    {3, 4, 6},  // L3
    {4, 5, 0},  // L4
    {5, 6, 1},  // L5
    {6, 0, 2},  // L6
};

// Fano plane multiplication table (octonion basis e1..e7).
// fano_product[i][j] = (sign, product_index).
// Encodes e_i * e_j = sign * e_k where k = fano_product[i][j].
// This is the Cayley-Dickson construction restricted to the Fano plane.
static const int fano_product_idx[7][7] = {
    // e0  e1  e2  e3  e4  e5  e6
    {  0,  3,  6,  1,  5,  4,  2 },  // e0 *
    {  3,  1,  4,  0,  2,  6,  5 },  // e1 *
    {  6,  4,  2,  5,  1,  3,  0 },  // e2 *
    {  1,  0,  5,  3,  6,  2,  4 },  // e3 *
    {  5,  2,  1,  6,  4,  0,  3 },  // e4 *
    {  4,  6,  3,  2,  0,  5,  1 },  // e5 *
    {  2,  5,  0,  4,  3,  1,  6 },  // e6 *
};

static const int fano_product_sign[7][7] = {
    // +1 if on a Fano line in the correct cyclic order, -1 otherwise
    { -1, +1, -1, +1, +1, -1, +1 },
    { -1, -1, +1, -1, +1, +1, -1 },
    { +1, -1, -1, +1, -1, +1, +1 },
    { +1, +1, -1, -1, +1, -1, -1 },
    { -1, +1, +1, -1, -1, +1, -1 },
    { +1, -1, +1, +1, -1, -1, +1 },
    { -1, +1, -1, +1, +1, -1, -1 },
};

int sp_fano_multiply(int i, int j, int *sign) {
    if (i < 0 || i >= 7 || j < 0 || j >= 7 || !sign) {
        if (sign) *sign = 0;
        return -1;
    }
    *sign = fano_product_sign[i][j];
    return fano_product_idx[i][j];
}

int sp_fano_shared_line(int expert_a, int expert_b) {
    int a = expert_a % 7;
    int b = expert_b % 7;
    if (a == b) return -1;

    for (int line = 0; line < 7; line++) {
        bool has_a = false, has_b = false;
        for (int p = 0; p < 3; p++) {
            if (sp_fano_lines[line][p] == a) has_a = true;
            if (sp_fano_lines[line][p] == b) has_b = true;
        }
        if (has_a && has_b) return line;
    }
    return -1;
}

void sp_fano_mixing_matrix(float *mixing_matrix, int n_experts) {
    if (!mixing_matrix || n_experts <= 0) return;

    // Base: experts on the same Fano line get weight 1.0,
    // others get a decay factor.
    for (int i = 0; i < n_experts; i++) {
        for (int j = 0; j < n_experts; j++) {
            if (i == j) {
                mixing_matrix[i * n_experts + j] = 1.0f;
            } else {
                int line = sp_fano_shared_line(i % 7, j % 7);
                if (line >= 0) {
                    // Same Fano line: strong interaction
                    mixing_matrix[i * n_experts + j] = 0.8f;
                } else {
                    // No shared line: weak interaction
                    mixing_matrix[i * n_experts + j] = 0.2f;
                }
            }
        }
    }
}
