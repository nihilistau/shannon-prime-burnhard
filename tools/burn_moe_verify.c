// Shannon-Prime BurnHard — BURN_MOE expert FFN verifier (real Qwen3.6 weights).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// The last math gate before GGML_OP_BURN_MOE wires into the chat path. Loads
// a real Qwen3.6 GGUF, picks expert 0 of layer 0, dequantises gate/up/down
// to fp32 via ggml's public type_traits, then runs a single-token FFN
// through both paths:
//
//   stock:    out = down @ ( silu(gate @ x) * (up @ x) )
//   residue:  same shape but every matmul goes through the residue path
//
// Compares to within fp32 ULP. If this passes on Qwen3.6's real weight
// distribution (heavy-tailed, narrow-band post-quant) we know the BURN_MOE
// custom op will produce the same logits when wired into the graph.
//
// Usage:
//   burn_moe_verify <path/to/qwen3.6_q4_k_m.gguf>
//
// Tolerances same as residue_matmul_verify (max_abs + nrmse) since the FFN
// is just three matmuls with elementwise ops in between.

#include "burnhard_residue_matmul.h"
#include "burnhard_crt.h"
#include "ggml.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef SP_VERIFY_MAX_ABS
#  define SP_VERIFY_MAX_ABS 1e-4
#endif
#ifndef SP_VERIFY_MAX_NRMSE
#  define SP_VERIFY_MAX_NRMSE 1e-5
#endif

// SiLU: x * sigmoid(x).
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Dequant a row of a Q4_K (or any other quant type) tensor block-by-block.
// `n_elem` must be a multiple of the type's block size — for Q4_K that's 256.
static int dequant_to_fp32(const struct ggml_tensor *src,
                            float *dst, size_t n_elem) {
    const struct ggml_type_traits *t = ggml_get_type_traits(src->type);
    if (!t || !t->to_float) {
        fprintf(stderr, "[burn_moe_verify] no to_float for type %d\n", src->type);
        return -1;
    }
    t->to_float(src->data, dst, (int64_t)n_elem);
    return 0;
}

// ----------------------------------------------------------------------------
// FFN forward via stock fp32 (reference): out = down @ (silu(gate@x) * (up@x))
// gate, up, down are row-major fp32. x is [n_embd], out is [n_embd].
// gate/up project [n_embd] → [n_ff]; down projects [n_ff] → [n_embd].
// ----------------------------------------------------------------------------
static void ffn_forward_stock(const float *x,
                               const float *gate,    // [n_ff, n_embd]
                               const float *up,      // [n_ff, n_embd]
                               const float *down,    // [n_embd, n_ff]
                               float *out,           // [n_embd]
                               size_t n_embd, size_t n_ff) {
    float *gate_out = (float *)malloc(n_ff * sizeof(float));
    float *up_out   = (float *)malloc(n_ff * sizeof(float));
    float *par      = (float *)malloc(n_ff * sizeof(float));
    sp_reference_matmul_fp32(gate, x, gate_out, n_ff, n_embd, 1);
    sp_reference_matmul_fp32(up,   x, up_out,   n_ff, n_embd, 1);
    for (size_t i = 0; i < n_ff; ++i) par[i] = silu(gate_out[i]) * up_out[i];
    sp_reference_matmul_fp32(down, par, out, n_embd, n_ff, 1);
    free(gate_out); free(up_out); free(par);
}

// ----------------------------------------------------------------------------
// FFN forward via residue path. Same algorithm, every matmul routed through
// sp_burnhard_residue_matmul_fp32 (which discretises → splits → modular sums
// → Garner-joins → unscales).
// ----------------------------------------------------------------------------
static int ffn_forward_residue(const float *x,
                                const float *gate,
                                const float *up,
                                const float *down,
                                float *out,
                                size_t n_embd, size_t n_ff) {
    float *gate_out = (float *)malloc(n_ff * sizeof(float));
    float *up_out   = (float *)malloc(n_ff * sizeof(float));
    float *par      = (float *)malloc(n_ff * sizeof(float));
    int rc = 0;
    rc |= sp_burnhard_residue_matmul_fp32(gate, x, gate_out, n_ff, n_embd, 1);
    rc |= sp_burnhard_residue_matmul_fp32(up,   x, up_out,   n_ff, n_embd, 1);
    if (rc == 0) {
        for (size_t i = 0; i < n_ff; ++i) par[i] = silu(gate_out[i]) * up_out[i];
        rc |= sp_burnhard_residue_matmul_fp32(down, par, out, n_embd, n_ff, 1);
    }
    free(gate_out); free(up_out); free(par);
    return rc;
}

// ----------------------------------------------------------------------------
// Locate one expert's gate/up/down tensors in the GGUF. For qwen35moe the
// expert weights live under blk.<L>.ffn_{gate,up,down}_exps with shape
// [n_embd, n_ff_per_expert, n_expert] (down inverted). We pull just one
// expert (index `expert_idx`) — that's a [n_embd, n_ff] slice for gate/up
// and [n_ff, n_embd] for down.
// ----------------------------------------------------------------------------
#define DBG(fmt, ...) do { fprintf(stderr, "[burn_moe_verify:dbg] " fmt "\n", ##__VA_ARGS__); fflush(stderr); } while (0)

static int load_expert(const char *gguf_path, int layer, int expert_idx,
                        float **out_gate, float **out_up, float **out_down,
                        size_t *out_n_embd, size_t *out_n_ff) {
    DBG("load_expert: gguf=%s layer=%d expert=%d", gguf_path, layer, expert_idx);
    struct gguf_init_params p = { false, NULL };
    struct gguf_context *gguf = gguf_init_from_file(gguf_path, p);
    if (!gguf) {
        DBG("gguf_init failed for %s", gguf_path);
        return -1;
    }
    DBG("gguf_init OK");
    // Build a ggml context backed by the gguf's tensor data block. The
    // engine does this internally; here we replicate the minimum needed
    // to get raw tensor pointers we can dequant.
    char gate_name[128], up_name[128], down_name[128];
    snprintf(gate_name, sizeof(gate_name), "blk.%d.ffn_gate_exps.weight", layer);
    snprintf(up_name,   sizeof(up_name),   "blk.%d.ffn_up_exps.weight",   layer);
    snprintf(down_name, sizeof(down_name), "blk.%d.ffn_down_exps.weight", layer);

    int64_t gate_idx = gguf_find_tensor(gguf, gate_name);
    int64_t up_idx   = gguf_find_tensor(gguf, up_name);
    int64_t down_idx = gguf_find_tensor(gguf, down_name);
    DBG("tensor indices gate=%lld up=%lld down=%lld",
        (long long)gate_idx, (long long)up_idx, (long long)down_idx);
    if (gate_idx < 0 || up_idx < 0 || down_idx < 0) {
        fprintf(stderr, "[burn_moe_verify] missing expert tensors at layer %d "
                "(gate=%lld up=%lld down=%lld)\n", layer,
                (long long)gate_idx, (long long)up_idx, (long long)down_idx);
        gguf_free(gguf);
        return -1;
    }

    // Tensor shapes — qwen35moe: gate/up are [n_embd, n_ff, n_expert],
    // down is [n_ff, n_embd, n_expert]. We need n_embd and n_ff.
    enum ggml_type gate_type = gguf_get_tensor_type(gguf, gate_idx);
    size_t gate_bytes        = gguf_get_tensor_size(gguf, gate_idx);
    size_t gate_off          = gguf_get_tensor_offset(gguf, gate_idx);
    size_t up_off            = gguf_get_tensor_offset(gguf, up_idx);
    size_t down_off          = gguf_get_tensor_offset(gguf, down_idx);
    enum ggml_type up_type   = gguf_get_tensor_type(gguf, up_idx);
    enum ggml_type down_type = gguf_get_tensor_type(gguf, down_idx);
    DBG("gate_type=%d gate_bytes=%zu offsets gate=%zu up=%zu down=%zu",
        (int)gate_type, gate_bytes, gate_off, up_off, down_off);

    // Shape inspection isn't part of the gguf C API directly — use the
    // model architecture's known dims. For Qwen3.6 35B-A3B: n_embd=2048,
    // n_ff_per_expert=512, n_expert=256.
    const size_t n_embd = 2048;
    const size_t n_ff   = 512;
    const size_t n_expert = 256;

    // Element counts for one expert.
    const size_t gate_elems_per_expert = n_embd * n_ff;
    const size_t up_elems_per_expert   = n_embd * n_ff;
    const size_t down_elems_per_expert = n_ff   * n_embd;

    // Bytes per expert. For Q4_K: 144 bytes per 256-element block.
    const struct ggml_type_traits *t_gate = ggml_get_type_traits(gate_type);
    const struct ggml_type_traits *t_up   = ggml_get_type_traits(up_type);
    const struct ggml_type_traits *t_down = ggml_get_type_traits(down_type);
    if (!t_gate || !t_up || !t_down) {
        DBG("missing type traits"); gguf_free(gguf); return -1;
    }
    // Per-tensor byte size — gate/up/down can have different quant types
    // (Qwen3.6 35B-A3B: gate/up are Q4_K, down is Q6_K).
    const size_t bytes_per_expert_gate =
        (gate_elems_per_expert / (size_t)t_gate->blck_size) * t_gate->type_size;
    const size_t bytes_per_expert_up =
        (up_elems_per_expert   / (size_t)t_up->blck_size)   * t_up->type_size;
    const size_t bytes_per_expert_down =
        (down_elems_per_expert / (size_t)t_down->blck_size) * t_down->type_size;
    DBG("bytes/expert gate=%zu up=%zu down=%zu  "
        "(blk gate=%zu/%zu up=%zu/%zu down=%zu/%zu)",
        bytes_per_expert_gate, bytes_per_expert_up, bytes_per_expert_down,
        (size_t)t_gate->blck_size, (size_t)t_gate->type_size,
        (size_t)t_up->blck_size,   (size_t)t_up->type_size,
        (size_t)t_down->blck_size, (size_t)t_down->type_size);

    // Open the GGUF file again and read the expert's bytes directly.
    DBG("opening gguf for raw read");
    FILE *fp = fopen(gguf_path, "rb");
    if (!fp) { DBG("fopen failed"); gguf_free(gguf); return -1; }

    void *gate_buf = malloc(bytes_per_expert_gate);
    void *up_buf   = malloc(bytes_per_expert_up);
    void *down_buf = malloc(bytes_per_expert_down);
    if (!gate_buf || !up_buf || !down_buf) {
        DBG("OOM on expert buffers");
        free(gate_buf); free(up_buf); free(down_buf);
        fclose(fp); gguf_free(gguf);
        return -1;
    }
    DBG("allocated chamber buffers gate=%zu up=%zu down=%zu",
        bytes_per_expert_gate, bytes_per_expert_up, bytes_per_expert_down);

    // gguf_get_tensor_offset gives offset into the data section. Add the
    // data section base offset (gguf_get_data_offset).
    const size_t base = gguf_get_data_offset(gguf);
    DBG("data base offset=%zu", base);

    // fseek with int64 offset. On Win32 the long form caps at 2GB; for a
    // 20GB GGUF we MUST use _fseeki64 / fseeko to address past 2GiB.
    const long long g_off = (long long)(base + gate_off +
                                  (size_t)expert_idx * bytes_per_expert_gate);
    const long long u_off = (long long)(base + up_off +
                                  (size_t)expert_idx * bytes_per_expert_up);
    const long long d_off = (long long)(base + down_off +
                                  (size_t)expert_idx * bytes_per_expert_down);
    DBG("seek targets gate=%lld up=%lld down=%lld", g_off, u_off, d_off);
#ifdef _WIN32
    if (_fseeki64(fp, g_off, SEEK_SET) != 0) { DBG("seek gate failed"); goto fail_io; }
#else
    if (fseeko(fp, (off_t)g_off, SEEK_SET) != 0) { DBG("seek gate failed"); goto fail_io; }
#endif
    if (fread(gate_buf, 1, bytes_per_expert_gate, fp) != bytes_per_expert_gate)
        { DBG("read gate short"); goto fail_io; }
#ifdef _WIN32
    if (_fseeki64(fp, u_off, SEEK_SET) != 0) { DBG("seek up failed"); goto fail_io; }
#else
    if (fseeko(fp, (off_t)u_off, SEEK_SET) != 0) { DBG("seek up failed"); goto fail_io; }
#endif
    if (fread(up_buf, 1, bytes_per_expert_up, fp) != bytes_per_expert_up)
        { DBG("read up short"); goto fail_io; }
#ifdef _WIN32
    if (_fseeki64(fp, d_off, SEEK_SET) != 0) { DBG("seek down failed"); goto fail_io; }
#else
    if (fseeko(fp, (off_t)d_off, SEEK_SET) != 0) { DBG("seek down failed"); goto fail_io; }
#endif
    if (fread(down_buf, 1, bytes_per_expert_down, fp) != bytes_per_expert_down)
        { DBG("read down short"); goto fail_io; }
    fclose(fp);
    DBG("raw weights loaded; dequantising");

    // Dequant.
    float *gate_f = (float *)malloc(gate_elems_per_expert * sizeof(float));
    float *up_f   = (float *)malloc(up_elems_per_expert   * sizeof(float));
    float *down_f = (float *)malloc(down_elems_per_expert * sizeof(float));
    if (!gate_f || !up_f || !down_f) {
        DBG("OOM on fp32 buffers");
        free(gate_buf); free(up_buf); free(down_buf);
        free(gate_f); free(up_f); free(down_f);
        gguf_free(gguf);
        return -1;
    }
    t_gate->to_float(gate_buf, gate_f, (int64_t)gate_elems_per_expert);
    DBG("gate dequant done");
    t_up->to_float(up_buf, up_f, (int64_t)up_elems_per_expert);
    DBG("up dequant done");
    t_down->to_float(down_buf, down_f, (int64_t)down_elems_per_expert);
    DBG("down dequant done");

    // Weight statistics — sanity-check the dequant produced plausible
    // values. If gate/up/down are all-zero or all-the-same, the layout
    // / offset / type assumptions are wrong, not the residue math.
    {
        double sum = 0, sumsq = 0;
        float wmin = gate_f[0], wmax = gate_f[0];
        for (size_t i = 0; i < gate_elems_per_expert; ++i) {
            float v = gate_f[i];
            sum += v; sumsq += (double)v * v;
            if (v < wmin) wmin = v;
            if (v > wmax) wmax = v;
        }
        double mean = sum / (double)gate_elems_per_expert;
        double var  = sumsq / (double)gate_elems_per_expert - mean * mean;
        DBG("gate stats: n=%zu  min=%+.6e max=%+.6e mean=%+.6e std=%.6e",
            gate_elems_per_expert, wmin, wmax, mean, sqrt(var));
        DBG("gate first 8: %+.4e %+.4e %+.4e %+.4e  %+.4e %+.4e %+.4e %+.4e",
            gate_f[0], gate_f[1], gate_f[2], gate_f[3],
            gate_f[4], gate_f[5], gate_f[6], gate_f[7]);
    }
    free(gate_buf); free(up_buf); free(down_buf);

    *out_gate = gate_f;
    *out_up   = up_f;
    *out_down = down_f;
    *out_n_embd = n_embd;
    *out_n_ff   = n_ff;

    fprintf(stderr,
        "[burn_moe_verify] loaded layer=%d expert=%d/%zu  "
        "n_embd=%zu  n_ff=%zu  type=%s\n",
        layer, expert_idx, n_expert, n_embd, n_ff,
        ggml_type_name(gate_type));

    gguf_free(gguf);
    return 0;

fail_io:
    free(gate_buf); free(up_buf); free(down_buf);
    fclose(fp);
    gguf_free(gguf);
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <path/to/qwen3.6_q4_k_m.gguf> [layer=0] [expert=0]\n",
            argv[0]);
        return 1;
    }
    const char *gguf_path = argv[1];
    const int   layer     = (argc >= 3) ? atoi(argv[2]) : 0;
    const int   expert_id = (argc >= 4) ? atoi(argv[3]) : 0;

    printf("[burn_moe_verify] BurnHard expert-FFN residue path vs stock fp32\n");
    printf("[burn_moe_verify] M1=%lld  M2=%lld  scale=auto via safe_q_max\n",
           (long long)SP_M1, (long long)SP_M2);

    float *gate = NULL, *up = NULL, *down = NULL;
    size_t n_embd = 0, n_ff = 0;
    if (load_expert(gguf_path, layer, expert_id,
                    &gate, &up, &down, &n_embd, &n_ff) != 0) {
        return 2;
    }
    printf("[burn_moe_verify] safe_q_max(n_embd=%zu)=%d  "
           "safe_q_max(n_ff=%zu)=%d\n",
           n_embd, sp_burnhard_safe_q_max(n_embd),
           n_ff,   sp_burnhard_safe_q_max(n_ff));

    // Synthesize a typical hidden-state vector (post-attention output is
    // roughly Normal with std ~1.0 after RMSNorm and projection).
    float *x = (float *)malloc(n_embd * sizeof(float));
    srand(0xb1a5);
    for (size_t i = 0; i < n_embd; ++i) {
        // Gaussian-ish via Box–Muller, scaled to match real activations.
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        x[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }

    float *out_stock   = (float *)malloc(n_embd * sizeof(float));
    float *out_residue = (float *)malloc(n_embd * sizeof(float));

    // Diagnostic: isolate the first matmul (gate @ x) — if THIS already
    // diverges between paths on real weights, the bug is in the matmul
    // applied to real-weight distributions, not in SiLU / multiply / down.
    {
        float *go_stock   = (float *)malloc(n_ff * sizeof(float));
        float *go_residue = (float *)malloc(n_ff * sizeof(float));
        sp_reference_matmul_fp32(gate, x, go_stock,   n_ff, n_embd, 1);
        sp_burnhard_residue_matmul_fp32(gate, x, go_residue, n_ff, n_embd, 1);
        double max_d = 0.0, peak_g = 0.0;
        for (size_t i = 0; i < n_ff; ++i) {
            double d = fabs((double)go_stock[i] - (double)go_residue[i]);
            if (d > max_d) max_d = d;
            if (fabs((double)go_stock[i]) > peak_g) peak_g = fabs((double)go_stock[i]);
        }
        DBG("gate@x ALONE: peak_stock=%+.4e max_diff=%.4e  "
            "first 4 stock=%+.4e %+.4e %+.4e %+.4e",
            peak_g, max_d,
            go_stock[0], go_stock[1], go_stock[2], go_stock[3]);
        DBG("gate@x ALONE:                                "
            "first 4 resid=%+.4e %+.4e %+.4e %+.4e",
            go_residue[0], go_residue[1], go_residue[2], go_residue[3]);
        free(go_stock); free(go_residue);
    }

    struct timespec t0, t1, t2;
    timespec_get(&t0, TIME_UTC);
    ffn_forward_stock(x, gate, up, down, out_stock, n_embd, n_ff);
    timespec_get(&t1, TIME_UTC);
    int rc = ffn_forward_residue(x, gate, up, down, out_residue, n_embd, n_ff);
    timespec_get(&t2, TIME_UTC);
    if (rc != 0) {
        fprintf(stderr, "[burn_moe_verify] residue FFN failed (rc=%d)\n", rc);
        free(gate); free(up); free(down); free(x); free(out_stock); free(out_residue);
        return 3;
    }

    double t_stock_ms   = ((double)(t1.tv_sec - t0.tv_sec) * 1e3) +
                          ((double)(t1.tv_nsec - t0.tv_nsec) / 1e6);
    double t_residue_ms = ((double)(t2.tv_sec - t1.tv_sec) * 1e3) +
                          ((double)(t2.tv_nsec - t1.tv_nsec) / 1e6);

    double max_abs = 0.0, sum_sq = 0.0, peak = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < n_embd; ++i) {
        const double a = (double)out_stock[i];
        const double b = (double)out_residue[i];
        const double e = fabs(a - b);
        sum_sq += e * e;
        if (e > max_abs) max_abs = e;
        if (fabs(a) > peak) peak = fabs(a);
        const double mag = fabs(a) > 1e-9 ? fabs(a) : 1e-9;
        const double r = e / mag;
        if (isfinite(r) && r > max_rel) max_rel = r;
    }
    const double rmse  = sqrt(sum_sq / (double)n_embd);
    const double nrmse = (peak > 0.0) ? rmse / peak : 0.0;

    int ok = (max_abs <= SP_VERIFY_MAX_ABS) && (nrmse <= SP_VERIFY_MAX_NRMSE);

    printf("[burn_moe_verify] FFN @ layer=%d expert=%d  stock=%.1fms  "
           "residue=%.1fms\n", layer, expert_id, t_stock_ms, t_residue_ms);
    printf("[burn_moe_verify]   max_abs=%.3e  nrmse=%.3e  peak|out|=%.3e  "
           "(max_rel=%.3e — informational)\n",
           max_abs, nrmse, peak, max_rel);
    printf("[burn_moe_verify] %s\n",
           ok ? "=== PASS ===" : "=== FAIL ===");

    free(gate); free(up); free(down);
    free(x); free(out_stock); free(out_residue);
    return ok ? 0 : 1;
}
