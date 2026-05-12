// Shannon-Prime Engine — SP-Flash Phase 3 native draft oracle
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#include "sp_flash_oracle_native.h"
#include "forward.h"  // SpFlashCaptureCtx

#include "shannon_prime.h"              // sp_kcorr_*, sp_kcorr_floor, sp_mobius_mask_*
#include "shannon_prime_flash_native.h" // sp_flash_vilenkin_pred_*, sp_flash_mobius_noise

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>

namespace sp::engine {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static int32_t argmax_native(const float* data, int n) {
    int32_t best = 0;
    float   best_val = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_val) { best_val = data[i]; best = i; }
    }
    return best;
}

// K-corr + exact-match acceptance.
// Primary: exact token match. Fallback on mismatch: K-corr between
// noise_emb[i] (the position's draft embedding) and ctx_vec (the Vilenkin
// context, common to all positions). Squarefree positions have
// noise_emb[i] == ctx_vec, giving K-corr = 1.0 (always accept); prime-
// squared positions have noise_emb[i] == mask_emb, giving lower K-corr.
static std::vector<int32_t> accept_native(
    const std::vector<int32_t>& draft,
    const std::vector<float>&   target_logits,  // [bs * n_vocab]
    int                         n_vocab,
    const std::vector<float>&   noise_emb,      // [bs * hidden]
    const std::vector<float>&   ctx_vec,        // [hidden]
    int                         hidden,
    float                       kcorr_floor,
    float*                      scratch)        // [2 * hidden]
{
    std::vector<int32_t> result;
    result.reserve(draft.size() + 1);
    const int bs = (int)draft.size();
    for (int i = 0; i < bs; ++i) {
        const int32_t target_tok =
            argmax_native(target_logits.data() + (size_t)i * n_vocab, n_vocab);
        if (draft[i] == target_tok) {
            result.push_back(draft[i]);
            continue;
        }
        const float* ne_i = noise_emb.data() + (size_t)i * hidden;
        if (sp_kcorr_accept(ne_i, ctx_vec.data(), hidden, kcorr_floor, scratch)) {
            result.push_back(draft[i]);
        } else {
            result.push_back(target_tok);
            return result;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Minimal meta.json reader
// ---------------------------------------------------------------------------

static bool slurp_file(const std::string& path, std::string& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return false; }
    out.resize((size_t)sz);
    (void)fread(&out[0], 1, (size_t)sz, f);
    fclose(f);
    return true;
}

// Find the first integer value for the given JSON key.
static int json_int(const std::string& j, const char* key, int dfl) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = j.find(needle);
    if (pos == std::string::npos) return dfl;
    pos = j.find(':', pos);
    if (pos == std::string::npos) return dfl;
    pos++;
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t' || j[pos] == '\n')) pos++;
    if (pos >= j.size() || j[pos] == 'n') return dfl;  // null
    try { return std::stoi(j.substr(pos)); } catch (...) { return dfl; }
}

// Parse a JSON integer array value for the given key; returns empty on null or missing.
static std::vector<int> json_int_array(const std::string& j, const char* key) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = j.find(needle);
    if (pos == std::string::npos) return {};
    pos = j.find('[', pos);
    if (pos == std::string::npos) return {};
    auto end = j.find(']', pos);
    if (end == std::string::npos) return {};
    std::vector<int> result;
    pos++;
    while (pos < end) {
        while (pos < end && !std::isdigit((unsigned char)j[pos]) && j[pos] != '-') pos++;
        if (pos >= end) break;
        try {
            size_t consumed = 0;
            result.push_back(std::stoi(j.substr(pos), &consumed));
            pos += consumed;
        } catch (...) { break; }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

// 128 KB is ample for a single ggml_mul_mat descriptor (a few tensors).
static constexpr size_t kNativeCtxSize = 128 * 1024;

struct SpFlashOracleNative::Impl {
    // Vilenkin predictor per calibrated target layer.
    std::vector<sp_flash_vilenkin_pred_t> preds;
    int num_target_layers = 0;
    int hidden_size       = 2048;
    std::vector<int> layer_ids;   // target layer indices from meta.json

    // Möbius schedule: mu_block[i] = μ(i+1) for i in [0, block_size).
    std::vector<int8_t> mu_block;

    // Mask embedding (one row from target_tok_embd) — [hidden_size] floats.
    std::vector<float> mask_emb;

    // Non-owning pointer to target lm_head tensor.
    ggml_tensor* target_output = nullptr;

    // ggml CPU backend + gallocr for the per-step logit projection graph.
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr  = nullptr;
    std::vector<uint8_t> ctx_mem;

    // Config
    int   block_size     = 8;
    int   cur_block_size = 8;
    bool  adaptive       = false;
    float adaptive_hi    = 0.75f;
    float adaptive_lo    = 0.50f;
    int   adaptive_steps = 50;

    // K-corr acceptance
    float              kcorr_floor   = 0.95f;
    std::vector<float> kcorr_scratch;   // [2 * hidden_size]

    // EMA tracking
    float ema_accept = 0.0f;
    int   step_count = 0;

    // Per-step working buffers (pre-allocated to avoid repeated heap churn).
    std::vector<float> vilenkin_skel_scratch;  // [SP_FLASH_N_SKEL]
    std::vector<float> vilenkin_pred_scratch;  // [hidden_size]
    std::vector<float> ctx_vec;                // [hidden_size]
    std::vector<float> proposal_emb;           // [block_size * hidden_size]
    std::vector<float> noise_emb;              // [block_size * hidden_size]

    SpFlashCaptureCtx* capture = nullptr;

    ~Impl() {
        if (allocr)  ggml_gallocr_free(allocr);
        if (backend) ggml_backend_free(backend);
        for (auto& p : preds) sp_flash_vilenkin_pred_free(&p);
    }
};

// ---------------------------------------------------------------------------
// SpFlashOracleNative public API
// ---------------------------------------------------------------------------

SpFlashOracleNative::SpFlashOracleNative() : impl_(std::make_unique<Impl>()) {}
SpFlashOracleNative::~SpFlashOracleNative() = default;

std::unique_ptr<SpFlashOracleNative> SpFlashOracleNative::load(const Config& cfg) {
    if (cfg.calibration_dir.empty()) {
        std::fprintf(stderr, "[sp-native] calibration_dir is empty\n");
        return nullptr;
    }
    if (!cfg.target_tok_embd || !cfg.target_output) {
        std::fprintf(stderr, "[sp-native] target_tok_embd / target_output must be non-null\n");
        return nullptr;
    }

    auto oracle = std::unique_ptr<SpFlashOracleNative>(new SpFlashOracleNative());
    Impl* I = oracle->impl_.get();

    // -----------------------------------------------------------------------
    // 1. Parse meta.json
    // -----------------------------------------------------------------------
    const std::string meta_path = cfg.calibration_dir + "/meta.json";
    std::string meta_str;
    if (!slurp_file(meta_path, meta_str)) {
        std::fprintf(stderr, "[sp-native] cannot read '%s'\n", meta_path.c_str());
        return nullptr;
    }

    I->hidden_size       = json_int(meta_str, "hidden_size",       2048);
    I->num_target_layers = json_int(meta_str, "num_target_layers", 0);
    I->layer_ids         = json_int_array(meta_str, "target_layer_ids");

    if (I->num_target_layers <= 0) {
        std::fprintf(stderr, "[sp-native] meta.json: invalid num_target_layers=%d\n",
                     I->num_target_layers);
        return nullptr;
    }
    if (I->layer_ids.empty()) {
        // Absent or null — fall back to 0..N-1 (capture hooks will match by position).
        for (int li = 0; li < I->num_target_layers; ++li) I->layer_ids.push_back(li);
        std::fprintf(stderr,
            "[sp-native] meta.json: target_layer_ids absent, using 0..%d\n",
            I->num_target_layers - 1);
    }

    std::fprintf(stderr,
        "[sp-native] meta: hidden=%d num_target_layers=%d layer_ids[0]=%d\n",
        I->hidden_size, I->num_target_layers, I->layer_ids[0]);

    // -----------------------------------------------------------------------
    // 2. Load calibration bins → sp_flash_vilenkin_pred_t per layer
    //    Each layer_N.bin is float32 [hidden_size × SP_FLASH_N_SKEL] row-major.
    // -----------------------------------------------------------------------
    I->preds.resize((size_t)I->num_target_layers);
    const size_t w_elems = (size_t)I->hidden_size * SP_FLASH_N_SKEL;
    std::vector<float> w_buf(w_elems);

    for (int li = 0; li < I->num_target_layers; ++li) {
        const std::string bin_path =
            cfg.calibration_dir + "/layer_" + std::to_string(li) + ".bin";
        FILE* f = fopen(bin_path.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "[sp-native] cannot open '%s'\n", bin_path.c_str());
            return nullptr;
        }
        const size_t nread = fread(w_buf.data(), sizeof(float), w_elems, f);
        fclose(f);
        if (nread != w_elems) {
            std::fprintf(stderr,
                "[sp-native] '%s': expected %zu floats, read %zu\n",
                bin_path.c_str(), w_elems, nread);
            return nullptr;
        }
        if (sp_flash_vilenkin_pred_init(&I->preds[li], I->hidden_size, w_buf.data()) != 0) {
            std::fprintf(stderr,
                "[sp-native] sp_flash_vilenkin_pred_init failed for layer %d\n", li);
            return nullptr;
        }
    }
    std::fprintf(stderr, "[sp-native] loaded %d calibration W matrices\n",
                 I->num_target_layers);

    // -----------------------------------------------------------------------
    // 3. Pre-compute Möbius mu array: mu_block[i] = μ(i+1) for i in [0, bs)
    // -----------------------------------------------------------------------
    I->block_size     = cfg.block_size > 0 ? cfg.block_size : 8;
    I->cur_block_size = I->block_size;
    const int bs = I->block_size;

    {
        sp_mobius_mask_t mob = {};
        if (sp_mobius_mask_init(&mob, bs + 1) != 0) {
            std::fprintf(stderr, "[sp-native] sp_mobius_mask_init failed\n");
            return nullptr;
        }
        I->mu_block.resize((size_t)bs);
        for (int i = 0; i < bs; ++i) I->mu_block[i] = mob.mu[i + 1];
        sp_mobius_mask_free(&mob);
    }

    // -----------------------------------------------------------------------
    // 4. Read mask embedding: one row from target_tok_embd.
    //    target_tok_embd is [hidden_size, vocab_size] col-major → row mask_tok_id
    //    starts at byte offset (mask_tok_id * hidden_size * sizeof(float)).
    // -----------------------------------------------------------------------
    I->mask_emb.resize((size_t)I->hidden_size);
    ggml_backend_tensor_get(cfg.target_tok_embd,
                            I->mask_emb.data(),
                            (size_t)cfg.mask_tok_id * I->hidden_size * sizeof(float),
                            (size_t)I->hidden_size * sizeof(float));

    // -----------------------------------------------------------------------
    // 5. CPU backend + gallocr for the per-step logit projection
    // -----------------------------------------------------------------------
    I->target_output = cfg.target_output;

    I->backend = ggml_backend_cpu_init();
    if (!I->backend) {
        std::fprintf(stderr, "[sp-native] failed to init CPU backend\n");
        return nullptr;
    }
    I->allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!I->allocr) {
        std::fprintf(stderr, "[sp-native] failed to create gallocr\n");
        return nullptr;
    }
    I->ctx_mem.resize(kNativeCtxSize);

    // -----------------------------------------------------------------------
    // 6. K-corr floor + scratch; working buffers
    // -----------------------------------------------------------------------
    I->kcorr_floor = sp_kcorr_floor(35.0f, 4, 0.03f);
    I->kcorr_scratch.resize((size_t)I->hidden_size * 2);

    I->vilenkin_skel_scratch.resize(SP_FLASH_N_SKEL);
    I->vilenkin_pred_scratch.resize((size_t)I->hidden_size);
    I->ctx_vec.resize((size_t)I->hidden_size);
    I->proposal_emb.resize((size_t)bs * I->hidden_size);
    I->noise_emb.resize((size_t)bs * I->hidden_size);

    I->adaptive      = cfg.adaptive;
    I->adaptive_hi   = cfg.adaptive_hi;
    I->adaptive_lo   = cfg.adaptive_lo;
    I->adaptive_steps= cfg.adaptive_steps;

    std::fprintf(stderr,
        "[sp-native] ready: kcorr_floor=%.4f block=%d adaptive=%s\n",
        I->kcorr_floor, I->block_size, I->adaptive ? "on" : "off");

    return oracle;
}

void SpFlashOracleNative::bind_capture(SpFlashCaptureCtx* ctx) {
    impl_->capture = ctx;
}

float SpFlashOracleNative::acceptance_rate()    const { return impl_->ema_accept; }
int   SpFlashOracleNative::current_block_size() const { return impl_->cur_block_size; }
const std::vector<int>& SpFlashOracleNative::target_layer_ids() const { return impl_->layer_ids; }

// ---------------------------------------------------------------------------
// step()
// ---------------------------------------------------------------------------

std::vector<int32_t> SpFlashOracleNative::step(
    const std::function<std::vector<float>(const std::vector<int32_t>&)>& target_logits_fn,
    int32_t /*last_token*/)
{
    Impl* I = impl_.get();

    if (!I->capture) {
        std::fprintf(stderr, "[sp-native] step(): capture context not bound\n");
        return {};
    }
    SpFlashCaptureCtx* cap = I->capture;

    const int hidden = I->hidden_size;
    const int n_tl   = I->num_target_layers;
    const int bs     = I->cur_block_size;

    // -----------------------------------------------------------------------
    // 1. Collect last-token hidden states from captured target layers → [n_tl, hidden]
    // -----------------------------------------------------------------------
    const int n_embd_cap = cap->n_embd > 0 ? cap->n_embd : hidden;
    std::vector<float> target_hiddens((size_t)n_tl * hidden, 0.0f);

    if ((int)cap->captured.size() < n_tl) {
        std::fprintf(stderr, "[sp-native] step(): only %d captured, expected %d\n",
                     (int)cap->captured.size(), n_tl);
        return {};
    }
    for (int ti = 0; ti < n_tl; ++ti) {
        const auto& cv = cap->captured[ti];
        if (cv.empty()) {
            std::fprintf(stderr, "[sp-native] step(): captured[%d] empty\n", ti);
            return {};
        }
        const int n_tok = (int)(cv.size() / (size_t)n_embd_cap);
        const int last  = (n_tok - 1) * n_embd_cap;
        const int dim   = std::min(hidden, n_embd_cap);
        std::memcpy(target_hiddens.data() + (size_t)ti * hidden,
                    cv.data() + last,
                    (size_t)dim * sizeof(float));
    }

    // -----------------------------------------------------------------------
    // 2. Vilenkin context: average Vilenkin-expanded skeleton across target layers.
    //    Result: ctx_vec [hidden] — the SP-native substitute for the FC projection.
    // -----------------------------------------------------------------------
    sp_flash_vilenkin_context(
        I->preds.data(),
        target_hiddens.data(),
        I->ctx_vec.data(),
        I->vilenkin_skel_scratch.data(),
        I->vilenkin_pred_scratch.data(),
        n_tl,
        hidden);

    // -----------------------------------------------------------------------
    // 3. Broadcast ctx_vec as proposal_emb: proposal_emb[i] = ctx_vec for all i.
    //    Squarefree positions will receive ctx_vec; prime-squared positions get
    //    mask_emb instead (applied by sp_flash_mobius_noise below).
    // -----------------------------------------------------------------------
    for (int i = 0; i < bs; ++i) {
        std::memcpy(I->proposal_emb.data() + (size_t)i * hidden,
                    I->ctx_vec.data(),
                    (size_t)hidden * sizeof(float));
    }

    // -----------------------------------------------------------------------
    // 4. Möbius noise schedule → noise_emb [bs * hidden]
    // -----------------------------------------------------------------------
    sp_flash_mobius_noise(
        I->noise_emb.data(),
        I->mask_emb.data(),
        I->proposal_emb.data(),
        I->mu_block.data(),
        bs,
        hidden);

    // -----------------------------------------------------------------------
    // 5. Logit projection: noise_emb [bs, hidden] → logits [bs, vocab]
    //    Tiny ggml graph: logits = target_output^T × noise_input
    //    (target_output is [hidden, vocab] col-major; mul_mat gives [vocab, bs])
    // -----------------------------------------------------------------------
    const int n_vocab = (int)I->target_output->ne[1];
    assert(n_vocab > hidden);

    ggml_init_params gip = {};
    gip.mem_size   = kNativeCtxSize;
    gip.mem_buffer = I->ctx_mem.data();
    gip.no_alloc   = true;
    ggml_context* gctx = ggml_init(gip);
    if (!gctx) {
        std::fprintf(stderr, "[sp-native] step(): ggml_init failed\n");
        return {};
    }

    ggml_tensor* noise_input = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hidden, bs);
    ggml_set_name(noise_input, "noise_input");
    ggml_set_input(noise_input);

    ggml_tensor* logits = ggml_mul_mat(gctx, I->target_output, noise_input);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    ggml_cgraph* graph = ggml_new_graph_custom(gctx, 32, false);
    ggml_build_forward_expand(graph, logits);

    if (!ggml_gallocr_alloc_graph(I->allocr, graph)) {
        std::fprintf(stderr, "[sp-native] step(): gallocr_alloc_graph failed\n");
        ggml_free(gctx);
        return {};
    }

    // noise_emb is [bs, hidden] row-major == ggml [hidden, bs] col-major (same bytes).
    ggml_backend_tensor_set(noise_input, I->noise_emb.data(), 0,
                            (size_t)bs * hidden * sizeof(float));

    if (ggml_backend_graph_compute(I->backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[sp-native] step(): graph compute failed\n");
        ggml_free(gctx);
        return {};
    }

    std::vector<float> draft_logits((size_t)bs * n_vocab);
    ggml_backend_tensor_get(logits, draft_logits.data(), 0,
                            draft_logits.size() * sizeof(float));
    ggml_free(gctx);

    // -----------------------------------------------------------------------
    // 6. Greedy draft tokens
    // -----------------------------------------------------------------------
    std::vector<int32_t> draft_tokens(bs);
    for (int i = 0; i < bs; ++i) {
        draft_tokens[i] = argmax_native(draft_logits.data() + (size_t)i * n_vocab, n_vocab);
    }

    // -----------------------------------------------------------------------
    // 7. Verify with target model
    // -----------------------------------------------------------------------
    std::vector<float> target_logits = target_logits_fn(draft_tokens);
    if ((int)target_logits.size() != bs * n_vocab) {
        std::fprintf(stderr,
            "[sp-native] step(): target_logits_fn returned %zu, expected %d\n",
            target_logits.size(), bs * n_vocab);
        return {};
    }

    // -----------------------------------------------------------------------
    // 8. K-corr + exact-match acceptance
    //    Compare noise_emb[i] against ctx_vec:
    //    - squarefree positions: noise_emb[i] == ctx_vec → K-corr = 1.0, always accept
    //    - prime-squared positions: noise_emb[i] == mask_emb → lower K-corr, may reject
    // -----------------------------------------------------------------------
    std::vector<int32_t> accepted = accept_native(
        draft_tokens, target_logits, n_vocab,
        I->noise_emb, I->ctx_vec, hidden,
        I->kcorr_floor, I->kcorr_scratch.data());

    // -----------------------------------------------------------------------
    // 9. EMA acceptance tracking
    // -----------------------------------------------------------------------
    const bool all_matched = ((int)accepted.size() == bs);
    const int  accept_len  = all_matched ? bs : (int)accepted.size() - 1;
    const float ratio      = (float)accept_len / (float)bs;
    ++I->step_count;
    const float alpha = (I->step_count == 1) ? 1.0f : (2.0f / (I->adaptive_steps + 1));
    I->ema_accept = I->ema_accept * (1.0f - alpha) + ratio * alpha;

    return accepted;
}

} // namespace sp::engine
